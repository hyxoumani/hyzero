use hyzero::game::board::GameResult;
use hyzero::game::history::GameHistory;
use hyzero::game::playerobj::Player;
use hyzero::game::board::GameBoard;
use hyzero::PrecomputedItems;
use hyzero::Color;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use tokio::net::{UnixListener, UnixStream};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

const SOCKET_PATH: &str = "/tmp/hyzero.sock";

struct SharedState {
    game_board: GameBoard,
    history: GameHistory,
    turn_count: usize,
}

#[tokio::main]
async fn main() {
    // Clean up stale socket
    let _ = std::fs::remove_file(SOCKET_PATH);

    println!("Computing precomputed items...");
    let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
    let player1 = Player::init_player(true);
    let player2 = Player::init_player(false);
    let game_board = GameBoard::init_game_board(precomputed.clone(), player1, player2);

    let state = Arc::new(Mutex::new(SharedState {
        game_board,
        history: GameHistory::new(),
        turn_count: 0,
    }));

    let listener = UnixListener::bind(SOCKET_PATH).expect("Failed to bind Unix socket");
    println!("Server listening on {}", SOCKET_PATH);

    // Accept White player
    println!("Waiting for White player...");
    let (white_stream, _) = listener.accept().await.expect("Failed to accept White");
    println!("White player connected.");

    // Accept Black player
    println!("Waiting for Black player...");
    let (black_stream, _) = listener.accept().await.expect("Failed to accept Black");
    println!("Black player connected.");

    // Channels: server notifies each client when it's their turn or opponent moved
    let (white_tx, white_rx) = mpsc::channel::<String>(16);
    let (black_tx, black_rx) = mpsc::channel::<String>(16);

    let s1 = state.clone();
    let white_handle = tokio::spawn(handle_client(
        white_stream, Color::White, s1, white_rx, black_tx.clone(),
    ));

    let s2 = state.clone();
    let black_handle = tokio::spawn(handle_client(
        black_stream, Color::Black, s2, black_rx, white_tx.clone(),
    ));

    // Signal White to start
    let _ = white_tx.send("YOUR_TURN\n".to_string()).await;

    let _ = tokio::join!(white_handle, black_handle);

    // Print move history
    let gs = state.lock().await;
    println!("\n--- Move History ---");
    for m in &gs.history.move_history {
        println!("{}", m);
    }

    let _ = std::fs::remove_file(SOCKET_PATH);
    println!("Server shut down.");
}

async fn handle_client(
    stream: UnixStream,
    color: Color,
    state: Arc<Mutex<SharedState>>,
    mut notify_rx: mpsc::Receiver<String>,
    opponent_tx: mpsc::Sender<String>,
) {
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);

    // Send color assignment
    let color_str = if color == Color::White { "white" } else { "black" };
    let _ = writer.write_all(format!("COLOR {}\n", color_str).as_bytes()).await;

    // If black, tell them to wait
    if color == Color::Black {
        let _ = writer.write_all(b"WAIT\n").await;
    }

    loop {
        tokio::select! {
            // Notification from opponent's handler
            msg = notify_rx.recv() => {
                match msg {
                    Some(notification) => {
                        let _ = writer.write_all(notification.as_bytes()).await;
                        if notification.starts_with("GAME_OVER") {
                            break;
                        }
                    }
                    None => break, // Channel closed
                }
            }
            // Input from this client
            line = async {
                let mut buf = String::new();
                let n = reader.read_line(&mut buf).await;
                (buf, n)
            } => {
                let (line, n) = line;
                if n.unwrap_or(0) == 0 {
                    println!("{:?} disconnected.", color);
                    break;
                }
                let line = line.trim().to_string();

                if !line.starts_with("MOVE ") {
                    let _ = writer.write_all(b"INVALID bad command\n").await;
                    continue;
                }
                let notation = line[5..].trim();


                let mut gs = state.lock().await;
                let expected_color = if gs.turn_count % 2 == 0 { Color::White } else { Color::Black };

                if expected_color != color {
                    let _ = writer.write_all(b"INVALID not your turn\n").await;
                    drop(gs);
                    continue;
                }

                let current_turn = gs.turn_count;
                match gs.game_board.process_move(notation, color, current_turn) {
                    Ok((_mv, result)) => {
                        let prefix = if color == Color::White { "W" } else { "B" };
                        let snapshot = gs.game_board.board_snapshot();
                        gs.history.record_move(prefix, notation, snapshot);
                        gs.turn_count += 1;

                        let board_msg = format!("BOARD {}\n", gs.game_board.bitboard_string());

                        println!("[Move {}] {}: {}", gs.turn_count, prefix, notation);
                        println!("  {}", gs.game_board.bitboard_string());

                        let _ = writer.write_all(format!("OK {}\n", notation).as_bytes()).await;
                        let _ = writer.write_all(board_msg.as_bytes()).await;

                        if result != GameResult::Ongoing {
                            let result_msg = format!("GAME_OVER {:?}\n", result);
                            let _ = writer.write_all(result_msg.as_bytes()).await;
                            let _ = opponent_tx.send(format!("OPPONENT_MOVED {}\n{}{}", notation, board_msg, result_msg)).await;
                            break;
                        }

                        let _ = opponent_tx.send(format!("OPPONENT_MOVED {}\n{}YOUR_TURN\n", notation, board_msg)).await;
                    }
                    Err(reason) => {
                        let _ = writer.write_all(format!("INVALID {}\n", reason).as_bytes()).await;
                    }
                }
                drop(gs);
            }
        }
    }
}
