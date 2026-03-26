use tokio::net::UnixStream;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, self};

const SOCKET_PATH: &str = "/tmp/hyzero.sock";

#[tokio::main]
async fn main() {
    let stream = UnixStream::connect(SOCKET_PATH)
        .await
        .expect("Failed to connect to server. Is the server running?");

    let (reader, mut writer) = stream.into_split();
    let mut server_reader = BufReader::new(reader);
    let stdin = io::BufReader::new(io::stdin());
    let mut stdin_lines = stdin.lines();

    // Read and display initial messages (COLOR + YOUR_TURN/WAIT)
    let mut line = String::new();
    server_reader.read_line(&mut line).await.expect("Failed to read from server");
    print!("{}", line);
    let my_color = if line.contains("white") { "White" } else { "Black" };

    loop {
        line.clear();
        let n = server_reader.read_line(&mut line).await.expect("Failed to read from server");
        if n == 0 {
            println!("Server disconnected.");
            break;
        }

        // Server may send multiple lines in one notification (e.g., OPPONENT_MOVED + YOUR_TURN)
        for msg in line.trim().split('\n') {
            let msg = msg.trim();
            if msg.is_empty() { continue; }

            if msg.starts_with("YOUR_TURN") {
                // Prompt for move
                loop {
                    eprint!("[{}] Your move: ", my_color);
                    match stdin_lines.next_line().await {
                        Ok(Some(input)) => {
                            let input = input.trim().to_string();
                            if input.is_empty() { continue; }
                            let _ = writer.write_all(format!("MOVE {}\n", input).as_bytes()).await;
                            break;
                        }
                        _ => {
                            println!("stdin closed.");
                            return;
                        }
                    }
                }
            } else if msg.starts_with("OK ") {
                println!("Move accepted: {}", &msg[3..]);
            } else if msg.starts_with("INVALID ") {
                println!("Invalid: {}", &msg[8..]);
                // Re-prompt
                loop {
                    eprint!("[{}] Your move: ", my_color);
                    match stdin_lines.next_line().await {
                        Ok(Some(input)) => {
                            let input = input.trim().to_string();
                            if input.is_empty() { continue; }
                            let _ = writer.write_all(format!("MOVE {}\n", input).as_bytes()).await;
                            break;
                        }
                        _ => return,
                    }
                }
            } else if msg.starts_with("OPPONENT_MOVED ") {
                println!("Opponent played: {}", &msg[15..]);
            } else if msg.starts_with("BOARD ") {
                println!("Board: {}", &msg[6..]);
            } else if msg.starts_with("GAME_OVER ") {
                println!("Game over: {}", &msg[10..]);
                return;
            } else if msg == "WAIT" {
                println!("Waiting for opponent...");
            } else {
                println!("Server: {}", msg);
            }
        }
    }
}
