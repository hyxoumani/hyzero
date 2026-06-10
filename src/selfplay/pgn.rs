//! PGN file writer for hyzero game logging.
//!
//! Used by eval ladder (always) and self-play (sampled at low rate).

use std::io::Write;

/// Write a single game to `path` in standard PGN format.
/// Caller chooses the path to keep eval and self-play PGN files separate.
///
/// `termination` records HOW the game ended (e.g. "checkmate", "stalemate",
/// "repetition", "fifty-move", "move-cap", "resignation", "adjudication"). It is
/// emitted as an additive `[Termination "..."]` header; downstream parsers split
/// on `[Event` and read headers generically, so this is backward compatible.
///
/// `initial_fen` is `Some(fen)` only when the game started from a non-standard
/// position (diverse self-play starts via `HYZERO_STARTS_FILE`). When set, the
/// standard `[SetUp "1"]` and `[FEN "<fen>"]` headers are emitted so a viewer can
/// replay the UCI moves from that position instead of the standard start. When
/// `None` (standard start) the headers are omitted, keeping standard-start PGNs
/// byte-identical to the prior format.
pub fn write_pgn_game(
    path: &str,
    event: &str,
    white_label: &str,
    black_label: &str,
    result: &str,
    termination: &str,
    initial_fen: Option<&str>,
    moves: &[String],
) {
    // Ensure parent directory exists (usually "logs/")
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let mut file = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[pgn] failed to open {path}: {e}");
            return;
        }
    };

    writeln!(file, "[Event \"{event}\"]").ok();
    writeln!(file, "[White \"{white_label}\"]").ok();
    writeln!(file, "[Black \"{black_label}\"]").ok();
    writeln!(file, "[Result \"{result}\"]").ok();
    writeln!(file, "[Termination \"{termination}\"]").ok();
    // Additive standard PGN headers for a non-standard start. Emitted only for
    // diverse-start games so standard-start PGNs stay byte-identical.
    if let Some(fen) = initial_fen {
        writeln!(file, "[SetUp \"1\"]").ok();
        writeln!(file, "[FEN \"{fen}\"]").ok();
    }
    writeln!(file).ok();

    let mut line = String::new();
    for (i, m) in moves.iter().enumerate() {
        if i % 2 == 0 {
            line.push_str(&format!("{}. ", i / 2 + 1));
        }
        line.push_str(m);
        line.push(' ');
        if line.len() > 75 {
            writeln!(file, "{}", line.trim()).ok();
            line.clear();
        }
    }
    if !line.is_empty() {
        line.push_str(result);
        writeln!(file, "{}", line.trim()).ok();
    } else {
        writeln!(file, "{result}").ok();
    }
    writeln!(file).ok();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The PGN writer must emit an additive `[Termination "..."]` header carrying
    /// the supplied cause, without disturbing the existing Event/White/Black/Result
    /// headers. FAILS before the header was added (the Termination line is absent).
    #[test]
    fn pgn_records_resignation_termination() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hyzero_pgn_term_{}.pgn", std::process::id()));
        let path_str = path.to_str().expect("utf8 temp path");
        let _ = std::fs::remove_file(&path);

        write_pgn_game(
            path_str,
            "Selfplay model_v1",
            "selfplay_white",
            "selfplay_black",
            "1-0",
            "resignation",
            None,
            &["e2e4".to_string(), "e7e5".to_string()],
        );

        let contents = std::fs::read_to_string(&path).expect("pgn written");
        let _ = std::fs::remove_file(&path);

        assert!(
            contents.contains("[Termination \"resignation\"]"),
            "expected resignation Termination header, got:\n{contents}"
        );
        // Existing headers must be preserved verbatim.
        assert!(contents.contains("[Event \"Selfplay model_v1\"]"));
        assert!(contents.contains("[Result \"1-0\"]"));
    }

    /// A game that started from a non-standard position must record the standard
    /// `[SetUp "1"]` and `[FEN "..."]` headers carrying that start, so a viewer can
    /// replay the UCI moves from it. FAILS before the headers were added (the FEN
    /// line is absent and the viewer replays from the standard start -> "0 / 0").
    #[test]
    fn pgn_records_fen_header_for_custom_start() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hyzero_pgn_fen_{}.pgn", std::process::id()));
        let path_str = path.to_str().expect("utf8 temp path");
        let _ = std::fs::remove_file(&path);

        let start = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1";
        write_pgn_game(
            path_str,
            "Eval Cycle 1 Game 1",
            "challenger",
            "champion",
            "1-0",
            "checkmate",
            Some(start),
            &["e2e4".to_string()],
        );

        let contents = std::fs::read_to_string(&path).expect("pgn written");
        let _ = std::fs::remove_file(&path);

        assert!(
            contents.contains("[SetUp \"1\"]") && contents.contains(&format!("[FEN \"{start}\"]")),
            "expected SetUp/FEN headers for a custom start, got:\n{contents}"
        );
    }

    /// A standard-start game must omit the SetUp/FEN headers entirely, so existing
    /// standard-start PGNs stay byte-identical to the prior format.
    #[test]
    fn pgn_omits_fen_header_for_standard_start() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("hyzero_pgn_nofen_{}.pgn", std::process::id()));
        let path_str = path.to_str().expect("utf8 temp path");
        let _ = std::fs::remove_file(&path);

        write_pgn_game(
            path_str,
            "Selfplay model_v1",
            "selfplay_white",
            "selfplay_black",
            "1/2-1/2",
            "stalemate",
            None,
            &["e2e4".to_string(), "e7e5".to_string()],
        );

        let contents = std::fs::read_to_string(&path).expect("pgn written");
        let _ = std::fs::remove_file(&path);

        assert!(
            !contents.contains("[SetUp") && !contents.contains("[FEN"),
            "standard-start game must omit SetUp/FEN headers, got:\n{contents}"
        );
    }
}
