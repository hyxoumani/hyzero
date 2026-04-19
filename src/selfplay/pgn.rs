//! PGN file writer for hyzero game logging.
//!
//! Used by eval ladder (always) and self-play (sampled at low rate).

use std::io::Write;

/// Write a single game to `path` in standard PGN format.
/// Caller chooses the path to keep eval and self-play PGN files separate.
pub fn write_pgn_game(
    path: &str,
    event: &str,
    white_label: &str,
    black_label: &str,
    result: &str,
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
