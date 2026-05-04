use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::data::ReplayFile;

/// Process-wide counter that disambiguates replays written in the same second.
static REPLAY_SEQ: AtomicU64 = AtomicU64::new(0);

/// Serialize `replay` to `<dir>/replay_<unix_secs>_<seq>_v<model_version>.replay`
/// using bincode. Creates `dir` if it does not exist. Returns the written path.
pub fn write_replay(replay: &ReplayFile, dir: &Path) -> io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let seq = REPLAY_SEQ.fetch_add(1, Ordering::Relaxed);
    let filename = format!(
        "replay_{secs}_{seq:06}_v{ver}.replay",
        ver = replay.model_version
    );
    let path = dir.join(filename);
    let bytes = bincode::serialize(replay).map_err(io::Error::other)?;
    std::fs::write(&path, bytes)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{ReplayFile, ReplayRecord};

    fn sample_replay() -> ReplayFile {
        ReplayFile {
            steps: vec![ReplayRecord {
                action: 796,
                legal_moves: vec![405, 731, 796],
                child_visits: vec![3, 5, 42],
                priors: vec![0.1, 0.2, 0.7],
                q_values: vec![0.0, 0.1, 0.3],
                root_value: 0.25,
                white_to_move: true,
            }],
            game_outcome: 1.0,
            model_version: 7,
            is_draw: false,
            starting_fen: None,
            c_puct: 1.5,
        }
    }

    #[test]
    fn write_then_load_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let original = sample_replay();
        let path = write_replay(&original, dir.path()).expect("write");
        assert!(path.exists());

        let bytes = std::fs::read(&path).unwrap();
        let loaded: ReplayFile = bincode::deserialize(&bytes).unwrap();

        assert_eq!(loaded.steps.len(), 1);
        assert_eq!(loaded.steps[0].action, 796);
        assert_eq!(loaded.steps[0].child_visits, vec![3, 5, 42]);
        assert_eq!(loaded.steps[0].priors, vec![0.1, 0.2, 0.7]);
        assert_eq!(loaded.model_version, 7);
        assert!((loaded.c_puct - 1.5).abs() < 1e-6);
        assert!(loaded.starting_fen.is_none());
    }

    #[test]
    fn filename_includes_model_version() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = sample_replay();
        r.model_version = 42;
        let path = write_replay(&r, dir.path()).expect("write");
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.contains("v42"), "filename {name} missing v42");
        assert!(name.ends_with(".replay"));
    }
}
