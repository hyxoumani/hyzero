//! Archive-pool enumeration for the dual-model evaluation ladder.
//!
//! Scans `checkpoints/best_v{NNN}.pt`, parses versions, returns the newest
//! `k` entries excluding the current champion's own version.

use std::path::{Path, PathBuf};

/// Return up to `k` newest archived champion checkpoints from `checkpoints_dir`,
/// excluding any entry whose parsed version equals `exclude_version`.
///
/// Mirrors the filename-parsing logic of `find_latest_archive_version` in
/// `src/bin/selfplay.rs` (strip prefix `best_v`, strip suffix `.pt`, parse `u64`).
///
/// Returns an empty vec if the directory is missing, unreadable, or contains
/// no matching files — never panics.
pub fn latest_archive_versions(
    checkpoints_dir: &Path,
    exclude_version: u64,
    k: usize,
) -> Vec<(u64, PathBuf)> {
    let dir = match std::fs::read_dir(checkpoints_dir) {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };

    let mut entries: Vec<(u64, PathBuf)> = Vec::new();
    for entry in dir.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if let Some(inner) = name_str.strip_prefix("best_v") {
            if let Some(num_str) = inner.strip_suffix(".pt") {
                if let Ok(v) = num_str.parse::<u64>() {
                    if v == exclude_version {
                        continue;
                    }
                    entries.push((v, entry.path()));
                }
            }
        }
    }

    // Newest-first.
    entries.sort_by(|a, b| b.0.cmp(&a.0));
    entries.truncate(k);
    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use tempfile::tempdir;

    #[test]
    fn returns_empty_on_missing_dir() {
        let p = PathBuf::from("/nonexistent/path/for/test/abc123");
        let got = latest_archive_versions(&p, 0, 3);
        assert!(got.is_empty());
    }

    #[test]
    fn returns_empty_when_no_matches() {
        let dir = tempdir().unwrap();
        for name in ["foo.pt", "best.pt", "best_v.pt", "best_vabc.pt"] {
            File::create(dir.path().join(name)).unwrap();
        }
        let got = latest_archive_versions(dir.path(), 0, 3);
        assert!(got.is_empty(), "expected empty, got {got:?}");
    }

    #[test]
    fn orders_newest_first() {
        let dir = tempdir().unwrap();
        for v in 1..=7 {
            File::create(dir.path().join(format!("best_v{v:03}.pt"))).unwrap();
        }
        let got = latest_archive_versions(dir.path(), 0, 3);
        let versions: Vec<u64> = got.iter().map(|(v, _)| *v).collect();
        assert_eq!(versions, vec![7, 6, 5]);
        for (v, p) in &got {
            assert!(
                p.to_string_lossy().ends_with(&format!("best_v{v:03}.pt")),
                "path {p:?} does not end with best_v{v:03}.pt"
            );
        }
    }

    #[test]
    fn excludes_current_version() {
        let dir = tempdir().unwrap();
        for v in 1..=7 {
            File::create(dir.path().join(format!("best_v{v:03}.pt"))).unwrap();
        }
        let got = latest_archive_versions(dir.path(), 7, 3);
        let versions: Vec<u64> = got.iter().map(|(v, _)| *v).collect();
        assert_eq!(versions, vec![6, 5, 4]);
    }

    #[test]
    fn truncates_to_k_when_more_available() {
        let dir = tempdir().unwrap();
        for v in 1..=5 {
            File::create(dir.path().join(format!("best_v{v:03}.pt"))).unwrap();
        }
        let got = latest_archive_versions(dir.path(), 0, 2);
        assert_eq!(got.len(), 2);
        let versions: Vec<u64> = got.iter().map(|(v, _)| *v).collect();
        assert_eq!(versions, vec![5, 4]);
    }

    #[test]
    fn returns_all_when_fewer_than_k() {
        let dir = tempdir().unwrap();
        File::create(dir.path().join("best_v001.pt")).unwrap();
        File::create(dir.path().join("best_v002.pt")).unwrap();
        let got = latest_archive_versions(dir.path(), 0, 3);
        assert_eq!(got.len(), 2);
        let versions: Vec<u64> = got.iter().map(|(v, _)| *v).collect();
        assert_eq!(versions, vec![2, 1]);
    }
}
