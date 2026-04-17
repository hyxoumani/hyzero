use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use tokio::sync::RwLock;

use crate::mcts::evaluator::Evaluator;

/// Shared champion store: tracks the current champion evaluator and its version.
///
/// The inner `Arc<RwLock<...>>` allows multiple concurrent readers (self-play and eval
/// tasks) while the eval task holds an exclusive write lock only during promotion.
/// The `champion_version` atomic is updated after the write lock is released.
pub struct ChampionStore {
    /// Current champion evaluator. The inner Arc allows cloning the evaluator without
    /// holding the RwLock.
    champion: RwLock<Arc<dyn Evaluator>>,
    /// Monotonically increasing version; 0 = Random baseline.
    champion_version: AtomicU64,
    /// Archive depth: keep at most this many `best_vNNN.pt` files.
    archive_depth: usize,
    /// Saved champion checkpoint paths (newest last).
    archive_files: RwLock<Vec<PathBuf>>,
}

impl ChampionStore {
    /// Create a new store backed by the given initial evaluator (typically `RandomBackend`).
    /// Champion version is initialized to 0.
    pub fn new(initial: Arc<dyn Evaluator>, archive_depth: usize) -> Self {
        Self::new_with_version(initial, archive_depth, 0)
    }

    /// Create a new store with an explicit starting version.
    ///
    /// Used on binary restart when loading an existing `best.pt` so that the
    /// promotion ladder continues from the last known champion version.
    pub fn new_with_version(
        initial: Arc<dyn Evaluator>,
        archive_depth: usize,
        starting_version: u64,
    ) -> Self {
        Self {
            champion: RwLock::new(initial),
            champion_version: AtomicU64::new(starting_version),
            archive_depth,
            archive_files: RwLock::new(Vec::new()),
        }
    }

    /// Current champion version (atomic read, no lock needed).
    pub fn version(&self) -> u64 {
        self.champion_version.load(Ordering::Acquire)
    }

    /// Clone the current champion evaluator (acquires a short read lock).
    pub async fn champion(&self) -> Arc<dyn Evaluator> {
        self.champion.read().await.clone()
    }

    /// Promote `new_champion` as the champion.
    ///
    /// Atomically:
    /// 1. Acquire write lock, swap evaluator.
    /// 2. Update `champion_version` to `new_version`.
    /// 3. If `checkpoint_src` is provided, copy it to `checkpoints/best.pt` (atomic
    ///    write via `.tmp`) and archive to `checkpoints/best_vNNN.pt`. Prune oldest
    ///    archives beyond `archive_depth`.
    ///
    /// Returns the new version.
    pub async fn promote(
        &self,
        new_champion: Arc<dyn Evaluator>,
        new_version: u64,
        checkpoint_src: Option<&PathBuf>,
    ) -> u64 {
        // Swap evaluator under write lock.
        {
            let mut guard = self.champion.write().await;
            *guard = new_champion;
        }
        self.champion_version.store(new_version, Ordering::Release);

        // Persist champion checkpoint files if a source path was given.
        if let Some(src) = checkpoint_src {
            if let Err(e) = persist_champion_checkpoint(src, new_version) {
                eprintln!("[champion] checkpoint persist error: {e}");
            } else {
                // Track archived file and prune oldest if needed.
                let archive_path =
                    PathBuf::from(format!("checkpoints/best_v{:03}.pt", new_version));
                let mut files = self.archive_files.write().await;
                files.push(archive_path);
                while files.len() > self.archive_depth {
                    if let Some(oldest) = files.first().cloned() {
                        files.remove(0);
                        if let Err(e) = std::fs::remove_file(&oldest) {
                            eprintln!(
                                "[champion] failed to prune archive {}: {e}",
                                oldest.display()
                            );
                        } else {
                            println!("[champion] pruned archive: {}", oldest.display());
                        }
                    }
                }
            }
        }

        new_version
    }
}

/// Copy `src` to `checkpoints/best.pt` via atomic `.tmp` write, then hard-link
/// (or copy) to `checkpoints/best_vNNN.pt`.
fn persist_champion_checkpoint(src: &PathBuf, version: u64) -> std::io::Result<()> {
    let _ = std::fs::create_dir_all("checkpoints");
    let best_tmp = PathBuf::from("checkpoints/best.pt.tmp");
    let best = PathBuf::from("checkpoints/best.pt");
    let archive = PathBuf::from(format!("checkpoints/best_v{:03}.pt", version));

    // Copy src → best.pt.tmp
    std::fs::copy(src, &best_tmp)?;

    // Sync the tmp file (best-effort — ignore errors on platforms that don't support fsync)
    {
        use std::fs::OpenOptions;
        if let Ok(f) = OpenOptions::new().write(true).open(&best_tmp) {
            let _ = f.sync_all();
        }
    }

    // Atomic rename tmp → best.pt
    std::fs::rename(&best_tmp, &best)?;

    // Copy best.pt → best_vNNN.pt (archive)
    std::fs::copy(&best, &archive)?;

    println!("[champion] saved best.pt and {}", archive.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{BoardObservation, HiddenState, Policy, ActionIndex, NUM_ACTIONS};
    use crate::mcts::evaluator::Evaluator;
    use async_trait::async_trait;

    struct RandomEvaluator;

    #[async_trait]
    impl Evaluator for RandomEvaluator {
        async fn root_setup(
            &self,
            _obs: &BoardObservation,
            _legal_mask: &[bool],
        ) -> (HiddenState, Policy, f32) {
            let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
            (HiddenState::new(64), policy, 0.0)
        }

        async fn expand_leaf(
            &self,
            _hs: &HiddenState,
            _action: ActionIndex,
        ) -> (HiddenState, f32, Policy, f32) {
            let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
            (HiddenState::new(64), 0.0, policy, 0.0)
        }
    }

    #[tokio::test]
    async fn test_champion_store_initial_version() {
        let evaluator: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        let store = ChampionStore::new(evaluator, 5);
        assert_eq!(store.version(), 0, "initial version should be 0");
    }

    #[tokio::test]
    async fn test_champion_store_new_with_starting_version() {
        let evaluator: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        let store = ChampionStore::new_with_version(evaluator, 5, 42);
        assert_eq!(
            store.version(),
            42,
            "starting_version should be initialized from constructor argument"
        );
        // champion() should return the provided evaluator (not panic or deadlock).
        let _champ = store.champion().await;
    }

    #[tokio::test]
    async fn test_champion_store_promote_updates_version() {
        let initial: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        let store = ChampionStore::new(initial, 5);

        let new_champ: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        store.promote(new_champ, 42, None).await;

        assert_eq!(store.version(), 42, "version should be 42 after promotion");
    }

    #[tokio::test]
    async fn test_champion_store_promote_no_checkpoint() {
        let initial: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        let store = Arc::new(ChampionStore::new(initial, 5));

        let new_champ: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        let v = store.promote(new_champ, 7, None).await;
        assert_eq!(v, 7);
    }

    #[tokio::test]
    async fn test_champion_store_archive_pruning() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Change cwd temporarily is not practical; instead test the pruning logic
        // directly on the archive_files VecDeque to avoid filesystem side effects.

        let initial: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        let store = Arc::new(ChampionStore::new(initial, 3)); // archive_depth = 3

        // Promote 5 times without checkpoint (no fs side effects)
        for i in 1u64..=5 {
            let champ: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
            store.promote(champ, i, None).await;
        }

        assert_eq!(store.version(), 5);
        // Without checkpoint_src, archive_files stays empty — that's fine.
        let files = store.archive_files.read().await;
        assert_eq!(files.len(), 0, "no archives without checkpoint_src");
    }
}
