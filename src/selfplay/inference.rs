use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};
use tokio::time::{timeout, Duration};

use crate::data::{BoardObservation, HiddenState, Policy, ActionIndex, NUM_ACTIONS};
use crate::mcts::evaluator::Evaluator;

/// Recoverable failure when a `ChannelEvaluator` cannot complete a request: either
/// the request channel to the batcher is closed (all receivers dropped) or the
/// reply oneshot was dropped before a result was produced. Both arise when the
/// backing `InferenceBatcher` task has stopped — historically these were
/// `.expect()` panics that, inside a spawned eval task, killed the task silently
/// and wedged the evaluation ladder. Callers now recover from this instead of
/// panicking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalError {
    /// The request channel to the batcher is closed (batcher task has exited).
    ChannelClosed,
    /// The reply oneshot was dropped before the batcher produced a result.
    ReplyDropped,
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::ChannelClosed => write!(f, "inference channel closed"),
            EvalError::ReplyDropped => write!(f, "inference reply dropped"),
        }
    }
}

impl std::error::Error for EvalError {}

/// Request sent from game tasks to the inference batcher.
pub enum InferenceRequest {
    RootSetup {
        observation: BoardObservation,
        /// Boolean mask of length NUM_ACTIONS; `true` means the action is legal.
        legal_mask: Vec<bool>,
        reply: oneshot::Sender<(HiddenState, Policy, f32)>,
    },
    ExpandLeaf {
        hidden_state: HiddenState,
        action: ActionIndex,
        reply: oneshot::Sender<(HiddenState, f32, Policy, f32)>,
    },
}

/// Abstraction over the neural network backend.
/// Implementations process a batch of requests (real PyO3 or random stub).
pub trait InferenceBackend: Send {
    fn evaluate_batch(&mut self, requests: Vec<InferenceRequest>);
}

/// Stub backend that returns random hidden states, uniform policies, and 0.0 values.
pub struct RandomBackend {
    pub hidden_channels: usize,
}

impl RandomBackend {
    pub fn new(hidden_channels: usize) -> Self {
        Self { hidden_channels }
    }
}

impl InferenceBackend for RandomBackend {
    fn evaluate_batch(&mut self, requests: Vec<InferenceRequest>) {
        let uniform_policy: Policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];

        for req in requests {
            match req {
                InferenceRequest::RootSetup { reply, .. } => {
                    let hs = HiddenState::new(self.hidden_channels);
                    let _ = reply.send((hs, uniform_policy.clone(), 0.0));
                }
                InferenceRequest::ExpandLeaf { reply, .. } => {
                    let hs = HiddenState::new(self.hidden_channels);
                    let _ = reply.send((hs, 0.0, uniform_policy.clone(), 0.0));
                }
            }
        }

    }
}

/// A backend that delegates to a hot-swappable inner backend.
///
/// The eval task calls `swap()` during promotion to replace the champion's
/// backend without restarting the batcher task. All other batch calls proceed
/// concurrently without interference (the Mutex is held only during swap and
/// for the duration of a single `evaluate_batch` call).
pub struct SwappableBackend {
    inner: Arc<Mutex<Box<dyn InferenceBackend>>>,
}

impl SwappableBackend {
    /// Create a new `SwappableBackend` wrapping `initial`.
    pub fn new(initial: Box<dyn InferenceBackend>) -> (Self, Arc<Mutex<Box<dyn InferenceBackend>>>) {
        let shared = Arc::new(Mutex::new(initial));
        let backend = SwappableBackend { inner: shared.clone() };
        (backend, shared)
    }
}

impl InferenceBackend for SwappableBackend {
    fn evaluate_batch(&mut self, requests: Vec<InferenceRequest>) {
        // Lock is released immediately after the call — safe for concurrent swaps.
        self.inner.lock().expect("SwappableBackend lock poisoned").evaluate_batch(requests);
    }
}

/// Configuration for the inference batcher.
#[derive(Debug, Clone)]
pub struct BatcherConfig {
    pub max_batch_size: usize,
    pub batch_timeout_ms: u64,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            batch_timeout_ms: 1,
        }
    }
}

/// Collects inference requests into batches, then dispatches to a backend.
/// Runs as a long-lived async task.
pub struct InferenceBatcher {
    rx: mpsc::Receiver<InferenceRequest>,
    backend: Box<dyn InferenceBackend>,
    config: BatcherConfig,
}

impl InferenceBatcher {
    pub fn new(
        rx: mpsc::Receiver<InferenceRequest>,
        backend: Box<dyn InferenceBackend>,
        config: BatcherConfig,
    ) -> Self {
        Self { rx, backend, config }
    }

    /// Run the batcher loop. Collects requests until batch is full or timeout fires.
    pub async fn run(&mut self) {
        loop {
            let mut batch: Vec<InferenceRequest> = Vec::with_capacity(self.config.max_batch_size);

            // Wait for the first request (blocking — no timeout)
            match self.rx.recv().await {
                Some(req) => batch.push(req),
                None => return, // Channel closed, all senders dropped
            }

            // Collect more requests up to batch size or timeout
            let deadline = Duration::from_millis(self.config.batch_timeout_ms);
            while batch.len() < self.config.max_batch_size {
                match timeout(deadline, self.rx.recv()).await {
                    Ok(Some(req)) => batch.push(req),
                    _ => break, // Timeout or channel closed
                }
            }

            self.backend.evaluate_batch(batch);
        }
    }
}

/// Evaluator that sends requests through a channel to the InferenceBatcher.
/// Implements the Evaluator trait so it can be used directly by MCTSTree.
///
/// If the backing batcher task has stopped (request channel closed or reply
/// oneshot dropped), the fallible `try_*` methods surface a recoverable
/// [`EvalError`]; the `Evaluator` trait methods (which must return concrete
/// tuples for the MCTS hot path) recover to a neutral result — a uniform policy
/// and zero value/reward — and log a one-time warning rather than panicking.
/// A spawned eval task therefore degrades the affected game instead of dying
/// silently and stranding the evaluation ladder.
#[derive(Clone)]
pub struct ChannelEvaluator {
    tx: mpsc::Sender<InferenceRequest>,
    /// Channel count for the neutral hidden state returned when `root_setup`
    /// recovers from a dropped batcher (expand_leaf reuses the input state's
    /// channel count, so this only backstops the root call).
    hidden_channels: usize,
    /// Set once a recovery has been logged so a dead batcher does not flood the
    /// log with one line per MCTS simulation.
    recovery_warned: Arc<AtomicBool>,
}

impl ChannelEvaluator {
    /// Create an evaluator with the default neutral hidden-state width (64).
    /// Prefer [`ChannelEvaluator::with_channels`] so recovery hidden states match
    /// the live model width; this constructor is kept for existing call sites and
    /// tests that do not exercise the recovery path.
    pub fn new(tx: mpsc::Sender<InferenceRequest>) -> Self {
        Self::with_channels(tx, 64)
    }

    /// Create an evaluator that, on recovery from a dropped batcher, returns a
    /// neutral root hidden state of `hidden_channels` channels.
    pub fn with_channels(tx: mpsc::Sender<InferenceRequest>, hidden_channels: usize) -> Self {
        Self {
            tx,
            hidden_channels,
            recovery_warned: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Fallible root setup: returns [`EvalError`] if the request channel is closed
    /// or the reply is dropped before the batcher answers. This is the recoverable
    /// path the `Evaluator` impl wraps.
    pub async fn try_root_setup(
        &self,
        observation: &BoardObservation,
        legal_mask: &[bool],
    ) -> Result<(HiddenState, Policy, f32), EvalError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let req = InferenceRequest::RootSetup {
            observation: observation.clone(),
            legal_mask: legal_mask.to_vec(),
            reply: reply_tx,
        };
        self.tx.send(req).await.map_err(|_| EvalError::ChannelClosed)?;
        reply_rx.await.map_err(|_| EvalError::ReplyDropped)
    }

    /// Fallible leaf expansion: returns [`EvalError`] if the request channel is
    /// closed or the reply is dropped before the batcher answers.
    pub async fn try_expand_leaf(
        &self,
        hidden_state: &HiddenState,
        action: ActionIndex,
    ) -> Result<(HiddenState, f32, Policy, f32), EvalError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let req = InferenceRequest::ExpandLeaf {
            hidden_state: hidden_state.clone(),
            action,
            reply: reply_tx,
        };
        self.tx.send(req).await.map_err(|_| EvalError::ChannelClosed)?;
        reply_rx.await.map_err(|_| EvalError::ReplyDropped)
    }

    /// Log the first recovery from a dropped batcher; subsequent recoveries on the
    /// same evaluator are silent so a dead batcher cannot spam the log.
    fn warn_recovery_once(&self, op: &str, err: EvalError) {
        if !self.recovery_warned.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[inference] WARN: {op} recovering with neutral result — {err} \
                 (backing batcher stopped); affected game/cycle is degraded, not wedged"
            );
        }
    }
}

#[async_trait]
impl Evaluator for ChannelEvaluator {
    async fn root_setup(&self, observation: &BoardObservation, legal_mask: &[bool]) -> (HiddenState, Policy, f32) {
        match self.try_root_setup(observation, legal_mask).await {
            Ok(result) => result,
            Err(err) => {
                self.warn_recovery_once("root_setup", err);
                let policy: Policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
                (HiddenState::new(self.hidden_channels), policy, 0.0)
            }
        }
    }

    async fn expand_leaf(&self, hidden_state: &HiddenState, action: ActionIndex) -> (HiddenState, f32, Policy, f32) {
        match self.try_expand_leaf(hidden_state, action).await {
            Ok(result) => result,
            Err(err) => {
                self.warn_recovery_once("expand_leaf", err);
                let policy: Policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
                // Reuse the input width so the neutral state matches the tree's
                // existing hidden-state shape.
                (HiddenState::new(hidden_state.channels), 0.0, policy, 0.0)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_root_setup_through_channel() {
        let (tx, rx) = mpsc::channel(32);
        let backend = Box::new(RandomBackend::new(64));
        let config = BatcherConfig { max_batch_size: 8, batch_timeout_ms: 10 };
        let mut batcher = InferenceBatcher::new(rx, backend, config);

        let evaluator = ChannelEvaluator::new(tx);

        // Spawn batcher in background
        let batcher_handle = tokio::spawn(async move { batcher.run().await });

        let obs = BoardObservation::default();
        let mask = vec![true; NUM_ACTIONS];
        let (hs, policy, value) = evaluator.root_setup(&obs, &mask).await;

        assert_eq!(hs.channels, 64);
        assert_eq!(hs.data.len(), 64 * 64);
        assert_eq!(policy.len(), NUM_ACTIONS);
        assert!((value - 0.0).abs() < f32::EPSILON);

        // Drop evaluator to close channel and stop batcher
        drop(evaluator);
        let _ = batcher_handle.await;
    }

    #[tokio::test]
    async fn test_expand_leaf_through_channel() {
        let (tx, rx) = mpsc::channel(32);
        let backend = Box::new(RandomBackend::new(64));
        let config = BatcherConfig { max_batch_size: 8, batch_timeout_ms: 10 };
        let mut batcher = InferenceBatcher::new(rx, backend, config);

        let evaluator = ChannelEvaluator::new(tx);
        let batcher_handle = tokio::spawn(async move { batcher.run().await });

        let hs_in = HiddenState::new(64);
        let (hs_out, reward, policy, value) = evaluator.expand_leaf(&hs_in, 42).await;

        assert_eq!(hs_out.channels, 64);
        assert_eq!(policy.len(), NUM_ACTIONS);
        assert!((reward - 0.0).abs() < f32::EPSILON);
        assert!((value - 0.0).abs() < f32::EPSILON);

        drop(evaluator);
        let _ = batcher_handle.await;
    }

    #[tokio::test]
    async fn test_batch_collects_multiple_requests() {
        use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

        let batch_count = Arc::new(AtomicUsize::new(0));
        let batch_count_clone = batch_count.clone();

        // Custom backend that tracks how many batches it receives
        struct CountingBackend {
            count: Arc<AtomicUsize>,
            channels: usize,
        }
        impl InferenceBackend for CountingBackend {
            fn evaluate_batch(&mut self, requests: Vec<InferenceRequest>) {
                self.count.fetch_add(1, Ordering::SeqCst);
                // Still need to reply so callers don't hang
                let policy: Policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
                for req in requests {
                    match req {
                        InferenceRequest::RootSetup { reply, .. } => {
                            let _ = reply.send((HiddenState::new(self.channels), policy.clone(), 0.0));
                        }
                        InferenceRequest::ExpandLeaf { reply, .. } => {
                            let _ = reply.send((HiddenState::new(self.channels), 0.0, policy.clone(), 0.0));
                        }
                    }
                }
            }
        }

        let (tx, rx) = mpsc::channel(64);
        let backend = Box::new(CountingBackend { count: batch_count_clone, channels: 64 });
        // Large timeout so requests accumulate
        let config = BatcherConfig { max_batch_size: 4, batch_timeout_ms: 50 };
        let mut batcher = InferenceBatcher::new(rx, backend, config);

        let batcher_handle = tokio::spawn(async move { batcher.run().await });

        // Send 4 requests concurrently — they should batch together
        let mut handles = Vec::new();
        for _ in 0..4 {
            let eval = ChannelEvaluator::new(tx.clone());
            handles.push(tokio::spawn(async move {
                let mask = vec![true; NUM_ACTIONS];
                eval.root_setup(&BoardObservation::default(), &mask).await
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        // With batch size 4 and 4 concurrent requests, should be 1-2 batches
        let batches = batch_count.load(Ordering::SeqCst);
        assert!((1..=2).contains(&batches), "Expected 1-2 batches, got {}", batches);

        drop(tx);
        let _ = batcher_handle.await;
    }

    /// Backend that drops every reply oneshot without answering, then signals it
    /// has seen a request. Used to reproduce the champion-promotion wedge: a
    /// batcher whose backend never replies (or, equivalently, one that has
    /// exited) must surface a recoverable error rather than parking the caller
    /// forever.
    struct DroppingBackend {
        seen: Arc<std::sync::atomic::AtomicBool>,
    }
    impl InferenceBackend for DroppingBackend {
        fn evaluate_batch(&mut self, requests: Vec<InferenceRequest>) {
            self.seen.store(true, std::sync::atomic::Ordering::SeqCst);
            // Drop each request (and thus its reply oneshot) without sending.
            drop(requests);
        }
    }

    /// REGRESSION (champion-promotion wedge, layer a): when the batcher's reply
    /// oneshot is dropped without an answer, `try_root_setup` must return
    /// `EvalError::ReplyDropped` — NOT panic and NOT hang. Pre-fix the production
    /// path used `.expect("inference reply dropped")`, which panicked inside the
    /// spawned eval task and stranded the ladder. The 5s timeout converts a
    /// regression (await that never completes) into a TEST failure, not a CI hang.
    #[tokio::test]
    async fn try_root_setup_returns_error_when_reply_dropped() {
        let seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (tx, rx) = mpsc::channel(8);
        let backend = Box::new(DroppingBackend { seen: seen.clone() });
        let config = BatcherConfig { max_batch_size: 1, batch_timeout_ms: 5 };
        let mut batcher = InferenceBatcher::new(rx, backend, config);
        let batcher_handle = tokio::spawn(async move { batcher.run().await });

        let evaluator = ChannelEvaluator::new(tx);
        let mask = vec![true; NUM_ACTIONS];
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            evaluator.try_root_setup(&BoardObservation::default(), &mask),
        )
        .await
        .expect("try_root_setup hung — recovery path regressed");

        assert!(
            matches!(result, Err(EvalError::ReplyDropped)),
            "expected Err(ReplyDropped), got {:?}",
            result.map(|_| "Ok")
        );
        assert!(seen.load(std::sync::atomic::Ordering::SeqCst), "backend never saw the request");

        drop(evaluator);
        let _ = batcher_handle.await;
    }

    /// REGRESSION (champion-promotion wedge, layer a): when the batcher task has
    /// fully exited (request channel closed), `try_expand_leaf` must return
    /// `EvalError::ChannelClosed` rather than panicking on `.expect("inference
    /// channel closed")`. This is the exact condition after a promotion drops the
    /// old champion's `ChannelEvaluator` and its batcher stops.
    #[tokio::test]
    async fn try_expand_leaf_returns_error_when_channel_closed() {
        let (tx, rx) = mpsc::channel::<InferenceRequest>(8);
        // Drop the receiver: the batcher is gone, so sends must fail.
        drop(rx);

        let evaluator = ChannelEvaluator::new(tx);
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            evaluator.try_expand_leaf(&HiddenState::new(64), 0),
        )
        .await
        .expect("try_expand_leaf hung — recovery path regressed");

        assert!(
            matches!(result, Err(EvalError::ChannelClosed)),
            "expected Err(ChannelClosed), got {:?}",
            result.map(|_| "Ok")
        );
    }

    /// The `Evaluator` trait impl must NOT panic when the batcher is gone: it
    /// recovers to a neutral result (uniform policy, zero value) so the MCTS hot
    /// path and the eval game survive a dropped batcher. Without the fix this
    /// `root_setup` call panicked.
    #[tokio::test]
    async fn evaluator_recovers_to_neutral_when_batcher_gone() {
        let (tx, rx) = mpsc::channel::<InferenceRequest>(8);
        drop(rx);

        let evaluator = ChannelEvaluator::with_channels(tx, 48);
        let mask = vec![true; NUM_ACTIONS];
        let (hs, policy, value) = tokio::time::timeout(
            Duration::from_secs(5),
            evaluator.root_setup(&BoardObservation::default(), &mask),
        )
        .await
        .expect("root_setup hung — recovery path regressed");

        assert_eq!(hs.channels, 48, "recovery hidden state must use configured width");
        assert_eq!(policy.len(), NUM_ACTIONS);
        assert!((value - 0.0).abs() < f32::EPSILON);
    }
}
