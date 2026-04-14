use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};
use tokio::time::{timeout, Duration};

use crate::data::{BoardObservation, HiddenState, Policy, ActionIndex, NUM_ACTIONS};
use crate::mcts::evaluator::Evaluator;

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
#[derive(Clone)]
pub struct ChannelEvaluator {
    tx: mpsc::Sender<InferenceRequest>,
}

impl ChannelEvaluator {
    pub fn new(tx: mpsc::Sender<InferenceRequest>) -> Self {
        Self { tx }
    }
}

#[async_trait]
impl Evaluator for ChannelEvaluator {
    async fn root_setup(&self, observation: &BoardObservation, legal_mask: &[bool]) -> (HiddenState, Policy, f32) {
        let (reply_tx, reply_rx) = oneshot::channel();
        let req = InferenceRequest::RootSetup {
            observation: observation.clone(),
            legal_mask: legal_mask.to_vec(),
            reply: reply_tx,
        };
        self.tx.send(req).await.expect("inference channel closed");
        reply_rx.await.expect("inference reply dropped")
    }

    async fn expand_leaf(&self, hidden_state: &HiddenState, action: ActionIndex) -> (HiddenState, f32, Policy, f32) {
        let (reply_tx, reply_rx) = oneshot::channel();
        let req = InferenceRequest::ExpandLeaf {
            hidden_state: hidden_state.clone(),
            action,
            reply: reply_tx,
        };
        self.tx.send(req).await.expect("inference channel closed");
        reply_rx.await.expect("inference reply dropped")
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
}
