use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use crate::data::{BoardObservation, HiddenState, Policy, NUM_ACTIONS, NUM_OBS_PLANES};
use crate::selfplay::inference::{InferenceBackend, InferenceRequest};

/// Cumulative count of inference fallbacks (uniform policy + value `0.0`)
/// emitted across every batch since process start. Fallbacks silently poison
/// the replay buffer with garbage targets, so we track them and abort once the
/// cumulative total exceeds `HYZERO_INFERENCE_FALLBACK_LIMIT`.
static FALLBACK_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Cumulative-fallback abort limit from `HYZERO_INFERENCE_FALLBACK_LIMIT`
/// (cached; default 100). Once the running total exceeds this the process
/// panics rather than keep feeding silent uniform-policy/value-0 data.
fn fallback_limit() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HYZERO_INFERENCE_FALLBACK_LIMIT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(100)
    })
}

/// True when a cumulative fallback count exceeds the configured limit and the
/// run should abort. Factored out for cheap unit testing of the threshold.
fn should_abort_fallbacks(cumulative: usize, limit: usize) -> bool {
    cumulative > limit
}

/// Record `n` newly emitted fallbacks for `context`, log the running total,
/// and panic once the cumulative count exceeds the configured limit. A dead
/// run is preferable to silently training on uniform-policy/value-0 targets.
fn record_fallbacks(n: usize, context: &str) {
    if n == 0 {
        return;
    }
    let cumulative = FALLBACK_COUNT.fetch_add(n, Ordering::Relaxed) + n;
    let limit = fallback_limit();
    eprintln!(
        "[PyO3Backend] {context}: emitted {n} inference fallback(s) \
         (cumulative {cumulative}, limit {limit})"
    );
    if should_abort_fallbacks(cumulative, limit) {
        panic!(
            "[PyO3Backend] cumulative inference fallbacks ({cumulative}) exceeded \
             HYZERO_INFERENCE_FALLBACK_LIMIT ({limit}); aborting to avoid poisoning \
             the replay buffer with uniform-policy/value-0 targets"
        );
    }
}

/// Reply sender for a batched `RootSetup`: (hidden, policy, value, moves-left `m`).
/// The trailing `Option<f32>` is `Some` only under HYZERO_MOVES_LEFT_HEAD=1.
type RootReplySender = tokio::sync::oneshot::Sender<(HiddenState, Policy, f32, Option<f32>)>;
/// Reply sender for a batched `ExpandLeaf`: (hidden, reward, policy, value, `m`).
type LeafReplySender = tokio::sync::oneshot::Sender<(HiddenState, f32, Policy, f32, Option<f32>)>;

/// PyO3 backend that delegates batch inference to the Python InferenceServer.
///
/// Holds a reference to a Python `InferenceServer` object and calls
/// `root_setup_batch` / `expand_leaf_batch` through the GIL.
pub struct PyO3Backend {
    /// Python `InferenceServer` instance.
    server: Py<PyAny>,
    /// Number of hidden channels (matches network config, default 64).
    hidden_channels: usize,
}

impl PyO3Backend {
    /// Create a new backend wrapping an existing Python `InferenceServer` object.
    pub fn new(server: Py<PyAny>, hidden_channels: usize) -> Self {
        Self {
            server,
            hidden_channels,
        }
    }

    /// Construct a backend by instantiating the Python InferenceServer from config.
    ///
    /// Imports `hyzero.config.DEFAULT_CONFIG` and `hyzero.inference.server.InferenceServer`
    /// and creates an instance on the given device.
    pub fn from_default_config(device: &str) -> PyResult<Self> {
        Python::attach(|py| {
            let config_obj = PyModule::import(py, "hyzero.config")?
                .getattr("DEFAULT_CONFIG")?
                .into_pyobject(py)?;
            let config_dict = config_obj.cast::<PyDict>()?;
            let hidden_channels: usize = config_dict
                .get_item("hidden_channels")?
                .ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err("hidden_channels not in config")
                })?
                .extract()?;
            let config = config_obj.unbind();
            let cls =
                PyModule::import(py, "hyzero.inference.server")?.getattr("InferenceServer")?;
            let server: Py<PyAny> = cls.call1((config, device))?.unbind();
            Ok(Self::new(server, hidden_channels))
        })
    }

    /// Build a zero-filled hidden state for fallback use.
    fn fallback_hidden(&self) -> HiddenState {
        HiddenState::new(self.hidden_channels)
    }

    /// Build a uniform policy for fallback use.
    fn fallback_policy() -> Policy {
        vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS]
    }
}

impl InferenceBackend for PyO3Backend {
    fn evaluate_batch(&mut self, requests: Vec<InferenceRequest>) {
        // Separate requests by type, preserving reply senders.
        let mut root_obs: Vec<BoardObservation> = Vec::new();
        let mut root_masks: Vec<Vec<bool>> = Vec::new();
        let mut root_replies: Vec<RootReplySender> = Vec::new();

        let mut leaf_hidden: Vec<HiddenState> = Vec::new();
        let mut leaf_actions: Vec<crate::data::ActionIndex> = Vec::new();
        let mut leaf_replies: Vec<LeafReplySender> = Vec::new();

        for req in requests {
            match req {
                InferenceRequest::RootSetup {
                    observation,
                    legal_mask,
                    reply,
                } => {
                    root_obs.push(observation);
                    root_masks.push(legal_mask);
                    root_replies.push(reply);
                }
                InferenceRequest::ExpandLeaf {
                    hidden_state,
                    action,
                    reply,
                } => {
                    leaf_hidden.push(hidden_state);
                    leaf_actions.push(action);
                    leaf_replies.push(reply);
                }
            }
        }

        Python::attach(|py| {
            // --- RootSetup batch ---
            if !root_obs.is_empty() {
                let b = root_obs.len();
                // Stack observations: each planes Vec<f32> length NUM_OBS_PLANES*64
                let mut flat: Vec<f32> = Vec::with_capacity(b * NUM_OBS_PLANES * 64);
                for obs in &root_obs {
                    flat.extend_from_slice(&obs.planes);
                }

                let result: PyResult<()> = (|| {
                    // Create numpy array [B*NUM_OBS_PLANES*64] then reshape to [B, NUM_OBS_PLANES, 8, 8]
                    let arr = flat.into_pyarray(py);
                    let obs_np = arr.reshape([b, NUM_OBS_PLANES, 8, 8])?;

                    // Build legal-mask array [B * NUM_ACTIONS] -> reshape to [B, NUM_ACTIONS]
                    let mut flat_masks: Vec<bool> = Vec::with_capacity(b * NUM_ACTIONS);
                    for mask in &root_masks {
                        flat_masks.extend_from_slice(mask);
                    }
                    let mask_arr = flat_masks.into_pyarray(py);
                    let masks_np = mask_arr.reshape([b, NUM_ACTIONS])?;

                    let ret =
                        self.server
                            .call_method1(py, "root_setup_batch", (obs_np, masks_np))?;
                    let tuple = ret.cast_bound::<PyTuple>(py)?;

                    // Unpack: (hidden [B,64,8,8], policies [B,NUM_ACTIONS], values [B])
                    // Call .flatten() via Python to get a 1-D contiguous array we can read
                    let hidden_flat: PyReadonlyArray1<f32> = tuple
                        .get_item(0)?
                        .call_method0("flatten")?
                        .cast_into::<PyArray1<f32>>()?
                        .readonly();

                    let policy_flat: PyReadonlyArray1<f32> = tuple
                        .get_item(1)?
                        .call_method0("flatten")?
                        .cast_into::<PyArray1<f32>>()?
                        .readonly();

                    let value_arr: PyReadonlyArray1<f32> =
                        tuple.get_item(2)?.cast_into::<PyArray1<f32>>()?.readonly();

                    // Optional trailing moves-left array [B], present only when the
                    // server runs with HYZERO_MOVES_LEFT_HEAD=1. Absent → every
                    // reply carries `None` and nodes keep the neutral 0.5.
                    let moves_left_arr: Option<PyReadonlyArray1<f32>> = if tuple.len() >= 4 {
                        Some(tuple.get_item(3)?.cast_into::<PyArray1<f32>>()?.readonly())
                    } else {
                        None
                    };

                    let hidden_data = hidden_flat.as_slice()?;
                    let policy_data = policy_flat.as_slice()?;
                    let value_data = value_arr.as_slice()?;
                    let moves_left_data: Option<&[f32]> = match &moves_left_arr {
                        Some(arr) => Some(arr.as_slice()?),
                        None => None,
                    };

                    let hidden_stride = self.hidden_channels * 64;
                    let policy_stride = NUM_ACTIONS;

                    if hidden_data.len() != b * hidden_stride {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "hidden array size mismatch: expected {}, got {}",
                            b * hidden_stride,
                            hidden_data.len()
                        )));
                    }
                    if policy_data.len() != b * policy_stride {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "policy array size mismatch: expected {}, got {}",
                            b * policy_stride,
                            policy_data.len()
                        )));
                    }
                    if value_data.len() != b {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "value array size mismatch: expected {}, got {}",
                            b,
                            value_data.len()
                        )));
                    }
                    if let Some(ml) = moves_left_data {
                        if ml.len() != b {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "moves_left array size mismatch: expected {}, got {}",
                                b,
                                ml.len()
                            )));
                        }
                    }

                    for (i, reply) in root_replies.drain(..).enumerate() {
                        let hs = HiddenState {
                            data: hidden_data[i * hidden_stride..(i + 1) * hidden_stride].to_vec(),
                            channels: self.hidden_channels,
                        };
                        let pol: Policy =
                            policy_data[i * policy_stride..(i + 1) * policy_stride].to_vec();
                        let val = value_data[i];
                        let m = moves_left_data.map(|ml| ml[i]);
                        let _ = reply.send((hs, pol, val, m));
                    }
                    Ok(())
                })();

                if let Err(e) = result {
                    eprintln!("[PyO3Backend] root_setup_batch error: {e}");
                    // Send fallbacks for any replies not yet consumed
                    let n = root_replies.len();
                    for reply in root_replies.drain(..) {
                        let _ = reply.send((
                            self.fallback_hidden(),
                            Self::fallback_policy(),
                            0.0,
                            None,
                        ));
                    }
                    record_fallbacks(n, "root_setup_batch");
                }
            }

            // --- ExpandLeaf batch ---
            if !leaf_hidden.is_empty() {
                let b = leaf_hidden.len();
                let hidden_channels = self.hidden_channels;

                let mut flat_hidden: Vec<f32> = Vec::with_capacity(b * hidden_channels * 64);
                for hs in &leaf_hidden {
                    flat_hidden.extend_from_slice(&hs.data);
                }

                let mut flat_actions: Vec<f32> = Vec::with_capacity(b * 3 * 64);
                for &action in &leaf_actions {
                    let planes = crate::data::encode_action_spatial(action);
                    flat_actions.extend_from_slice(&planes);
                }

                let result: PyResult<()> = (|| {
                    let hidden_arr = flat_hidden.into_pyarray(py);
                    let hidden_np = hidden_arr.reshape([b, hidden_channels, 8, 8])?;

                    let action_arr = flat_actions.into_pyarray(py);
                    let actions_np = action_arr.reshape([b, 3, 8, 8])?;

                    let ret = self.server.call_method1(
                        py,
                        "expand_leaf_batch",
                        (hidden_np, actions_np),
                    )?;
                    let tuple = ret.cast_bound::<PyTuple>(py)?;

                    // Unpack: (new_hidden [B,64,8,8], rewards [B], policies [B,4096], values [B])
                    let new_hidden_flat: PyReadonlyArray1<f32> = tuple
                        .get_item(0)?
                        .call_method0("flatten")?
                        .cast_into::<PyArray1<f32>>()?
                        .readonly();

                    let reward_arr: PyReadonlyArray1<f32> =
                        tuple.get_item(1)?.cast_into::<PyArray1<f32>>()?.readonly();

                    let policy_flat: PyReadonlyArray1<f32> = tuple
                        .get_item(2)?
                        .call_method0("flatten")?
                        .cast_into::<PyArray1<f32>>()?
                        .readonly();

                    let value_arr: PyReadonlyArray1<f32> =
                        tuple.get_item(3)?.cast_into::<PyArray1<f32>>()?.readonly();

                    // Optional trailing moves-left array [B]; present only under
                    // HYZERO_MOVES_LEFT_HEAD=1. Absent → `None` per reply (node 0.5).
                    let moves_left_arr: Option<PyReadonlyArray1<f32>> = if tuple.len() >= 5 {
                        Some(tuple.get_item(4)?.cast_into::<PyArray1<f32>>()?.readonly())
                    } else {
                        None
                    };

                    let hidden_data = new_hidden_flat.as_slice()?;
                    let reward_data = reward_arr.as_slice()?;
                    let policy_data = policy_flat.as_slice()?;
                    let value_data = value_arr.as_slice()?;
                    let moves_left_data: Option<&[f32]> = match &moves_left_arr {
                        Some(arr) => Some(arr.as_slice()?),
                        None => None,
                    };

                    let hidden_stride = hidden_channels * 64;
                    let policy_stride = NUM_ACTIONS;

                    if hidden_data.len() != b * hidden_stride {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "hidden array size mismatch: expected {}, got {}",
                            b * hidden_stride,
                            hidden_data.len()
                        )));
                    }
                    if policy_data.len() != b * policy_stride {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "policy array size mismatch: expected {}, got {}",
                            b * policy_stride,
                            policy_data.len()
                        )));
                    }
                    if value_data.len() != b {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "value array size mismatch: expected {}, got {}",
                            b,
                            value_data.len()
                        )));
                    }
                    if reward_data.len() != b {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "reward array size mismatch: expected {}, got {}",
                            b,
                            reward_data.len()
                        )));
                    }
                    if let Some(ml) = moves_left_data {
                        if ml.len() != b {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "moves_left array size mismatch: expected {}, got {}",
                                b,
                                ml.len()
                            )));
                        }
                    }

                    for (i, reply) in leaf_replies.drain(..).enumerate() {
                        let hs = HiddenState {
                            data: hidden_data[i * hidden_stride..(i + 1) * hidden_stride].to_vec(),
                            channels: hidden_channels,
                        };
                        let reward = reward_data[i];
                        let pol: Policy =
                            policy_data[i * policy_stride..(i + 1) * policy_stride].to_vec();
                        let val = value_data[i];
                        let m = moves_left_data.map(|ml| ml[i]);
                        let _ = reply.send((hs, reward, pol, val, m));
                    }
                    Ok(())
                })();

                if let Err(e) = result {
                    eprintln!("[PyO3Backend] expand_leaf_batch error: {e}");
                    let n = leaf_replies.len();
                    for reply in leaf_replies.drain(..) {
                        let _ = reply.send((
                            self.fallback_hidden(),
                            0.0,
                            Self::fallback_policy(),
                            0.0,
                            None,
                        ));
                    }
                    record_fallbacks(n, "expand_leaf_batch");
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::oneshot;

    #[test]
    fn fallback_abort_triggers_only_above_limit() {
        // At or below the limit the run continues; strictly above it aborts.
        assert!(!should_abort_fallbacks(0, 100));
        assert!(!should_abort_fallbacks(100, 100));
        assert!(should_abort_fallbacks(101, 100));
        // A zero limit aborts on the first fallback.
        assert!(should_abort_fallbacks(1, 0));
    }

    fn make_server() -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let config = PyModule::import(py, "hyzero.config")?
                .getattr("DEFAULT_CONFIG")?
                .into_pyobject(py)?
                .unbind();
            let cls =
                PyModule::import(py, "hyzero.inference.server")?.getattr("InferenceServer")?;
            let server: Py<PyAny> = cls.call1((config, "cpu"))?.unbind();
            Ok(server)
        })
    }

    #[test]
    #[ignore = "requires hyzero Python package"]
    fn test_root_setup_batch() {
        let server = make_server().expect("failed to create InferenceServer");
        let mut backend = PyO3Backend::new(server, 64);

        let (tx, mut rx) = oneshot::channel();
        let obs = BoardObservation::default();
        let req = InferenceRequest::RootSetup {
            observation: obs,
            legal_mask: vec![true; NUM_ACTIONS],
            reply: tx,
        };

        backend.evaluate_batch(vec![req]);

        let (hs, policy, value, _m) = rx.try_recv().expect("no reply received");
        assert_eq!(hs.channels, 128, "hidden_channels should be 128");
        assert_eq!(
            hs.data.len(),
            128 * 64,
            "hidden data length should be 128*64"
        );
        assert_eq!(
            policy.len(),
            NUM_ACTIONS,
            "policy length should be NUM_ACTIONS"
        );
        let policy_sum: f32 = policy.iter().sum();
        assert!(
            (policy_sum - 1.0).abs() < 1e-3,
            "policy should sum to ~1.0, got {policy_sum}"
        );
        let _ = value; // verify it is f32
    }

    #[test]
    #[ignore = "requires hyzero Python package"]
    fn test_expand_leaf_batch() {
        let server = make_server().expect("failed to create InferenceServer");
        let mut backend = PyO3Backend::new(server, 64);

        let (tx, mut rx) = oneshot::channel();
        let hs_in = HiddenState::new(64);
        let req = InferenceRequest::ExpandLeaf {
            hidden_state: hs_in,
            action: 42,
            reply: tx,
        };

        backend.evaluate_batch(vec![req]);

        let (hs_out, reward, policy, value, _m) = rx.try_recv().expect("no reply received");
        assert_eq!(hs_out.channels, 128, "hidden_channels should be 128");
        assert_eq!(
            hs_out.data.len(),
            128 * 64,
            "hidden data length should be 128*64"
        );
        assert_eq!(
            policy.len(),
            NUM_ACTIONS,
            "policy length should be NUM_ACTIONS"
        );
        let policy_sum: f32 = policy.iter().sum();
        assert!(
            (policy_sum - 1.0).abs() < 1e-3,
            "policy should sum to ~1.0, got {policy_sum}"
        );
        let _ = (reward, value); // verify they are f32
    }
}
