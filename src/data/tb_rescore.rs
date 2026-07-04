//! lc0-style tablebase tail-rescoring of self-play value targets.
//!
//! When `HYZERO_TB_RESCORE` is truthy AND the WDL CSV exists, this module loads a
//! `normfen -> wdl` map once (via [`OnceLock`]). Self-play looks up each played
//! position's normalized FEN (see [`crate::game::GameBoard::to_normfen`]); on a
//! hit the position's side-to-move WDL supersedes the computed TD/outcome value
//! target for that step (see `replay_buffer::sample_batch`).
//!
//! POV / sign mapping: the CSV `wdl` column is already expressed from the
//! side-to-move point of view (STM POV), exactly the convention that
//! `compute_td_target` uses for its targets. The mapping is therefore the
//! IDENTITY — the stored `i8` WDL becomes the `f32` target directly, with no
//! sign flip. Concretely, for a White-winning KQvK position the CSV holds `+1`
//! on the White-to-move row and `-1` on the Black-to-move row, so the override
//! target is `+1.0` when White is to move and `-1.0` when Black is to move.
//!
//! When the env is off or the file is missing, the map resolves to `None`, every
//! lookup returns `None`, and behavior is byte-identical to the pre-rescore code.

use std::collections::HashMap;
use std::sync::OnceLock;

/// Default location of the exported WDL CSV (overridable via `HYZERO_TB_WDL_PATH`).
const DEFAULT_TB_WDL_PATH: &str = "data/syzygy/tb_wdl.csv";

/// Whether an env var is "truthy": present and not one of `""`/`0`/`false`/`no`
/// (case-insensitive, trimmed). Absence is falsey (rescoring off by default).
fn env_truthy(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "no")
        }
        Err(_) => false,
    }
}

/// Parse the CSV body into a `normfen -> wdl` map.
///
/// Each non-empty line is `normfen,wdl` where `normfen` is the first four FEN
/// fields (which themselves contain spaces) and `wdl ∈ {-1,0,1}`. The split is
/// on the LAST comma so the space-bearing normfen is preserved verbatim. Blank
/// lines, a leading `normfen,wdl` header, and lines with an unparseable WDL are
/// skipped. Pure: no env or filesystem access.
fn parse_csv(contents: &str) -> HashMap<String, i8> {
    let mut map = HashMap::new();
    for line in contents.lines() {
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            continue;
        }
        let Some(comma) = line.rfind(',') else {
            continue;
        };
        let (normfen, wdl_str) = (&line[..comma], &line[comma + 1..]);
        let Ok(wdl) = wdl_str.trim().parse::<i8>() else {
            continue; // skips a "normfen,wdl" header row too
        };
        map.insert(normfen.to_string(), wdl);
    }
    map
}

/// Convert a stored WDL (`i8`, STM POV) into a value target (`f32`, STM POV).
/// Identity mapping — see the module POV note.
fn wdl_to_target(wdl: i8) -> f32 {
    wdl as f32
}

/// Look up a normfen in an explicit map, returning the STM-POV value target.
/// Pure; used by both the global path and the unit tests.
fn lookup_in(map: &HashMap<String, i8>, normfen: &str) -> Option<f32> {
    map.get(normfen).copied().map(wdl_to_target)
}

/// The process-wide WDL map, loaded once. `None` when rescoring is disabled or
/// the CSV is absent/unreadable — every lookup then returns `None`.
fn wdl_map() -> Option<&'static HashMap<String, i8>> {
    static MAP: OnceLock<Option<HashMap<String, i8>>> = OnceLock::new();
    MAP.get_or_init(|| {
        if !env_truthy("HYZERO_TB_RESCORE") {
            return None;
        }
        let path =
            std::env::var("HYZERO_TB_WDL_PATH").unwrap_or_else(|_| DEFAULT_TB_WDL_PATH.to_string());
        match std::fs::read_to_string(&path) {
            Ok(contents) => {
                let map = parse_csv(&contents);
                eprintln!(
                    "[tb_rescore] loaded {} WDL entries from {}",
                    map.len(),
                    path
                );
                Some(map)
            }
            Err(_) => {
                eprintln!("[tb_rescore] HYZERO_TB_RESCORE set but {path} missing — disabled");
                None
            }
        }
    })
    .as_ref()
}

/// Whether tablebase rescoring is active (env truthy AND CSV loaded). Cheap after
/// the first call; self-play gates its per-step normfen computation on this.
pub fn tb_rescore_active() -> bool {
    wdl_map().is_some()
}

/// Look up `normfen` in the loaded WDL map, returning the STM-POV value target.
/// Returns `None` when rescoring is inactive or the position is not covered.
pub fn tb_rescore_lookup(normfen: &str) -> Option<f32> {
    wdl_map().and_then(|m| lookup_in(m, normfen))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `parse_csv` keeps the space-bearing normfen intact by splitting on the last
    /// comma, and skips a header row with a non-numeric WDL.
    #[test]
    fn parse_csv_preserves_normfen_and_skips_header() {
        let csv = "normfen,wdl\n\
                   4k3/8/4K3/8/8/8/4R3/8 w - -,1\n\
                   8/8/8/8/8/8/8/k1K5 b - -,0\n\
                   \n\
                   8/8/8/8/8/8/8/K1k5 w - -,-1\n";
        let map = parse_csv(csv);
        assert_eq!(map.len(), 3, "header + blank line skipped");
        assert_eq!(map.get("4k3/8/4K3/8/8/8/4R3/8 w - -"), Some(&1i8));
        assert_eq!(map.get("8/8/8/8/8/8/8/k1K5 b - -"), Some(&0i8));
        assert_eq!(map.get("8/8/8/8/8/8/8/K1k5 w - -"), Some(&-1i8));
    }

    /// WDL is STM POV and passes through unchanged: a White-winning KQvK position
    /// yields target +1 when White is to move and −1 when Black is to move.
    #[test]
    fn wdl_pov_sign_maps_stm_winning_to_plus_one() {
        use crate::game::fen::board_from_fen;
        use crate::PrecomputedItems;
        use std::sync::Arc;

        let pre = Arc::new(PrecomputedItems::begin_precomputing());
        // White KQ vs lone Black K — White is winning regardless of side to move.
        let (wtm_board, wtm_color, _) =
            board_from_fen("4k3/8/8/8/8/8/Q7/4K3 w - - 0 1", pre.clone()).unwrap();
        let (btm_board, btm_color, _) =
            board_from_fen("4k3/8/8/8/8/8/Q7/4K3 b - - 0 1", pre.clone()).unwrap();

        // The CSV stores STM POV: +1 on the White-to-move row, −1 on the Black row.
        let mut map: HashMap<String, i8> = HashMap::new();
        map.insert(wtm_board.to_normfen(wtm_color), 1);
        map.insert(btm_board.to_normfen(btm_color), -1);

        assert_eq!(
            lookup_in(&map, &wtm_board.to_normfen(wtm_color)),
            Some(1.0),
            "White winning + White to move -> +1 target"
        );
        assert_eq!(
            lookup_in(&map, &btm_board.to_normfen(btm_color)),
            Some(-1.0),
            "White winning + Black to move -> -1 target"
        );
    }

    /// A normfen not present in the map yields no override.
    #[test]
    fn lookup_miss_returns_none() {
        let map: HashMap<String, i8> = HashMap::new();
        assert_eq!(lookup_in(&map, "8/8/8/8/8/8/8/K1k5 w - -"), None);
    }

    /// `env_truthy` is off by default (absent) and off for the disabling values,
    /// on otherwise — so an unset `HYZERO_TB_RESCORE` keeps rescoring inactive.
    #[test]
    fn env_truthy_is_off_by_default_and_for_falsey_values() {
        use crate::data::types::TestEnvGuard;
        let _env = TestEnvGuard::new(&["HYZERO_TB_RESCORE_TEST"]);
        // SAFETY: protected by TestEnvGuard; no concurrent env-var access.
        unsafe {
            std::env::remove_var("HYZERO_TB_RESCORE_TEST");
            assert!(!env_truthy("HYZERO_TB_RESCORE_TEST"), "absent ⇒ off");
            for v in ["", "0", "false", "FALSE", "no", "  no  "] {
                std::env::set_var("HYZERO_TB_RESCORE_TEST", v);
                assert!(!env_truthy("HYZERO_TB_RESCORE_TEST"), "{v:?} ⇒ off");
            }
            for v in ["1", "true", "yes", "on"] {
                std::env::set_var("HYZERO_TB_RESCORE_TEST", v);
                assert!(env_truthy("HYZERO_TB_RESCORE_TEST"), "{v:?} ⇒ on");
            }
        }
    }
}
