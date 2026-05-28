//! Elo rating math for the dual-model evaluation ladder.
//!
//! Pure functions only — used by `EvaluationTask` to update a candidate's
//! rating against fixed-rating archived champions.

/// Initial rating assigned to both candidate and opponents at the start of a cycle.
pub const INITIAL_RATING: f32 = 1500.0;

/// Standard chess K-factor for per-game updates.
pub const K_FACTOR: f32 = 32.0;

/// Expected score for player A vs. player B given their ratings.
///
/// Returns a value in (0, 1): the probability A wins (1.0) plus half the
/// probability of a draw (0.5), per standard Elo math.
pub fn expected_score(r_a: f32, r_b: f32) -> f32 {
    1.0 / (1.0 + 10f32.powf((r_b - r_a) / 400.0))
}

/// Update a player's rating after a single game.
///
/// `score` is 1.0 for win, 0.5 for draw, 0.0 for loss.
/// `k` is the K-factor (32.0 is the project default).
pub fn update_rating(rating: f32, opp_rating: f32, score: f32, k: f32) -> f32 {
    rating + k * (score - expected_score(rating, opp_rating))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_equal_ratings_is_half() {
        let v = expected_score(1500.0, 1500.0);
        assert!((v - 0.5).abs() < 1e-6);
    }

    #[test]
    fn expected_higher_rating_above_half() {
        let v = expected_score(1600.0, 1500.0);
        assert!(v > 0.5);
        assert!(v < 1.0);
    }

    #[test]
    fn update_win_vs_equal_adds_16() {
        let r = update_rating(1500.0, 1500.0, 1.0, 32.0);
        assert!((r - 1516.0).abs() < 1e-3);
    }

    #[test]
    fn update_loss_vs_equal_subtracts_16() {
        let r = update_rating(1500.0, 1500.0, 0.0, 32.0);
        assert!((r - 1484.0).abs() < 1e-3);
    }

    #[test]
    fn update_draw_vs_equal_is_noop() {
        let r = update_rating(1500.0, 1500.0, 0.5, 32.0);
        assert!((r - 1500.0).abs() < 1e-6);
    }

    #[test]
    fn update_loss_when_ahead_loses_more() {
        // Player at 1520 loses to 1500: expected ~0.529, so delta = -K * 0.529 ≈ -16.93.
        // Delta is LARGER (more Elo lost) than the equal-rating loss (-16.0), per the
        // standard Elo "loss-when-favored is more painful" property. The plan's literal
        // assertion `r < 1484.0` is mathematically impossible (1520 - 17 = 1503); we
        // assert the actual property the test name promises (loses MORE than 16 points).
        let r = update_rating(1520.0, 1500.0, 0.0, 32.0);
        let delta = 1520.0 - r;
        assert!(
            delta > 16.0,
            "expected delta > 16 (loses more than equal), got {delta}"
        );
    }

    #[test]
    fn sequential_table_driven() {
        // Hand-computed reference values for scores [1.0, 0.5, 1.0, 0.0, 1.0]
        // vs. fixed opponent at 1500.0 with K=32, starting at 1500.0. Values computed
        // at f32 precision (cross-validated with Python `struct.pack('f', r)`).
        let scores = [1.0, 0.5, 1.0, 0.0, 1.0];
        let expected = [
            1516.0_f32,
            1_515.263_67,
            1_530.561_2,
            1_513.157_3,
            1_528.551_8,
        ];
        let mut r = 1500.0_f32;
        for (i, s) in scores.iter().enumerate() {
            r = update_rating(r, 1500.0, *s, 32.0);
            assert!(
                (r - expected[i]).abs() < 1e-3,
                "step {i}: got {r}, expected {}",
                expected[i]
            );
        }
    }
}
