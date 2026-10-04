//! Early-stop conditions and plateau state for one invocation.

use crate::args::PlateauMetric;
use serde_json::{json, Value};

/// Early-stop rules of one train-scratch invocation; any rule ends the run.
#[derive(Clone, Copy)]
pub(crate) struct StopRules {
    pub(crate) score_above: Option<f64>,
    pub(crate) lap_below: Option<f64>,
    pub(crate) lapped_percent: Option<f64>,
    pub(crate) plateau: Option<usize>,
    pub(crate) plateau_metric: PlateauMetric,
}

/// Plateau bookkeeping; counts start with the invocation, not the run.
pub(super) struct StopState {
    pub(super) best_lap: Option<f64>,
    pub(super) best_score: f64,
    pub(super) stale: usize,
}

impl StopRules {
    pub(super) fn to_json(self) -> Value {
        json!({
            "score_above": self.score_above, "lap_below": self.lap_below,
            "lapped_percent": self.lapped_percent, "plateau": self.plateau,
            "plateau_metric": match self.plateau_metric { PlateauMetric::Lap => "lap", PlateauMetric::Score => "score" },
        })
    }

    /// Stop reason after `generation`, if a rule fired. Checked in a fixed order.
    // Keep the stop inputs explicit and evaluate rules in their recorded order.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn check(
        &self,
        state: &mut StopState,
        generation: usize,
        lap: Option<f64>,
        score: f64,
        lapped: usize,
        population: usize,
        stop_request: bool,
    ) -> Option<Value> {
        let improved = match self.plateau_metric {
            PlateauMetric::Lap => lap.is_some_and(|t| state.best_lap.is_none_or(|best| t < best)),
            PlateauMetric::Score => score > state.best_score,
        };
        if let Some(t) = lap {
            state.best_lap = Some(state.best_lap.map_or(t, |best| best.min(t)));
        }
        state.best_score = state.best_score.max(score);
        state.stale = if improved { 0 } else { state.stale + 1 };
        let percent = 100.0 * lapped as f64 / population as f64;
        let reason = |condition: &str, value: Value, threshold: Value| {
            Some(json!({"condition": condition, "generation": generation, "value": value, "threshold": threshold}))
        };
        if stop_request {
            return reason("stop_request", Value::Null, Value::Null);
        }
        if let (Some(limit), Some(t)) = (self.lap_below, lap) {
            if t <= limit {
                return reason("lap_below", json!(t), json!(limit));
            }
        }
        if let Some(limit) = self.score_above {
            if score >= limit {
                return reason("score_above", json!(score), json!(limit));
            }
        }
        if let Some(limit) = self.lapped_percent {
            if percent >= limit {
                return reason("lapped_percent", json!(percent), json!(limit));
            }
        }
        if let Some(limit) = self.plateau {
            if state.stale >= limit {
                let mut stop = reason("plateau", json!(state.stale), json!(limit)).unwrap();
                stop["metric"] = self.to_json()["plateau_metric"].clone();
                return Some(stop);
            }
        }
        None
    }
}
