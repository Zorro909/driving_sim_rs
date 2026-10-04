//! Equal-weight evaluation of one ordered population on several tracks.
use crate::training::evolution::{reward_values, AgentResult, RewardSpec};

pub struct BatchEvaluation {
    tracks: usize,
    rewards: Vec<f64>,
    totals: Vec<[f64; 14]>,
    updates: Vec<u64>,
    lap_counts: Vec<usize>,
    lap_sums: Vec<f64>,
}

pub struct BatchSummary {
    pub fitness: Vec<f64>,
    pub mean_scores: Vec<f64>,
    pub metric_totals: Vec<[f64; 14]>,
    pub update_counts: Vec<u64>,
    pub lap_counts: Vec<usize>,
    pub lapped_all_tracks: usize,
    pub best_mean_lap_s: Option<f64>,
}

impl BatchEvaluation {
    pub fn new(population: usize) -> Self {
        assert!(population > 0);
        Self {
            tracks: 0,
            rewards: vec![0.0; population],
            totals: vec![[0.0; 14]; population],
            updates: vec![0; population],
            lap_counts: vec![0; population],
            lap_sums: vec![0.0; population],
        }
    }

    pub fn record(&mut self, results: &[AgentResult], specs: &[RewardSpec]) {
        assert_eq!(results.len(), self.rewards.len());
        let rewards = reward_values(results, specs);
        for (i, result) in results.iter().enumerate() {
            self.rewards[i] += rewards[i];
            self.updates[i] += result.update_count;
            for (metric, value) in result.metrics.iter().enumerate() {
                if metric == 10 {
                    self.totals[i][metric] = value.unwrap_or(0.0);
                } else if metric != 12 && metric != 13 {
                    self.totals[i][metric] += value.unwrap_or(0.0);
                }
            }
            if let Some(performance) = result.metrics[12].filter(|v| *v > 0.0) {
                self.lap_counts[i] += 1;
                self.lap_sums[i] += 1.0 / performance;
            }
        }
        self.tracks += 1;
    }

    pub fn finish(self) -> BatchSummary {
        assert!(self.tracks > 0);
        let count = self.tracks as f64;
        let complete_laps: Vec<f64> = self
            .lap_counts
            .iter()
            .zip(&self.lap_sums)
            .filter(|(laps, _)| **laps == self.tracks)
            .map(|(_, sum)| sum / count)
            .collect();
        let best_mean_lap_s = complete_laps.into_iter().reduce(f64::min);
        let lapped_all_tracks = self.lap_counts.iter().filter(|&&laps| laps == self.tracks).count();
        let mut totals = self.totals;
        for (i, row) in totals.iter_mut().enumerate() {
            // Non-additive metrics retain their definitions over the full evaluation.
            row[12] = if self.lap_counts[i] == self.tracks {
                count / self.lap_sums[i]
            } else {
                0.0
            };
            row[13] = if row[3] != 0.0 {
                row[0] / (row[3] * (5.0 / 384.0))
            } else {
                0.0
            };
        }
        BatchSummary {
            fitness: self.rewards.into_iter().map(|value| value / count).collect(),
            mean_scores: totals.iter().map(|row| row[0] / count).collect(),
            metric_totals: totals,
            update_counts: self.updates,
            lap_counts: self.lap_counts,
            lapped_all_tracks,
            best_mean_lap_s,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::network::Network;

    #[test]
    fn consistency_beats_a_specialist_and_missing_laps_are_not_zero_times() {
        let network = Network::from_vector(&[1, 1], vec![0.0; 2]);
        let specs = vec![RewardSpec {
            metric: "total_score".into(),
            weight: 100,
            kind: "default".into(),
        }];
        let mut batch = BatchEvaluation::new(3);
        for scores in [[1000.0, 20.0, 0.0], [0.0, 20.0, 30.0], [0.0, 20.0, 30.0]] {
            let results: Vec<_> = scores
                .into_iter()
                .enumerate()
                .map(|(i, score)| {
                    let mut metrics = [Some(0.0); 14];
                    metrics[0] = Some(score);
                    metrics[10] = Some(7.0);
                    metrics[12] = if i == 0 { None } else { Some(0.1) };
                    AgentResult {
                        network: &network,
                        metrics,
                        update_count: 10,
                    }
                })
                .collect();
            batch.record(&results, &specs);
        }
        let summary = batch.finish();
        assert_eq!(summary.fitness, vec![1.0 / 3.0, 0.5, 2.0 / 3.0]);
        assert!(summary.fitness[1] > summary.fitness[0]);
        assert_eq!(summary.lapped_all_tracks, 2);
        assert_eq!(summary.best_mean_lap_s, Some(10.0));
        assert_eq!(summary.update_counts, vec![30; 3]);
        assert_eq!(summary.metric_totals[0][10], 7.0);
        assert_eq!(summary.lap_counts[0], 0);
    }

    #[test]
    fn mixed_signed_and_average_rewards_are_combined_after_each_track_is_ranked() {
        let network = Network::from_vector(&[1, 1], vec![0.0; 2]);
        let specs = vec![
            RewardSpec {
                metric: "total_score".into(),
                weight: 60,
                kind: "average".into(),
            },
            RewardSpec {
                metric: "collision_count".into(),
                weight: -20,
                kind: "default".into(),
            },
            RewardSpec {
                metric: "best_lap_performance".into(),
                weight: 20,
                kind: "default".into(),
            },
        ];
        let mut batch = BatchEvaluation::new(3);
        let mut expected = vec![0.0; 3];
        for track in 0..2 {
            let results: Vec<_> = (0..3)
                .map(|i| {
                    let mut metrics = [Some(0.0); 14];
                    metrics[0] = Some([[90.0, 60.0, 30.0], [10.0, 20.0, 30.0]][track][i]);
                    metrics[11] = Some((i * 10) as f64);
                    metrics[12] = if i == track {
                        None
                    } else {
                        Some(1.0 / (10.0 + i as f64))
                    };
                    AgentResult {
                        network: &network,
                        metrics,
                        update_count: [100, 10, 20][i],
                    }
                })
                .collect();
            for (sum, reward) in expected.iter_mut().zip(reward_values(&results, &specs)) {
                *sum += reward / 2.0;
            }
            batch.record(&results, &specs);
        }
        let summary = batch.finish();
        assert_eq!(summary.fitness, expected);
        assert_eq!(summary.mean_scores, vec![50.0, 40.0, 30.0]);
        assert_eq!(summary.lap_counts, vec![1, 1, 2]);
        // Two different fast drivers each miss a track; only the third is eligible.
        assert_eq!(summary.lapped_all_tracks, 1);
        assert_eq!(summary.best_mean_lap_s, Some(12.0));
    }
}
