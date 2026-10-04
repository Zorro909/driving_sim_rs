use super::runner::final_candidate_leader;
use super::stop::{StopRules, StopState};
use crate::args::{PlateauMetric, ScratchReward};
use altd_sim::nn::network::Network;
use altd_sim::training::batch_evaluation::BatchEvaluation;
use altd_sim::training::evolution::{reproduce, reward_values, AgentResult, EvolutionSettings};
use altd_sim::training::pyrandom::PyRandom;
use altd_sim::training::TrainingStats;
use serde_json::json;

#[test]
fn final_candidate_selection_breaks_ties_by_population_index() {
    assert_eq!(final_candidate_leader(&[1.0, 3.0, 3.0, 2.0]), 1);
    assert_eq!(final_candidate_leader(&[0.0, 0.0, 0.0]), 0);
}

#[test]
fn reward_choice_changes_the_preserved_parent() {
    // The distance leader has no completed lap; the fastest lap belongs to
    // the car with the least distance. Exercise real stats and reproduction.
    let networks: Vec<_> = (0..3)
        .map(|i| Network::from_vector(&[1, 1], vec![i as f64, 0.0]))
        .collect();
    let stats: Vec<_> = [(1000.0, None), (800.0, Some(60.0)), (600.0, Some(40.0))]
        .into_iter()
        .map(|(distance, lap)| {
            let mut stats = TrainingStats::new(0.0);
            stats.total_score = distance;
            stats.best_lap_time = lap;
            stats.update_count = 100;
            stats
        })
        .collect();
    let agents: Vec<_> = networks
        .iter()
        .zip(&stats)
        .map(|(network, stats)| AgentResult {
            network,
            metrics: stats.metrics(),
            update_count: stats.update_count,
        })
        .collect();
    for (reward, expected) in [(ScratchReward::Distance, 0), (ScratchReward::BestLapTime, 2)] {
        let settings = EvolutionSettings {
            population: 1,
            selection_size: 1,
            preserve_parents: "on_custom".into(),
            preserve_parents_size: 1,
            rewards: vec![reward.spec()],
            ..Default::default()
        };
        let result = reproduce(&agents, &settings, &mut PyRandom::new(1));
        assert_eq!(result.preserved_count, 1);
        assert_eq!(result.networks[0].params, networks[expected].params);
    }
    let rewards = reward_values(&agents, &[ScratchReward::BestLapTime.spec()]);
    assert!(rewards[2] > rewards[1]);
    assert!(rewards[2] > rewards[0]);
}

#[test]
fn batch_stop_conditions_require_one_driver_to_succeed_across_all_tracks() {
    let network = Network::from_vector(&[1, 1], vec![0.0; 2]);
    let mut batch = BatchEvaluation::new(2);
    for track in 0..2 {
        let results: Vec<_> = (0..2)
            .map(|i| {
                let mut metrics = [Some(0.0); 14];
                metrics[0] = Some(if i == track { 100.0 } else { 0.0 });
                metrics[12] = if i == track { Some(0.1) } else { None };
                AgentResult {
                    network: &network,
                    metrics,
                    update_count: 10,
                }
            })
            .collect();
        batch.record(&results, &[ScratchReward::Distance.spec()]);
    }
    let summary = batch.finish();
    let rules = StopRules {
        score_above: Some(75.0),
        lap_below: Some(15.0),
        lapped_percent: Some(50.0),
        plateau: None,
        plateau_metric: PlateauMetric::Lap,
    };
    let mut state = StopState {
        best_lap: None,
        best_score: f64::NEG_INFINITY,
        stale: 0,
    };
    assert_eq!(
        rules.check(
            &mut state,
            0,
            summary.best_mean_lap_s,
            summary.mean_scores.into_iter().fold(f64::NEG_INFINITY, f64::max),
            summary.lapped_all_tracks,
            2,
            false
        ),
        None
    );
}

#[test]
fn stop_rules_use_generation_laps_and_fixed_order() {
    let rules = StopRules {
        score_above: Some(50.0),
        lap_below: Some(40.0),
        lapped_percent: Some(25.0),
        plateau: Some(2),
        plateau_metric: PlateauMetric::Lap,
    };
    let mut state = StopState {
        best_lap: None,
        best_score: f64::NEG_INFINITY,
        stale: 0,
    };
    assert_eq!(rules.check(&mut state, 0, Some(41.0), 10.0, 1, 10, false), None);
    // The all-time best lap does not satisfy lap_below; a later slower generation must not stop.
    let reason = rules.check(&mut state, 1, Some(40.0), 60.0, 5, 10, false).unwrap();
    assert_eq!(
        reason,
        json!({"condition": "lap_below", "generation": 1, "value": 40.0, "threshold": 40.0})
    );
    assert_eq!(
        rules.check(&mut state, 2, Some(41.0), 60.0, 3, 10, true).unwrap()["condition"],
        "stop_request"
    );
    assert_eq!(
        rules.check(&mut state, 3, None, 60.0, 0, 10, false).unwrap()["condition"],
        "score_above"
    );
    let rules = StopRules {
        score_above: None,
        lap_below: None,
        ..rules
    };
    assert_eq!(
        rules.check(&mut state, 4, Some(45.0), 0.0, 3, 10, false).unwrap()["condition"],
        "lapped_percent"
    );
    // Slower laps and lapless generations are stale; state.stale is 5 by now.
    let rules = StopRules {
        lapped_percent: None,
        ..rules
    };
    let reason = rules.check(&mut state, 5, Some(39.0), 0.0, 1, 10, false);
    assert_eq!(reason, None, "a faster lap resets the plateau");
    assert_eq!(
        rules.check(&mut state, 6, Some(39.0), 0.0, 1, 10, false),
        None,
        "ties do not improve"
    );
    assert_eq!(
        rules.check(&mut state, 7, None, 0.0, 0, 10, false).unwrap()["metric"],
        "lap"
    );
}
