use super::*;
use crate::nn::network::parameter_count;
use crate::training::evolution::Metrics;
use crate::training::game_random::GameRandom;
use crate::training::pyrandom::PyRandom;

fn settings(population: usize) -> EvolutionSettings {
    EvolutionSettings {
        algorithm: "shade".into(),
        population,
        ..EvolutionSettings::default()
    }
}

const SHAPE: [usize; 3] = [3, 4, 2];
const MATH: MathProfile = MathProfile::Proton;

fn seed() -> Network {
    Network::xavier(&SHAPE, &mut PyRandom::new(5))
}

/// Cars whose `total_score` is `score(params)`.
fn agents<'a>(networks: &'a [Network], score: &dyn Fn(&[f64]) -> f64) -> Vec<AgentResult<'a>> {
    networks
        .iter()
        .map(|network| {
            let mut metrics: Metrics = [None; 14];
            metrics[0] = Some(score(&network.params));
            AgentResult {
                network,
                metrics,
                update_count: 10,
            }
        })
        .collect()
}

fn bits(networks: &[Network]) -> Vec<Vec<u64>> {
    networks
        .iter()
        .map(|n| n.params.iter().map(|p| p.to_bits()).collect())
        .collect()
}

fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

fn step(
    shade: &mut Shade,
    cars: &[Network],
    s: &EvolutionSettings,
    rng: &mut TrainingRandom,
    score: &dyn Fn(&[f64]) -> f64,
) -> Produced {
    let results = agents(cars, score);
    shade.next(&results, None, s, rng, &mut Vec::new(), MATH, true)
}

fn into_record(produced: Produced) -> (Vec<Network>, ShadeRecord) {
    let Some(Record::Shade(record)) = produced.record else {
        panic!("no shade record")
    };
    (produced.generation.networks, record)
}

fn best(shade: &Shade) -> &Best {
    shade.search.best.as_ref().unwrap()
}

#[test]
fn the_first_population_is_the_seed_and_its_trials() {
    let (s, seed) = (settings(10), seed());
    let mut rng = TrainingRandom::from(PyRandom::new(1));
    let mut shade = Shade::default();
    let generation = shade.start(&seed, &s, &mut rng, MATH, false).generation;
    let cars = &generation.networks;
    assert_eq!(cars.len(), 10);
    assert_eq!(generation.preserved_count, 1);
    assert_eq!(cars[0].params, seed.params);
    for car in &cars[1..] {
        assert_ne!(car.params, seed.params);
        assert!(distance(&car.params, &seed.params) < 5.0 * s.shade.initial_sigma * (seed.params.len() as f64).sqrt());
    }
    assert_eq!(shade.search.parents.len(), 9);
    assert!(shade.search.scores.iter().all(Option::is_none));
}

#[test]
fn a_better_trial_replaces_its_parent_and_archives_it() {
    let s = settings(8);
    let mut rng = TrainingRandom::from(PyRandom::new(3));
    let mut shade = Shade::default();
    let cars = shade.start(&seed(), &s, &mut rng, MATH, false).generation.networks;
    let first_parents = shade.search.parents.clone();
    // Slot k scores k: every trial is the first reading of its parent.
    let slot: Vec<f64> = (0..cars.len()).map(|k| k as f64).collect();
    let results: Vec<AgentResult> = cars
        .iter()
        .zip(&slot)
        .map(|(network, &score)| {
            let mut metrics: Metrics = [None; 14];
            metrics[0] = Some(score);
            AgentResult {
                network,
                metrics,
                update_count: 10,
            }
        })
        .collect();
    let next = shade
        .next(&results, None, &s, &mut rng, &mut Vec::new(), MATH, false)
        .generation
        .networks;
    assert_eq!(next.len(), 8);
    for i in 0..7 {
        assert_eq!(shade.search.parents[i], cars[i + 1].params);
        assert_eq!(shade.search.scores[i], Some((i + 1) as f64));
    }
    assert_eq!(shade.search.archive, first_parents);
    assert_eq!(
        best(&shade).score,
        Some(7.0),
        "the best car scored 0, the highest parent beats it"
    );
    assert_eq!(best(&shade).params, cars[7].params);
    assert_eq!(next[0].params, cars[7].params);
    // The history moved off its prior, by the replacing trials' F and CR.
    assert_eq!(shade.search.next_cell, 1);
    assert!(shade.search.f_history.iter().any(|&f| f != 0.5));

    // A trial that does not beat its parent changes nothing.
    let before = shade.search.parents.clone();
    let worse = |_: &[f64]| -1.0;
    let cars = next;
    step(&mut shade, &cars, &s, &mut rng, &worse);
    assert_eq!(shade.search.parents, before);
    assert_eq!(shade.search.archive.len(), 7);
}

#[test]
fn climbs_a_quadratic() {
    let target: Vec<f64> = (0..parameter_count(&SHAPE))
        .map(|i| ((i * 7 % 11) as f64 - 5.0) * 0.2)
        .collect();
    let score = |p: &[f64]| -distance(p, &target);
    let s = settings(40);
    let mut rng = TrainingRandom::from(PyRandom::new(2));
    let mut shade = Shade::default();
    let seed = seed();
    let start_distance = distance(&seed.params, &target);
    let mut cars = shade.start(&seed, &s, &mut rng, MATH, false).generation.networks;
    for _ in 0..200 {
        let results = agents(&cars, &score);
        cars = shade
            .next(&results, None, &s, &mut rng, &mut Vec::new(), MATH, false)
            .generation
            .networks;
    }
    let end_distance = distance(&best(&shade).params, &target);
    assert!(
        end_distance < start_distance / 3.0,
        "shade moved from {start_distance} to {end_distance}"
    );
}

#[test]
fn the_best_car_needs_the_margin_to_change() {
    let mut s = settings(8);
    s.shade.promote_margin_rel = 0.5;
    let mut rng = TrainingRandom::from(PyRandom::new(4));
    let mut shade = Shade::default();
    let cars = shade.start(&seed(), &s, &mut rng, MATH, false).generation.networks;
    let seed_params = cars[0].params.clone();
    // The best car scores 10; the best parent 12 is not 50% better.
    let results: Vec<AgentResult> = cars
        .iter()
        .enumerate()
        .map(|(k, network)| {
            let mut metrics: Metrics = [None; 14];
            metrics[0] = Some(if k == 0 { 10.0 } else { 12.0 });
            AgentResult {
                network,
                metrics,
                update_count: 10,
            }
        })
        .collect();
    shade.next(&results, None, &s, &mut rng, &mut Vec::new(), MATH, false);
    assert_eq!(best(&shade).params, seed_params);
    assert_eq!(best(&shade).score, Some(10.0));
    assert_eq!(best(&shade).stalled, 1);
}

#[test]
fn the_worst_parents_jump_to_the_best_car_when_the_search_stalls() {
    let mut s = settings(12);
    s.shade.rejump_gens = 2;
    s.shade.rejump_frac = 0.5;
    let mut rng = TrainingRandom::from(PyRandom::new(6));
    let mut shade = Shade::default();
    let mut cars = shade.start(&seed(), &s, &mut rng, MATH, false).generation.networks;
    // Everything scores the same: nothing improves after the first reading.
    for gen in 0..3 {
        cars = step(&mut shade, &cars, &s, &mut rng, &|_| 1.0).generation.networks;
        let unscored = shade.search.scores.iter().filter(|x| x.is_none()).count();
        match gen {
            // The first readings replace every parent; the best car stays unbeaten at 1.0.
            0 => assert_eq!((unscored, best(&shade).stalled), (0, 1)),
            // The second generation scores no better, and the stall reaches 2: 5 of 11 jump.
            1 => assert_eq!((unscored, best(&shade).stalled), (5, 0)),
            _ => {}
        }
    }
}

#[test]
fn records_rebuild_the_population_and_continue_identically() {
    let s = settings(14);
    let mut rng = TrainingRandom::from(PyRandom::new(8));
    let mut shade = Shade::default();
    let (mut cars, first) = into_record(shade.start(&seed(), &s, &mut rng, MATH, true));
    let (again, after, _) = first.rebuild();
    assert_eq!(bits(&again), bits(&cars));
    assert_eq!(after.to_json(), rng.to_json());

    let score = |p: &[f64]| p.iter().map(|x| x.sin()).sum::<f64>();
    for _ in 0..5 {
        let (next, record) = into_record(step(&mut shade, &cars, &s, &mut rng, &score));
        record.validate().unwrap();
        let (rebuilt, rebuilt_rng, mut restored) = record.rebuild();
        assert_eq!(bits(&rebuilt), bits(&next));
        assert_eq!(rebuilt_rng.to_json(), rng.to_json());
        // Both continue from the same cars and generator.
        let (mut a_rng, mut b_rng) = (rng.clone(), rebuilt_rng);
        let a = step(&mut shade, &next, &s, &mut a_rng, &score);
        let b = step(&mut restored, &next, &s, &mut b_rng, &score);
        assert_eq!(bits(&a.generation.networks), bits(&b.generation.networks));
        assert_eq!(a_rng.to_json(), b_rng.to_json());
        // Take `shade` back to before that trial step.
        let (_, _, state) = record.rebuild();
        shade = state;
        cars = next;
    }
}

#[test]
fn game_streams_drive_it_too() {
    let s = settings(10);
    let mut rng = TrainingRandom::from(GameRandom::new([1, 2, 3, 4], 5));
    let mut shade = Shade::default();
    let cars = shade.start(&seed(), &s, &mut rng, MATH, false).generation.networks;
    let results = agents(&cars, &|p| p[1]);
    let next = shade
        .next(&results, None, &s, &mut rng, &mut Vec::new(), MATH, false)
        .generation;
    assert_eq!(next.networks.len(), 10);
    assert_ne!(next.networks[1].params, cars[1].params);
}

#[test]
fn weights_stay_inside_the_limit() {
    let mut s = settings(10);
    s.shade.max_weight = 0.1;
    let mut rng = TrainingRandom::from(PyRandom::new(9));
    let mut shade = Shade::default();
    let mut cars = shade.start(&seed(), &s, &mut rng, MATH, false).generation.networks;
    for _ in 0..3 {
        cars = step(&mut shade, &cars, &s, &mut rng, &|p| p[0]).generation.networks;
        assert!(cars.iter().flat_map(|c| &c.params).all(|p| p.abs() <= 0.1));
    }
}

#[test]
fn a_foreign_population_restarts_from_its_best_car() {
    let s = settings(10);
    let mut rng = TrainingRandom::from(PyRandom::new(10));
    let mut shade = Shade::default();
    let cars = shade.start(&seed(), &s, &mut rng, MATH, false).generation.networks;
    let results = agents(&cars, &|p| p[2]);
    let leader = (0..cars.len()).fold(0, |b, i| if cars[i].params[2] > cars[b].params[2] { i } else { b });

    shade.invalidate();
    let next = shade
        .next(&results, None, &s, &mut rng, &mut Vec::new(), MATH, false)
        .generation
        .networks;
    assert_eq!(next[0].params, cars[leader].params);
    assert!(
        shade.search.scores.iter().all(Option::is_none),
        "the search began again"
    );

    // A changed population size does the same instead of failing.
    let bigger = settings(14);
    let next = shade
        .next(
            &agents(&next, &|p| p[0]),
            None,
            &bigger,
            &mut rng,
            &mut Vec::new(),
            MATH,
            false,
        )
        .generation
        .networks;
    assert_eq!(next.len(), 14);
}

#[test]
fn settings_are_checked() {
    let ok = ShadeSettings::default();
    assert!(validate(&ok, 5).is_ok());
    assert!(validate(&ok, 4).unwrap_err().contains("population"));
    let bad = |f: &dyn Fn(&mut ShadeSettings)| {
        let mut s = ShadeSettings::default();
        f(&mut s);
        validate(&s, 20).is_err()
    };
    assert!(bad(&|s| s.memory_h = 0));
    assert!(bad(&|s| s.p = 0.0));
    assert!(bad(&|s| s.p = 1.5));
    assert!(bad(&|s| s.initial_sigma = 0.0));
    assert!(bad(&|s| s.initial_sigma = f64::NAN));
    assert!(bad(&|s| s.promote_margin_rel = -0.1));
    assert!(bad(&|s| s.rejump_frac = 1.0));
    assert!(bad(&|s| s.max_weight = -1.0));
}

#[test]
fn corrupt_records_are_rejected() {
    let s = settings(10);
    let mut rng = TrainingRandom::from(PyRandom::new(11));
    let mut shade = Shade::default();
    let (_, record) = into_record(shade.start(&seed(), &s, &mut rng, MATH, true));
    record.validate().unwrap();
    let broken: [&dyn Fn(&mut ShadeRecord); 7] = [
        &|r| r.shape = vec![3],
        &|r| {
            r.search.parents[0].pop();
        },
        &|r| r.search.parents[1][0] = f64::NAN,
        &|r| r.search.scores[0] = Some(f64::INFINITY),
        &|r| r.search.next_cell = 99,
        &|r| r.search.f_history[0] = 0.0,
        &|r| r.sampling.population = 3,
    ];
    for (i, damage) in broken.iter().enumerate() {
        let mut r = record.clone();
        damage(&mut r);
        assert!(r.validate().is_err(), "damage {i}");
    }
}
