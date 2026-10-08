use super::*;
use crate::training::evolution::Metrics;
use crate::training::game_random::GameRandom;
use crate::training::pyrandom::PyRandom;

fn settings(population: usize, elite_count: usize) -> EvolutionSettings {
    EvolutionSettings {
        algorithm: "ars".into(),
        population,
        ars: ArsSettings {
            nu: 0.1,
            alpha: 0.05,
            elite_count,
            ..ArsSettings::default()
        },
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
    ars: &mut Ars,
    cars: &[Network],
    s: &EvolutionSettings,
    rng: &mut TrainingRandom,
    score: &dyn Fn(&[f64]) -> f64,
) -> Produced {
    let results = agents(cars, score);
    ars.next(&results, None, s, rng, &mut Vec::new(), MATH, true)
}

fn into_record(produced: Produced) -> (Vec<Network>, ArsRecord) {
    let Some(Record::Ars(record)) = produced.record else {
        panic!("no ars record")
    };
    (produced.generation.networks, record)
}

#[test]
fn the_first_population_is_the_seed_and_antithetic_probes() {
    let (s, seed) = (settings(10, 2), seed());
    let mut rng = TrainingRandom::from(PyRandom::new(1));
    let mut ars = Ars::default();
    let generation = ars.start(&seed, &s, &mut rng, MATH, false).generation;
    let cars = &generation.networks;
    assert_eq!(cars.len(), 10);
    assert_eq!(generation.preserved_count, 1, "no elites yet, only the search point");
    assert_eq!(cars[0].params, seed.params);
    // 10 cars: the point, 4 pairs and one spare copy of the point.
    assert_eq!(cars[9].params, seed.params);
    for pair in cars[1..9].chunks(2) {
        for ((p, m), t) in pair[0].params.iter().zip(&pair[1].params).zip(&seed.params) {
            assert!(((p + m) / 2.0 - t).abs() < 1e-12, "probes straddle the point");
        }
        assert_ne!(pair[0].params, pair[1].params);
    }
}

#[test]
fn climbs_a_quadratic() {
    let target: Vec<f64> = (0..parameter_count(&SHAPE))
        .map(|i| ((i * 7 % 11) as f64 - 5.0) * 0.2)
        .collect();
    let score = |p: &[f64]| -distance(p, &target);
    let s = settings(40, 3);
    let mut rng = TrainingRandom::from(PyRandom::new(2));
    let mut ars = Ars::default();
    let seed = seed();
    let start_distance = distance(&seed.params, &target);
    let mut cars = ars.start(&seed, &s, &mut rng, MATH, false).generation.networks;
    for _ in 0..150 {
        let results = agents(&cars, &score);
        cars = ars
            .next(&results, None, &s, &mut rng, &mut Vec::new(), MATH, false)
            .generation
            .networks;
    }
    let end_distance = distance(&ars.heads[0].theta, &target);
    assert!(
        end_distance < start_distance / 3.0,
        "ars moved from {start_distance} to {end_distance}"
    );
}

#[test]
fn the_pool_holds_the_best_distinct_cars_and_leads_the_next_generation() {
    let s = settings(12, 3);
    let mut rng = TrainingRandom::from(PyRandom::new(3));
    let mut ars = Ars::default();
    let first = ars.start(&seed(), &s, &mut rng, MATH, false).generation.networks;
    let score = |p: &[f64]| p.iter().sum::<f64>();
    // The point is in the population twice (slot 0 and the spare); the pool keeps it once.
    let mut order: Vec<usize> = (0..first.len()).collect();
    order.sort_by(|&a, &b| score(&first[b].params).partial_cmp(&score(&first[a].params)).unwrap());
    let mut best: Vec<Network> = Vec::new();
    for &i in &order {
        if best.len() < 3 && !best.iter().any(|n| n.params == first[i].params) {
            best.push(first[i].clone());
        }
    }
    let results = agents(&first, &score);
    let next = ars
        .next(&results, None, &s, &mut rng, &mut Vec::new(), MATH, false)
        .generation;
    assert_eq!(ars.pool.len(), 3);
    assert_eq!(
        bits(&next.networks[..3]),
        bits(&best),
        "the elites come first, best first"
    );
    assert_eq!(next.preserved_count, 4);
    assert_eq!(
        next.networks[3].params, ars.heads[0].theta,
        "then the moved search point"
    );
    assert_eq!(next.networks.len(), 12);
}

#[test]
fn identical_scores_leave_the_search_point_alone() {
    let s = settings(10, 1);
    let mut rng = TrainingRandom::from(PyRandom::new(4));
    let mut ars = Ars::default();
    let seed = seed();
    let cars = ars.start(&seed, &s, &mut rng, MATH, false).generation.networks;
    let results = agents(&cars, &|_| 7.0);
    ars.next(&results, None, &s, &mut rng, &mut Vec::new(), MATH, false);
    assert_eq!(ars.heads[0].theta, seed.params);
}

#[test]
fn a_foreign_population_restarts_from_its_best_car() {
    let s = settings(10, 1);
    let mut rng = TrainingRandom::from(PyRandom::new(6));
    let mut ars = Ars::default();
    ars.start(&seed(), &s, &mut rng, MATH, false);
    let others: Vec<Network> = (0..6)
        .map(|i| Network::xavier(&SHAPE, &mut PyRandom::new(100 + i)))
        .collect();
    let results = agents(&others, &|p| p[0]);
    let top = (0..6)
        .max_by(|&a, &b| others[a].params[0].partial_cmp(&others[b].params[0]).unwrap())
        .unwrap();
    let next = ars
        .next(&results, None, &s, &mut rng, &mut Vec::new(), MATH, false)
        .generation;
    assert_eq!(ars.heads[0].theta, others[top].params);
    assert_eq!(next.networks.len(), 10);
}

/// A record rebuilds its population and generator, and the rebuilt search
/// state goes on exactly as the original does.
#[test]
fn records_rebuild_the_population_and_continue_identically() {
    let s = settings(14, 2);
    let mut rng = TrainingRandom::from(PyRandom::new(8));
    let mut ars = Ars::default();
    let started = ars.start(&seed(), &s, &mut rng, MATH, true);
    let (mut cars, first) = into_record(started);
    let (again, after, _) = first.rebuild();
    assert_eq!(bits(&again), bits(&cars));
    assert_eq!(after.to_json(), rng.to_json());

    let score = |p: &[f64]| p.iter().map(|x| x.sin()).sum::<f64>();
    for _ in 0..4 {
        let (next, record) = into_record(step(&mut ars, &cars, &s, &mut rng, &score));
        record.validate().unwrap();
        let (rebuilt, rebuilt_rng, mut restored) = record.rebuild();
        assert_eq!(bits(&rebuilt), bits(&next));
        assert_eq!(rebuilt_rng.to_json(), rng.to_json());
        // Both continue from the same cars and generator.
        let (mut a_rng, mut b_rng) = (rng.clone(), rebuilt_rng);
        let a = step(&mut ars, &next, &s, &mut a_rng, &score);
        let b = step(&mut restored, &next, &s, &mut b_rng, &score);
        assert_eq!(bits(&a.generation.networks), bits(&b.generation.networks));
        assert_eq!(a_rng.to_json(), b_rng.to_json());
        // Take `ars` back to before that trial step.
        let (_, _, state) = record.rebuild();
        ars = state;
        cars = next;
    }
}

#[test]
fn game_streams_drive_it_too() {
    let s = settings(10, 1);
    let mut rng = TrainingRandom::from(GameRandom::new([1, 2, 3, 4], 5));
    let mut ars = Ars::default();
    let cars = ars.start(&seed(), &s, &mut rng, MATH, false).generation.networks;
    let results = agents(&cars, &|p| p[1]);
    let next = ars
        .next(&results, None, &s, &mut rng, &mut Vec::new(), MATH, false)
        .generation;
    assert_eq!(next.networks.len(), 10);
    assert_ne!(next.networks[1].params, cars[1].params);
}

#[test]
fn weights_stay_inside_the_limit() {
    let mut s = settings(10, 1);
    s.ars.max_weight = 0.3;
    let mut rng = TrainingRandom::from(PyRandom::new(9));
    let mut ars = Ars::default();
    let seed = Network::from_vector(&SHAPE, vec![0.29; parameter_count(&SHAPE)]);
    let cars = ars.start(&seed, &s, &mut rng, MATH, false).generation.networks;
    assert!(cars.iter().flat_map(|n| &n.params).all(|p| p.abs() <= 0.3));
}

#[test]
fn settings_are_checked() {
    let ok = ArsSettings::default();
    validate(&ok, 6).unwrap();
    assert!(validate(&ok, 5).unwrap_err().contains("elite_count + 3"));
    for bad in [
        ArsSettings { nu: 0.0, ..ok.clone() },
        ArsSettings {
            alpha: f64::NAN,
            ..ok.clone()
        },
        ArsSettings {
            top_frac: 0.0,
            ..ok.clone()
        },
        ArsSettings {
            top_frac: 1.5,
            ..ok.clone()
        },
        ArsSettings {
            max_weight: -1.0,
            ..ok.clone()
        },
    ] {
        assert!(validate(&bad, 50).is_err(), "{bad:?}");
    }
}

#[test]
fn corrupt_records_are_rejected() {
    let s = settings(8, 1);
    let mut rng = TrainingRandom::from(PyRandom::new(10));
    let mut ars = Ars::default();
    let (_, good) = into_record(ars.start(&seed(), &s, &mut rng, MATH, true));
    good.validate().unwrap();
    let edits: [&dyn Fn(&mut ArsRecord); 4] = [
        &|r| {
            r.heads[0].theta.pop();
        },
        &|r| r.sampling.nu = 0.0,
        &|r| r.sampling.population = 2,
        &|r| r.shape = vec![3],
    ];
    for edit in edits {
        let mut broken = good.clone();
        edit(&mut broken);
        assert!(broken.validate().is_err());
    }
}

/// Scores that rank the first pair's plus probe best and the rest alike.
fn fixed_scores(len: usize) -> Vec<f64> {
    let mut scores = vec![0.0; len];
    scores[1] = 10.0;
    scores
}

fn next_with(ars: &mut Ars, cars: &[Network], s: &EvolutionSettings, seed: i64) -> Vec<Network> {
    let results = agents(cars, &|_| 0.0);
    let mut rng = TrainingRandom::from(PyRandom::new(seed));
    ars.next(
        &results,
        Some(fixed_scores(cars.len())),
        s,
        &mut rng,
        &mut Vec::new(),
        MATH,
        false,
    )
    .generation
    .networks
}

#[test]
fn the_step_divides_by_the_noise_the_population_was_sampled_with() {
    let run = |later_nu: f64| {
        let s = settings(10, 0);
        let (mut ars, mut rng) = (Ars::default(), TrainingRandom::from(PyRandom::new(1)));
        let cars = ars.start(&seed(), &s, &mut rng, MATH, false).generation.networks;
        let mut changed = s.clone();
        changed.ars.nu = later_nu;
        next_with(&mut ars, &cars, &changed, 2);
        ars.heads[0].theta.clone()
    };
    // The step happens before the new population is sampled, whatever nu says by then.
    assert_eq!(run(0.1), run(0.001));
}

#[test]
fn a_rebuilt_search_keeps_the_noise_it_sampled_with() {
    let s = settings(10, 0);
    let mut rng = TrainingRandom::from(PyRandom::new(1));
    let (networks, record) = into_record(Ars::default().start(&seed(), &s, &mut rng, MATH, true));
    let mut other = s.clone();
    other.ars.nu = 0.001;
    let (_, _, mut a) = record.rebuild();
    let (_, _, mut b) = record.rebuild();
    next_with(&mut a, &networks, &s, 3);
    next_with(&mut b, &networks, &other, 3);
    assert_eq!(a.heads[0].theta, b.heads[0].theta);
}

#[test]
fn max_weight_bounds_the_search_point_and_elites_even_without_a_step() {
    let mut s = settings(10, 2);
    s.ars.max_weight = 0.01;
    let (mut ars, mut rng) = (Ars::default(), TrainingRandom::from(PyRandom::new(1)));
    let within = |cars: &[Network]| cars.iter().all(|n| n.params.iter().all(|p| p.abs() <= 0.01));
    let first = ars.start(&seed(), &s, &mut rng, MATH, false).generation.networks;
    assert!(within(&first), "the first population obeys the limit");
    // Tied scores skip the step; the elites and the point still obey the limit.
    let second = next_with(&mut ars, &first, &s, 2);
    assert!(within(&second));
    // A tighter limit later applies to the pool as well.
    s.ars.max_weight = 0.005;
    let third = next_with(&mut ars, &second, &s, 3);
    assert!(third.iter().all(|n| n.params.iter().all(|p| p.abs() <= 0.005)));
}

fn heads_settings(population: usize, elite_count: usize, heads: usize) -> EvolutionSettings {
    let mut s = settings(population, elite_count);
    s.ars.heads = heads;
    s
}

#[test]
fn pairs_are_split_among_the_heads_with_the_remainder_to_the_first() {
    assert_eq!(
        (0..3).map(|h| pair_range(7, 3, h)).collect::<Vec<_>>(),
        [(0, 3), (3, 2), (5, 2)]
    );
    assert_eq!(
        (0..2).map(|h| pair_range(1, 2, h)).collect::<Vec<_>>(),
        [(0, 1), (1, 0)]
    );
}

#[test]
fn every_head_gets_its_own_point_and_probes() {
    // 2 elites, 3 heads: 2 + 3 points + 5 pairs (3 + 1 + 1) = 15, and one spare.
    let (s, seed) = (heads_settings(16, 2, 3), seed());
    let mut rng = TrainingRandom::from(PyRandom::new(1));
    let mut ars = Ars::default();
    let first = ars.start(&seed, &s, &mut rng, MATH, false);
    assert_eq!(first.generation.preserved_count, 3, "no elites yet, one point per head");
    let cars = first.generation.networks;
    assert_eq!(cars.len(), 16);
    // Different noise, same start: the heads begin at the seed and separate by their probes.
    assert!(cars[..3].iter().all(|c| c.params == seed.params));
    let results = agents(&cars, &|p| p[0]);
    let next = ars
        .next(&results, None, &s, &mut rng, &mut Vec::new(), MATH, false)
        .generation;
    assert_eq!(ars.heads.len(), 3);
    assert_eq!(next.networks.len(), 16);
    // Elites are in front of the heads' points.
    let moved = &next.networks[2..5];
    for (car, head) in moved.iter().zip(&ars.heads) {
        assert_eq!(car.params, head.theta);
    }
    assert_ne!(ars.heads[0].theta, ars.heads[1].theta, "heads follow their own probes");
}

#[test]
fn several_heads_climb_a_quadratic() {
    let target: Vec<f64> = (0..parameter_count(&SHAPE))
        .map(|i| ((i * 7 % 11) as f64 - 5.0) * 0.2)
        .collect();
    let score = |p: &[f64]| -distance(p, &target);
    let s = heads_settings(60, 3, 3);
    let mut rng = TrainingRandom::from(PyRandom::new(2));
    let (mut ars, seed) = (Ars::default(), seed());
    let start_distance = distance(&seed.params, &target);
    let mut cars = ars.start(&seed, &s, &mut rng, MATH, false).generation.networks;
    for _ in 0..150 {
        let results = agents(&cars, &score);
        cars = ars
            .next(&results, None, &s, &mut rng, &mut Vec::new(), MATH, false)
            .generation
            .networks;
    }
    for head in &ars.heads {
        assert!(distance(&head.theta, &target) < start_distance / 3.0);
    }
}

#[test]
fn the_head_count_can_change_between_generations() {
    let mut rng = TrainingRandom::from(PyRandom::new(3));
    let (mut ars, mut s) = (Ars::default(), heads_settings(20, 1, 1));
    let cars = ars.start(&seed(), &s, &mut rng, MATH, false).generation.networks;
    s.ars.heads = 3;
    let (cars, record) = into_record(step(&mut ars, &cars, &s, &mut rng, &|p| p[0]));
    assert_eq!((ars.heads.len(), record.heads.len()), (3, 3));
    record.validate().unwrap();
    assert_eq!(
        record.heads[1].theta, record.heads[0].theta,
        "new heads start from head 0"
    );
    s.ars.heads = 1;
    let next = step(&mut ars, &cars, &s, &mut rng, &|p| p[0]).generation;
    assert_eq!(ars.heads.len(), 1);
    assert_eq!(next.networks.len(), 20);
}

fn adapting(every: u32) -> ArsSettings {
    ArsSettings {
        alpha: 0.05,
        alpha_min: 0.01,
        alpha_max: 0.2,
        alpha_adapt_every: every,
        alpha_adapt_fast: 0.5,
        alpha_adapt_slow: 0.05,
        alpha_adapt_threshold: 0.0,
        alpha_adapt_up: 2.0,
        alpha_adapt_down: 0.5,
        ..ArsSettings::default()
    }
}

#[test]
fn alpha_grows_while_the_reward_rises_and_shrinks_while_it_falls() {
    let ars = adapting(1);
    let mut head = Head::new(vec![0.0]);
    head.adapt(1.0, &ars);
    assert_eq!(head.alpha, Some(0.05), "the first reading has no trend");
    for reward in [2.0, 3.0] {
        head.adapt(reward, &ars);
    }
    assert_eq!(head.alpha, Some(0.2), "rising rewards double alpha up to the maximum");
    for reward in [2.0, 0.0, -2.0, -4.0, -6.0, -8.0, -10.0, -12.0] {
        head.adapt(reward, &ars);
    }
    assert_eq!(head.alpha, Some(0.01), "falling rewards halve it down to the minimum");
}

#[test]
fn alpha_is_checked_only_every_so_many_generations_and_ignores_missing_rewards() {
    let ars = adapting(3);
    let mut head = Head::new(vec![0.0]);
    for reward in [0.0, 1.0, 2.0] {
        head.adapt(reward, &ars);
    }
    assert_eq!(head.alpha, Some(0.1), "one check after the third generation");
    head.adapt(f64::NAN, &ars);
    head.adapt(f64::NAN, &ars);
    let (fast, slow) = (head.fast, head.slow);
    head.adapt(f64::NAN, &ars);
    assert_eq!((head.fast, head.slow), (fast, slow));
    assert_eq!(head.alpha, Some(0.2), "the averages still rise, so alpha follows");
}

#[test]
fn equal_bounds_freeze_alpha_and_the_threshold_holds_it_on_small_trends() {
    let frozen = ArsSettings {
        alpha_min: 0.05,
        alpha_max: 0.05,
        ..adapting(1)
    };
    let mut head = Head::new(vec![0.0]);
    for reward in [0.0, 5.0, 10.0] {
        head.adapt(reward, &frozen);
    }
    assert_eq!(head.alpha, Some(0.05));

    let tolerant = ArsSettings {
        alpha_adapt_threshold: 10.0,
        ..adapting(1)
    };
    let mut head = Head::new(vec![0.0]);
    for reward in [1.0, 1.1, 1.2] {
        head.adapt(reward, &tolerant);
    }
    assert_eq!(head.alpha, Some(0.05));
}

#[test]
fn the_adapted_alpha_scales_the_step() {
    let step_of = |alpha_min: f64| {
        let mut s = settings(10, 0);
        s.ars.alpha_min = alpha_min;
        s.ars.alpha_max = alpha_min;
        s.ars.alpha = alpha_min;
        let (mut ars, mut rng) = (Ars::default(), TrainingRandom::from(PyRandom::new(1)));
        let cars = ars.start(&seed(), &s, &mut rng, MATH, false).generation.networks;
        next_with(&mut ars, &cars, &s, 2);
        distance(&ars.heads[0].theta, &seed().params)
    };
    let (small, large) = (step_of(0.01), step_of(0.02));
    assert!((large / small - 2.0).abs() < 1e-9, "{small} {large}");
}

#[test]
fn records_keep_the_step_size_state() {
    let s = settings(14, 1);
    let mut rng = TrainingRandom::from(PyRandom::new(8));
    let mut ars = Ars::default();
    let (mut cars, _) = into_record(ars.start(&seed(), &s, &mut rng, MATH, true));
    let score = |p: &[f64]| p.iter().sum::<f64>();
    for _ in 0..3 {
        let (next, record) = into_record(step(&mut ars, &cars, &s, &mut rng, &score));
        assert_eq!(record.heads, ars.heads);
        cars = next;
    }
    assert!(ars.heads[0].alpha.is_some() && ars.heads[0].fast.is_some() && ars.heads[0].since_check == 3);
}

#[test]
fn corrupt_head_state_is_rejected() {
    let s = heads_settings(12, 1, 2);
    let mut rng = TrainingRandom::from(PyRandom::new(10));
    let (_, good) = into_record(Ars::default().start(&seed(), &s, &mut rng, MATH, true));
    good.validate().unwrap();
    let edits: [&dyn Fn(&mut ArsRecord); 4] = [
        &|r| {
            r.heads.pop();
        },
        &|r| r.heads[1].alpha = Some(-1.0),
        &|r| r.heads[0].fast = Some(f64::INFINITY),
        &|r| r.sampling.population = 5,
    ];
    for edit in edits {
        let mut broken = good.clone();
        edit(&mut broken);
        assert!(broken.validate().is_err());
    }
}

#[test]
fn head_and_adaptation_settings_are_checked() {
    let ok = ArsSettings::default();
    validate(&ok, 6).unwrap();
    assert!(validate(&ArsSettings { heads: 2, ..ok.clone() }, 8)
        .unwrap_err()
        .contains("3 * heads"));
    validate(&ArsSettings { heads: 2, ..ok.clone() }, 9).unwrap();
    for bad in [
        ArsSettings { heads: 0, ..ok.clone() },
        ArsSettings {
            alpha_min: 0.3,
            alpha_max: 0.1,
            ..ok.clone()
        },
        ArsSettings {
            alpha_min: 0.0,
            ..ok.clone()
        },
        ArsSettings {
            alpha_adapt_every: 0,
            ..ok.clone()
        },
        ArsSettings {
            alpha_adapt_fast: 0.0,
            ..ok.clone()
        },
        ArsSettings {
            alpha_adapt_slow: 1.5,
            ..ok.clone()
        },
        ArsSettings {
            alpha_adapt_threshold: -1.0,
            ..ok.clone()
        },
        ArsSettings {
            alpha_adapt_up: f64::NAN,
            ..ok.clone()
        },
        ArsSettings {
            alpha_adapt_down: 0.0,
            ..ok.clone()
        },
    ] {
        assert!(validate(&bad, 50).is_err(), "{bad:?}");
    }
}

#[test]
fn an_edited_probe_makes_the_search_start_from_the_best_car() {
    let s = settings(10, 0);
    let (mut ars, mut rng) = (Ars::default(), TrainingRandom::from(PyRandom::new(1)));
    let mut cars = ars.start(&seed(), &s, &mut rng, MATH, false).generation.networks;
    cars[3].params[0] += 1.0;
    ars.invalidate();
    next_with(&mut ars, &cars, &s, 2);
    assert_eq!(ars.heads[0].theta, cars[1].params, "the leader of the fixed scores");
}
