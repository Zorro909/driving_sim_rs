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
    let end_distance = distance(&ars.theta, &target);
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
    assert_eq!(next.networks[3].params, ars.theta, "then the moved search point");
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
    assert_eq!(ars.theta, seed.params);
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
    assert_eq!(ars.theta, others[top].params);
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
            r.theta.pop();
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
        ars.theta
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
    assert_eq!(a.theta, b.theta);
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

#[test]
fn an_edited_probe_makes_the_search_start_from_the_best_car() {
    let s = settings(10, 0);
    let (mut ars, mut rng) = (Ars::default(), TrainingRandom::from(PyRandom::new(1)));
    let mut cars = ars.start(&seed(), &s, &mut rng, MATH, false).generation.networks;
    cars[3].params[0] += 1.0;
    ars.invalidate();
    next_with(&mut ars, &cars, &s, 2);
    assert_eq!(ars.theta, cars[1].params, "the leader of the fixed scores");
}
