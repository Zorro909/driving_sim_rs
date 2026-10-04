use altd_sim::{nn::network::Network, training::game_random::GameRandom};
use serde_json::{json, Value};
pub(crate) fn verify(d: &Value) -> Value {
    let mut rng = GameRandom::new([1, 2, 3, 4], 12345);
    assert_eq!(rng.normal_state(), d["before"]["normals"]);
    let mut errors = Vec::new();
    let mut checked = 0;
    for r in d["rows"].as_array().unwrap() {
        let n = r["length"].as_u64().unwrap() as usize;
        let actual = rng.normal_array(n, 1.0);
        for (i, (a, e)) in actual.iter().zip(r["bits"].as_array().unwrap()).enumerate() {
            checked += 1;
            if a.to_bits() != e.as_u64().unwrap() {
                errors.push(json!({"batch":n,"index":i,"actual":a.to_bits(),"expected":e}));
            }
        }
        assert_eq!(rng.normal_state(), r["state"], "state batch {n}");
    }
    for r in d["draws"].as_array().unwrap() {
        let bound = r["bound"].as_u64().unwrap() as usize;
        assert_eq!(rng.randrange(bound), r["integer"].as_u64().unwrap() as usize);
        assert_eq!(rng.random().to_bits(), r["bits"].as_u64().unwrap());
    }
    assert_eq!(rng.decision_json(), d["decisions_after"]);
    for key in ["initialization", "mutation"] {
        let mut rng = GameRandom::new([1, 2, 3, 4], 12345);
        let n = if key == "initialization" {
            Network::xavier_game(&[2, 3, 2], &mut rng)
        } else {
            Network::from_vector(&[2, 3, 2], vec![0.0; 17]).mutate_xavier_game(0.2, &mut rng)
        };
        for (i, (a, e)) in n.params.iter().zip(d[key]["bits"].as_array().unwrap()).enumerate() {
            checked += 1;
            if a.to_bits() != e.as_u64().unwrap() {
                errors.push(json!({"case":key,"index":i,"actual":a.to_bits(),"expected":e}));
            }
        }
        assert_eq!(rng.normal_state(), d[key]["state"], "state {key}");
    }
    if let Some(cases) = d["reproduction"].as_array() {
        for case in cases {
            use altd_sim::training::evolution::{reproduce_game, AgentResult, EvolutionSettings};
            let parents: Vec<_> = (0..3)
                .map(|p| Network::from_vector(&[2, 3, 2], (0..17).map(|j| (100 * p + j + 1) as f64).collect()))
                .collect();
            let agents: Vec<_> = parents
                .iter()
                .enumerate()
                .map(|(i, n)| {
                    let mut metrics = [None; 14];
                    metrics[0] = Some([10.0, 20.0, 20.0][i]);
                    AgentResult {
                        network: n,
                        metrics,
                        update_count: 0,
                    }
                })
                .collect();
            let selection = case["selection"].as_str().unwrap().to_lowercase();
            let crossover = case["crossover"]
                .as_str()
                .unwrap()
                .replace("SinglePoint", "single_point")
                .to_lowercase();
            let settings = EvolutionSettings {
                population: 4,
                selection_algorithm: selection.clone(),
                selection_size: 2,
                crossover: crossover.clone(),
                mutation_rate: 0.2,
                weight_decay: 0.0005,
                preserve_parents_size: 2,
                ..Default::default()
            };
            let mut rng = GameRandom::new([1, 2, 3, 4], 12345);
            let result = reproduce_game(&agents, &settings, &mut rng);
            assert_eq!(result.preserved_count, case["preserved"].as_u64().unwrap() as usize);
            for (i, (n, e)) in result
                .networks
                .iter()
                .zip(case["networks"].as_array().unwrap())
                .enumerate()
            {
                for (j, (a, b)) in n.params.iter().zip(e.as_array().unwrap()).enumerate() {
                    checked += 1;
                    if a.to_bits() != b.as_u64().unwrap() {
                        errors.push(json!({"selection":selection,"crossover":crossover,"network":i,"parameter":j,"actual":a.to_bits(),"expected":b}));
                    }
                }
            }
            assert_eq!(
                rng.decision_json(),
                case["decisions"],
                "decision state {selection}/{crossover}"
            );
            assert_eq!(
                rng.normal_state(),
                case["normals"],
                "normal state {selection}/{crossover}"
            );
        }
    }
    if let Some(seeds) = d["seeds"].as_array() {
        for r in seeds {
            let seed = r["seed"].as_i64().unwrap() as i32;
            let mut rng = GameRandom::new([1, 2, 3, 4], seed);
            assert_eq!(rng.normal_state(), r["initial"], "initial state seed {seed}");
            for (i, (a, e)) in rng
                .normal_array(33, 1.0)
                .iter()
                .zip(r["bits"].as_array().unwrap())
                .enumerate()
            {
                checked += 1;
                if a.to_bits() != e.as_u64().unwrap() {
                    errors.push(json!({"seed":seed,"index":i,"actual":a.to_bits(),"expected":e}));
                }
            }
            assert_eq!(rng.normal_state(), r["state"], "after state seed {seed}");
        }
    }
    json!({"checked":checked,"errors":errors})
}
