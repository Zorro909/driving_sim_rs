use serde_json::{json, Value};
pub(crate) fn verify(data: &Value) -> Value {
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    let mut errors = std::collections::BTreeMap::<String, Vec<Value>>::new();
    let d = |r: &Value, k: &str| f64::from_bits(r[k].as_u64().unwrap());
    let mut check = |name: &str, want: u64, got: u64, context: &Value| {
        *counts.entry(name.into()).or_default() += 1;
        if want != got {
            let e = errors.entry(name.into()).or_default();
            e.push(json!({"want":want,"got":got,"input":context}));
        }
    };
    for r in data["unary"].as_array().unwrap() {
        let x = d(r, "x");
        check(
            "exp",
            r["exp"].as_u64().unwrap(),
            altd_sim::math::double_math::exp(x).to_bits(),
            r,
        );
        check(
            "log",
            r["log"].as_u64().unwrap(),
            altd_sim::math::double_math::log(x).to_bits(),
            r,
        );
    }
    for r in data["powers"].as_array().unwrap() {
        check(
            "pow",
            r["result"].as_u64().unwrap(),
            altd_sim::math::double_math::pow(d(r, "x"), d(r, "y")).to_bits(),
            r,
        );
    }
    for r in data["grips"].as_array().unwrap() {
        let v = d(r, "velocity");
        let h = d(r, "handbrake");
        let ease = altd_sim::physics::car::godot_ease(h, 0.3);
        let g = 0.20000000298023224
            + (0.8 + (0.1 - 0.8) * ease) / (1.0 + altd_sim::math::double_math::exp((v - 450.0) * 0.00800000037997961));
        check("ease", r["ease"].as_u64().unwrap(), ease.to_bits(), r);
        check("grip", r["result"].as_u64().unwrap(), g.to_bits(), r);
        check(
            "force",
            r["force"].as_u64().unwrap(),
            ((7.3f32 as f64 * g) as f32).to_bits() as u64,
            r,
        );
    }
    for r in data["normals"].as_array().unwrap() {
        let mut rng =
            altd_sim::training::game_random::GameRandom::new([1, 2, 3, 4], r["seed"].as_i64().unwrap() as i32);
        let expected = r["bits"].as_array().unwrap();
        for (i, (v, e)) in rng.normal_array(expected.len(), 0.37).iter().zip(expected).enumerate() {
            check(
                "normal",
                e.as_u64().unwrap(),
                v.to_bits(),
                &json!({"seed":r["seed"],"index":i}),
            );
        }
    }
    for r in data["normalization"].as_array().unwrap() {
        let shape: Vec<_> = r["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let count: usize = shape.windows(2).map(|p| (p[0] + 1) * p[1]).sum();
        let n = altd_sim::nn::network::Network::from_vector(&shape, vec![0.0; count]);
        for (i, a) in ["xavier", "gaussian", "uniform"].iter().enumerate() {
            check(
                "normalization",
                r["factors"][i].as_u64().unwrap(),
                altd_sim::training::evolution::mutation_factor(&n, a).to_bits(),
                r,
            );
        }
    }
    for r in data["distances"].as_array().unwrap() {
        let sum = r["a"]
            .as_array()
            .unwrap()
            .iter()
            .zip(r["b"].as_array().unwrap())
            .map(|(a, b)| {
                altd_sim::math::double_math::pow(
                    f64::from_bits(a.as_u64().unwrap()) - f64::from_bits(b.as_u64().unwrap()),
                    2.0,
                )
            })
            .fold(0.0, |a, b| a + b);
        check("distance", r["result"].as_u64().unwrap(), sum.sqrt().to_bits(), r);
    }
    let error_counts: std::collections::BTreeMap<_, _> = errors.iter().map(|(k, v)| (k, v.len())).collect();
    let samples: std::collections::BTreeMap<_, _> = errors
        .iter()
        .map(|(k, v)| (k, v.iter().take(3).collect::<Vec<_>>()))
        .collect();
    json!({"checked":counts,"errors":error_counts,"samples":samples})
}
