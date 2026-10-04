use altd_sim::track::random_track::{self, RandomTrackConfig, TrackGenerator};
use serde_json::Value;

pub(crate) fn verify(data: &Value) {
    let catalog = &data["catalog"];
    let seed = catalog["hashcode_seed"].as_u64().unwrap() as u32;
    for row in catalog["connection_hashes"].as_array().unwrap() {
        assert_eq!(
            random_track::combine_hash(
                seed,
                row["side"].as_i64().unwrap() as i32,
                row["index"].as_i64().unwrap() as i32
            ),
            row["hash"].as_i64().unwrap() as i32
        );
    }
    for (block, want) in random_track::blocks()
        .iter()
        .zip(catalog["block_hashes"].as_array().unwrap())
    {
        assert_eq!(
            random_track::connection_set_hash(seed, &block.connections),
            want.as_i64().unwrap() as i32
        );
    }
    let generator = TrackGenerator::new(seed);
    assert_eq!(generator.lookup_json(), catalog["lookups"]);
    for case in data["cases"].as_array().unwrap() {
        let mut config: RandomTrackConfig = serde_json::from_value(case["config"].clone()).unwrap();
        let mut state = serde_json::from_value(case["state"].clone()).unwrap();
        let result = generator.generate(&mut config, &mut state);
        if let Some(error) = case["error"].as_str() {
            assert_eq!(result.unwrap_err(), error);
        } else {
            assert_eq!(serde_json::to_value(result.unwrap()).unwrap(), case["result"]);
        }
        assert_eq!(serde_json::to_value(state).unwrap(), case["after"]);
        assert_eq!(serde_json::to_value(config).unwrap(), case["input_after"]);
    }
}
