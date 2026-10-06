//! Rebuild game-capture scenes from fixed public generator recipes.
use altd_sim::track::training_tracks::{training_scene_at, RandomTrainingTrackSettings};
use serde_json::Value;

fn merge(target: &mut Value, overrides: &Value) {
    if let (Some(target), Some(overrides)) = (target.as_object_mut(), overrides.as_object()) {
        for (key, value) in overrides {
            if let Some(existing) = target.get_mut(key) {
                merge(existing, value);
            } else {
                target.insert(key.clone(), value.clone());
            }
        }
    } else {
        *target = overrides.clone();
    }
}

pub(crate) fn scene(data: &Value) -> Value {
    let Some(recipe) = data.get("recipe") else {
        return data["scene"].clone();
    };
    let settings: RandomTrainingTrackSettings =
        serde_json::from_str(include_str!("../../assets/random_track_settings.json")).unwrap();
    let template: Value = serde_json::from_str(include_str!("../../assets/scenes/rally_template.json")).unwrap();
    let (_, mut scene) = training_scene_at(
        &template,
        &settings,
        None,
        recipe["seed"].as_i64().unwrap(),
        recipe["generation"].as_u64().unwrap(),
        recipe["slot"].as_u64().unwrap() as usize,
        altd_sim::math::profile::MathProfile::Proton,
    )
    .unwrap();
    if let Some(overrides) = data.get("scene_overrides") {
        merge(&mut scene, overrides);
    }
    if let Some(order) = data.get("physics_shape_order").and_then(Value::as_array) {
        let shapes = scene["track"]["physics_shapes"].as_array().unwrap().clone();
        scene["track"]["physics_shapes"] = order
            .iter()
            .map(|index| shapes[index.as_u64().unwrap() as usize].clone())
            .collect();
    }
    if let Some(events) = scene.get_mut("redraw_events").and_then(Value::as_array_mut) {
        for event in events {
            if event["scene"].get("recipe").is_some() {
                event["scene"] = self::scene(&event["scene"]);
            }
        }
    }
    scene
}
