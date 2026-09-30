//! CPU generation and bounded prefetch of tracks for independent training cars.
use crate::game_random::GameRandom;
use crate::random_track::{self, GeneratedTrack, RandomTrackConfig, TrackGenerator};
use crate::vec2::V2;
use crate::world::World;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};

/// A fixed setting or uniformly sampled choices.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TrackChoice<T> {
    Fixed(T),
    Choices(Vec<T>),
}

impl<T: Clone> TrackChoice<T> {
    fn values(&self) -> &[T] {
        match self {
            Self::Fixed(value) => std::slice::from_ref(value),
            Self::Choices(values) => values,
        }
    }

    fn sample(&self, rng: &mut GameRandom) -> T {
        let values = self.values();
        values[rng.randrange(values.len())].clone()
    }
}

/// Tile count. Closed paths on the game's cardinal grid require even lengths.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TrackLength {
    Fixed(i32),
    Range { min: i32, max: i32 },
}

/// Fixed surface sets, explicit alternative sets, or a pool with sampled count.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TrackSurfaces {
    Fixed(Vec<usize>),
    Choices(Vec<Vec<usize>>),
    Random {
        pool: Vec<usize>,
        count: TrackChoice<usize>,
    },
}

impl TrackSurfaces {
    fn sample(&self, rng: &mut GameRandom) -> Vec<usize> {
        match self {
            Self::Fixed(surfaces) => surfaces.clone(),
            Self::Choices(sets) => sets[rng.randrange(sets.len())].clone(),
            Self::Random { pool, count } => {
                let count = count.sample(rng);
                let mut surfaces = pool.clone();
                for i in (1..surfaces.len()).rev() {
                    surfaces.swap(i, rng.randrange(i + 1));
                }
                surfaces.truncate(count);
                surfaces
            }
        }
    }

    fn validate(&self) -> Result<(), String> {
        let valid_set = |set: &[usize]| {
            set.iter().all(|&s| s <= 2) && (0..set.len()).all(|i| !set[..i].contains(&set[i]))
        };
        match self {
            Self::Fixed(set) if valid_set(set) => Ok(()),
            Self::Choices(sets) if !sets.is_empty() && sets.iter().all(|s| valid_set(s)) => Ok(()),
            Self::Random { pool, count } if !pool.is_empty() && valid_set(pool)
                && !count.values().is_empty() && count.values().iter().all(|&n| n > 0 && n <= pool.len()) => Ok(()),
            _ => Err("surfaces must use distinct IDs 0 (asphalt), 1 (dirt), 2 (ice); random counts must be in 1..=pool size and choices must not be empty".into()),
        }
    }
}

impl TrackLength {
    fn bounds(&self) -> (i32, i32) {
        match *self {
            Self::Fixed(length) => (length, length),
            Self::Range { min, max } => (min, max),
        }
    }
}

/// Training variants of the game's TrackFactory settings. Surface IDs are
/// asphalt = 0, dirt = 1, ice = 2; distribution is random = 0, sections = 1.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RandomTrainingTrackSettings {
    pub length: TrackLength,
    pub allow_double: TrackChoice<bool>,
    pub surfaces: TrackSurfaces,
    pub distribution: TrackChoice<usize>,
    pub start: Option<[i32; 2]>,
}

impl Default for RandomTrainingTrackSettings {
    fn default() -> Self {
        Self {
            length: TrackLength::Range { min: 12, max: 40 },
            allow_double: TrackChoice::Fixed(true),
            surfaces: TrackSurfaces::Random {
                pool: vec![0, 1, 2],
                count: TrackChoice::Choices(vec![1, 2, 3]),
            },
            distribution: TrackChoice::Fixed(1),
            start: None,
        }
    }
}

impl RandomTrainingTrackSettings {
    pub fn validate(&self) -> Result<(), String> {
        let (min, max) = self.length.bounds();
        let [x, y, width, height] = random_track::bounds();
        if min < 4 || max > width * height || min > max || min + min % 2 > max {
            return Err(format!(
                "length must contain an even tile count in 4..={} with min <= max",
                width * height
            ));
        }
        if self.allow_double.values().is_empty() || self.distribution.values().is_empty() {
            return Err("random track choices must not be empty".into());
        }
        self.surfaces.validate()?;
        if self.distribution.values().iter().any(|&d| d > 1) {
            return Err("distribution must be 0 (random) or 1 (sections)".into());
        }
        if self
            .start
            .is_some_and(|p| p[0] < x || p[0] >= x + width || p[1] < y || p[1] >= y + height)
        {
            return Err("start must be inside the game's track grid".into());
        }
        Ok(())
    }

    fn sample(&self, rng: &mut GameRandom) -> RandomTrackConfig {
        let (min, max) = self.length.bounds();
        let first = min + min % 2;
        let length = first + 2 * rng.randrange(((max - first) / 2 + 1) as usize) as i32;
        RandomTrackConfig {
            length,
            allow_double: self.allow_double.sample(rng),
            start: self.start,
            start_direction: None,
            surfaces: Some(self.surfaces.sample(rng)),
            distribution: self.distribution.sample(rng),
        }
    }
}

pub struct PreparedTrainingTrack {
    pub generation: u64,
    pub track: GeneratedTrack,
    pub world: Arc<World>,
    pub position: V2,
    pub rotation: f64,
    pub gpu_world: Option<crate::gpu::PreparedGpuWorld>,
}

// A generation owns its RNG state. Prefetch depth and process restarts cannot
// change track selection or consume the population's reproduction RNG.
fn track_state(seed: i64, generation: u64) -> [u64; 4] {
    let mut state = (seed as u64) ^ generation.wrapping_mul(0xd1342543de82ef95);
    std::array::from_fn(|_| {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    })
}

fn prepare(
    template: &Value,
    settings: &RandomTrainingTrackSettings,
    generator: Option<&TrackGenerator>,
    seed: i64,
    generation: u64,
) -> Result<PreparedTrainingTrack, String> {
    let mut rng = GameRandom::new(track_state(seed, generation), 0);
    // The game may shorten failed paths. Retry rather than violate the user's
    // minimum; keep both configuration sampling and generation bounded.
    for _ in 0..16 {
        let mut config = settings.sample(&mut rng);
        let mut state = rng.decision_state;
        let result = match generator {
            Some(generator) => generator.generate(&mut config, &mut state),
            None => random_track::generate(&mut config, &mut state),
        };
        rng.decision_state = state;
        if let Ok(track) = result {
            if track.config.length < settings.length.bounds().0 {
                continue;
            }
            let mut scene = track.to_scene(template);
            // CLI training uses independent cars on both backends. Native shared
            // TileMap redraw replay remains available through TrainingRunner.
            scene["track"]["native_broadphase"] = Value::Bool(false);
            let position = crate::world::vector(&scene["reset_position"]);
            let rotation = scene["reset_rotation"].as_f64().unwrap();
            return Ok(PreparedTrainingTrack {
                generation,
                track,
                world: Arc::new(World::from_scene(&scene)),
                position,
                rotation,
                gpu_world: None,
            });
        }
    }
    Err(format!("could not generate track for generation {generation} within the requested settings after 16 attempts"))
}

/// A dedicated CPU producer prepares geometry and spatial queries while the
/// simulator consumes tracks. The queue is bounded and never reuses a track
/// when empty: the consumer waits for the next generation or receives an error.
pub struct TrainingTrackBuffer {
    receiver: Option<mpsc::Receiver<Result<PreparedTrainingTrack, String>>>,
    worker: Option<JoinHandle<()>>,
}

impl TrainingTrackBuffer {
    pub fn new(
        template: Value,
        settings: RandomTrainingTrackSettings,
        seed: i64,
        first_generation: u64,
        capacity: usize,
        prepare_gpu: bool,
    ) -> Result<Self, String> {
        settings.validate()?;
        if capacity == 0 {
            return Err("track buffer size must be positive".into());
        }
        let hash_seed = template["runtime"]["hashcode_seed"]
            .as_u64()
            .map(u32::try_from)
            .transpose()
            .map_err(|_| "HashCode seed exceeds u32")?;
        let (sender, receiver) = mpsc::sync_channel(capacity);
        let worker = thread::Builder::new()
            .name("training-tracks".into())
            .spawn(move || {
                let generator = hash_seed.map(TrackGenerator::new);
                for generation in first_generation..u64::MAX {
                    let result =
                        prepare(&template, &settings, generator.as_ref(), seed, generation)
                            .and_then(|mut track| {
                                if prepare_gpu {
                                    track.gpu_world =
                                        Some(crate::gpu::PreparedGpuWorld::new(&track.world)?);
                                }
                                Ok(track)
                            });
                    let failed = result.is_err();
                    if sender.send(result).is_err() || failed {
                        break;
                    }
                }
            })
            .map_err(|e| format!("could not start CPU track producer: {e}"))?;
        Ok(Self {
            receiver: Some(receiver),
            worker: Some(worker),
        })
    }

    pub fn next_track(&self) -> Result<PreparedTrainingTrack, String> {
        self.receiver
            .as_ref()
            .unwrap()
            .recv()
            .map_err(|_| "CPU track producer stopped".to_owned())?
    }
}

impl Drop for TrainingTrackBuffer {
    fn drop(&mut self) {
        // Disconnect first to release a producer blocked on a full queue.
        self.receiver.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn default_sampling_covers_surface_counts_subsets_and_even_lengths() {
        let settings = RandomTrainingTrackSettings::default();
        settings.validate().unwrap();
        let mut rng = GameRandom::new(track_state(7, 0), 0);
        let mut seen = HashSet::new();
        let mut counts = [0; 3];
        for _ in 0..4096 {
            let config = settings.sample(&mut rng);
            assert!((12..=40).contains(&config.length) && config.length % 2 == 0);
            let mut surfaces = config.surfaces.unwrap();
            counts[surfaces.len() - 1] += 1;
            surfaces.sort();
            seen.insert(surfaces);
        }
        assert_eq!(seen.len(), 7, "all nonempty subsets should occur");
        assert!(
            counts.iter().all(|&count| (1100..1600).contains(&count)),
            "sample counts uniformly, rather than subsets: {counts:?}"
        );
    }

    #[test]
    fn settings_reject_invalid_lengths_choices_and_surface_counts() {
        for value in [
            serde_json::json!({"length": 5}),
            serde_json::json!({"length": {"min": 8, "max": 4}}),
            serde_json::json!({"length": {"min": 3, "max": 6}}),
            serde_json::json!({"length": 1000000}),
            serde_json::json!({"allow_double": []}),
            serde_json::json!({"distribution": [2]}),
            serde_json::json!({"surfaces": [3]}),
            serde_json::json!({"surfaces": [0, 0]}),
            serde_json::json!({"surfaces": {"pool": [0, 1], "count": [1, 3]}}),
            serde_json::json!({"surfaces": {"pool": [0, 1], "count": []}}),
            serde_json::json!({"start": [-1000, 0]}),
        ] {
            let settings: RandomTrainingTrackSettings =
                serde_json::from_value(value.clone()).unwrap();
            assert!(settings.validate().is_err(), "accepted {value}");
        }
        let settings: RandomTrainingTrackSettings = serde_json::from_value(serde_json::json!({
            "length": {"min": 5, "max": 9}, "surfaces": [[0], [1, 2]], "allow_double": [false, true], "distribution": [0, 1]
        })).unwrap();
        settings.validate().unwrap();
        let mut rng = GameRandom::new(track_state(9, 0), 0);
        for _ in 0..100 {
            let config = settings.sample(&mut rng);
            assert!([6, 8].contains(&config.length));
            assert!([vec![0], vec![1, 2]].contains(&config.surfaces.unwrap()));
        }
    }
}
