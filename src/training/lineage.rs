//! Checkpoint RNG streams and reproducible parent lineage.

use crate::math::profile::MathProfile;
use crate::nn::network::Network;
use crate::training::evolution::{reproduce, AgentResult, Breeding, Choice, EvolutionSettings, Generation, CROSSOVERS};
use crate::training::pyrandom::PyRandom;
use serde_json::Value;

/// Untagged checkpoints retain the historical Python backend. Game replay
/// checkpoints carry both independent .NET streams under an explicit tag.
#[derive(Clone)]
// Keep the public variants and their RNG state inline; boxing changes the API and adds an allocation.
#[allow(clippy::large_enum_variant)]
pub enum TrainingRandom {
    Python(PyRandom),
    Game(crate::training::game_random::GameRandom),
}
impl From<PyRandom> for TrainingRandom {
    fn from(r: PyRandom) -> Self {
        Self::Python(r)
    }
}
impl From<crate::training::game_random::GameRandom> for TrainingRandom {
    fn from(r: crate::training::game_random::GameRandom) -> Self {
        Self::Game(r)
    }
}
impl TrainingRandom {
    pub fn to_json(&self) -> Value {
        match self {
            Self::Python(r) => r.to_json(),
            Self::Game(r) => r.to_json(),
        }
    }
    pub fn from_json(v: &Value) -> Self {
        if v["backend"].as_str() == Some("game") {
            Self::Game(crate::training::game_random::GameRandom::from_json(v))
        } else {
            Self::Python(PyRandom::from_json(v))
        }
    }
    /// The game's streams use `math` (.NET Math.Log); the Python backend
    /// reproduces the historical trainer, which ran on musl's.
    pub fn reproduce(&mut self, agents: &[AgentResult], settings: &EvolutionSettings, math: MathProfile) -> Generation {
        match self {
            Self::Python(r) => reproduce(agents, settings, r),
            Self::Game(r) => crate::training::evolution::reproduce_game(agents, settings, r, math),
        }
    }
    pub(super) fn reproduce_scored(
        &mut self,
        agents: &[AgentResult],
        scores: Vec<f64>,
        settings: &EvolutionSettings,
        scratch: &mut Vec<f64>,
        math: MathProfile,
    ) -> Generation {
        match self {
            Self::Python(r) => {
                crate::training::evolution::reproduce_scored_with_scratch(agents, scores, settings, r, scratch)
            }
            Self::Game(r) => crate::training::evolution::reproduce_game_scored(agents, scores, settings, r, math),
        }
    }
    /// The next `count` standard normal variates, in `out`.
    pub(crate) fn standard_normals_into(&mut self, count: usize, out: &mut Vec<f64>, math: MathProfile) {
        match self {
            Self::Python(r) => r.standard_normals_into(count, out),
            Self::Game(r) => *out = r.normal_array(count, 1.0, math),
        }
    }
    pub fn xavier(&mut self, shape: &[usize], math: MathProfile) -> Network {
        match self {
            Self::Python(r) => Network::xavier(shape, r),
            Self::Game(r) => Network::xavier_game(shape, r, math),
        }
    }
    pub(crate) fn choose(&mut self, scores: &[f64], settings: &EvolutionSettings) -> Choice {
        match self {
            Self::Python(r) => crate::training::evolution::choose(scores, settings, r),
            Self::Game(r) => crate::training::evolution::choose(scores, settings, r),
        }
    }
    pub(crate) fn breed(
        &mut self,
        selected: &[&Network],
        preserved: Vec<Network>,
        settings: &EvolutionSettings,
        math: MathProfile,
        profile: &mut crate::training::training_profile::Profile,
    ) -> Vec<Network> {
        match self {
            Self::Python(r) => {
                crate::training::evolution::breed(selected, preserved, settings, r, &mut Vec::new(), profile)
            }
            Self::Game(r) => crate::training::evolution::breed_game(selected, preserved, settings, r, math, profile),
        }
    }
}

/// One reproduction as checkpoints store it: the distinct parents, which of
/// them were selected and preserved, the breeding settings, and the
/// generator state after selection. `rebuild` repeats the breeding.
#[derive(Clone)]
pub(crate) struct Lineage {
    /// Distinct parents in first use: the best car of the scored population
    /// first, then preserved cars, then selected cars.
    pub(crate) parents: Vec<Network>,
    /// Indices into `parents`, in draw order; repeats are kept.
    pub(crate) selected: Vec<u32>,
    /// Indices into `parents`, best first.
    pub(crate) preserved: Vec<u32>,
    pub(crate) breeding: Breeding,
    /// The generator state after selection, before crossover.
    pub(crate) rng: TrainingRandom,
    /// The math the game's streams draw with.
    pub(crate) math: MathProfile,
}

impl Lineage {
    fn new(
        agents: &[AgentResult],
        choice: &Choice,
        breeding: Breeding,
        rng: TrainingRandom,
        math: MathProfile,
    ) -> Lineage {
        let mut slots = std::collections::HashMap::new();
        let mut parents = Vec::new();
        let mut slot = |car: usize| -> u32 {
            *slots.entry(car).or_insert_with(|| {
                parents.push(agents[car].network.clone());
                (parents.len() - 1) as u32
            })
        };
        slot(choice.best);
        let preserved = choice.preserved.iter().map(|&car| slot(car)).collect();
        let selected = choice.selected.iter().map(|&car| slot(car)).collect();
        Lineage {
            parents,
            selected,
            preserved,
            breeding,
            rng,
            math,
        }
    }

    /// Whether `rebuild` can breed this lineage; a checkpoint may be corrupt.
    pub(crate) fn validate(&self) -> Result<(), String> {
        let b = &self.breeding;
        let Some(first) = self.parents.first() else {
            return Err("the checkpoint has no networks".into());
        };
        if self.parents.iter().any(|p| p.shape != first.shape) {
            return Err("the checkpoint's parents differ in shape".into());
        }
        if first.shape.len() < 2 || first.shape.contains(&0) {
            return Err("the checkpoint's parents have an invalid shape".into());
        }
        // Checked, since a shape from a checkpoint may be huge (usize is 32 bits on wasm).
        let expected = first.shape.windows(2).try_fold(0usize, |sum, w| {
            w[0].checked_add(1)?.checked_mul(w[1])?.checked_add(sum)
        });
        if self.parents.iter().any(|p| Some(p.params.len()) != expected) {
            return Err("the checkpoint's parents have the wrong number of parameters".into());
        }
        if b.population == 0 {
            return Err("the checkpoint has no cars".into());
        }
        if self
            .selected
            .iter()
            .chain(&self.preserved)
            .any(|&i| i as usize >= self.parents.len())
        {
            return Err("a checkpoint parent index is out of range".into());
        }
        if self.preserved.len() > b.population {
            return Err("the checkpoint keeps more parents than it has cars".into());
        }
        if self.selected.is_empty() && self.preserved.len() < b.population {
            return Err("the checkpoint selects no parents".into());
        }
        if !CROSSOVERS.contains(&b.crossover.as_str()) {
            return Err("the checkpoint has an unknown crossover".into());
        }
        if !(0.0..=10.0).contains(&b.mutation_rate) || !(0.0..=1.0).contains(&b.weight_decay) {
            return Err("the checkpoint's mutation rate or weight decay is out of range".into());
        }
        Ok(())
    }

    /// The bred population and the generator state after it. Call `validate` first.
    pub(crate) fn rebuild(&self) -> (Vec<Network>, TrainingRandom) {
        let mut rng = self.rng.clone();
        let selected: Vec<&Network> = self.selected.iter().map(|&i| &self.parents[i as usize]).collect();
        let preserved = self
            .preserved
            .iter()
            .map(|&i| self.parents[i as usize].clone())
            .collect();
        let mut profile = crate::training::training_profile::Profile::new("rebuild");
        let networks = rng.breed(&selected, preserved, &self.breeding.settings(), self.math, &mut profile);
        (networks, rng)
    }
}

/// `TrainingRandom::reproduce_scored`, also returning the reproduction as a
/// `Lineage`. `agents` must not be empty.
pub(super) fn reproduce_traced(
    rng: &mut TrainingRandom,
    agents: &[AgentResult],
    scores: Vec<f64>,
    settings: &EvolutionSettings,
    math: MathProfile,
) -> (Generation, Lineage) {
    crate::training::evolution::validate_scores(agents, &scores);
    assert!(!agents.is_empty(), "a training generation needs at least one network");
    let mut profile = crate::training::training_profile::Profile::new("reproduce_traced");
    let choice = rng.choose(&scores, settings);
    let lineage = Lineage::new(agents, &choice, Breeding::of(settings), rng.clone(), math);
    let selected: Vec<&Network> = choice.selected.iter().map(|&i| agents[i].network).collect();
    let preserved: Vec<Network> = choice.preserved.iter().map(|&i| agents[i].network.clone()).collect();
    let preserved_count = preserved.len();
    profile.mark("selection");
    let networks = rng.breed(&selected, preserved, settings, math, &mut profile);
    (
        Generation {
            networks,
            preserved_count,
            rewards: scores,
        },
        lineage,
    )
}
