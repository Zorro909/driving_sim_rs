//! Population optimizers: how one generation of cars becomes the next.
//!
//! The runner evaluates a whole population, scores it, and hands both to an
//! `Optimizer`, which returns the networks of the next population. The GA
//! (`Ga`) breeds from the scored cars alone. Others, such as `Ars`, keep a
//! search state of their own between generations and use the cars only for
//! their scores.

use super::ars::{Ars, ArsRecord};
use super::lineage::reproduce_traced;
use super::{Lineage, TrainingRandom};
use crate::math::profile::MathProfile;
use crate::nn::network::Network;
use crate::training::evolution::{reward_values, AgentResult, EvolutionSettings, Generation};

/// What a checkpoint needs to rebuild the population an optimizer produced.
#[derive(Clone)]
// Records are moved once per generation; boxing the larger variant buys nothing.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Record {
    /// The parents the GA bred from.
    Lineage(Lineage),
    /// The search state ARS sampled from.
    Ars(ArsRecord),
}

impl Record {
    #[cfg(test)]
    pub(crate) fn into_lineage(self) -> Option<Lineage> {
        match self {
            Record::Lineage(lineage) => Some(lineage),
            Record::Ars(_) => None,
        }
    }
}

/// A new population, with the record of how it was made when asked for.
pub(crate) struct Produced {
    pub(crate) generation: Generation,
    pub(crate) record: Option<Record>,
}

pub(crate) trait Optimizer: Send + Sync {
    /// The `settings.algorithm` this optimizer implements.
    fn name(&self) -> &'static str;

    /// The first population, built from `seed`. Forgets any earlier state.
    /// `math` is the session's profile, which the game's random streams draw with.
    fn start(
        &mut self,
        seed: &Network,
        settings: &EvolutionSettings,
        rng: &mut TrainingRandom,
        math: MathProfile,
        trace: bool,
    ) -> Produced;

    /// The population after `agents`, the cars of the finished generation in
    /// population order. `scores` are their rewards (`reward_values` when
    /// absent). `scratch` is reusable storage for random draws. With `trace`,
    /// the result carries the record a checkpoint rebuilds it from.
    #[allow(clippy::too_many_arguments)] // the call mirrors `start`, plus the cars and scratch
    fn next(
        &mut self,
        agents: &[AgentResult],
        scores: Option<Vec<f64>>,
        settings: &EvolutionSettings,
        rng: &mut TrainingRandom,
        scratch: &mut Vec<f64>,
        math: MathProfile,
        trace: bool,
    ) -> Produced;

    /// A copy of the search state, to put back when the population `next`
    /// produced cannot be installed.
    fn boxed_clone(&self) -> Box<dyn Optimizer>;

    /// The installed cars were edited from outside: forget whatever ties the
    /// optimizer's state to the population it produced.
    #[allow(dead_code)] // only the test and wasm `setNetworkJson` edit installed cars
    fn invalidate(&mut self) {}
}

/// The optimizer for `algorithm`. Unknown names panic, as the other settings do.
pub(crate) fn create(algorithm: &str) -> Box<dyn Optimizer> {
    match algorithm {
        "ga" => Box::new(Ga),
        "ars" => Box::<Ars>::default(),
        other => panic!("unknown algorithm: {other}"),
    }
}

/// Selection, crossover and mutation (`training::evolution`).
pub(crate) struct Ga;

impl Ga {
    fn produce(
        agents: &[AgentResult],
        scores: Vec<f64>,
        settings: &EvolutionSettings,
        rng: &mut TrainingRandom,
        scratch: &mut Vec<f64>,
        math: MathProfile,
        trace: bool,
    ) -> Produced {
        if trace {
            let (generation, lineage) = reproduce_traced(rng, agents, scores, settings, math);
            Produced {
                generation,
                record: Some(Record::Lineage(lineage)),
            }
        } else {
            Produced {
                generation: rng.reproduce_scored(agents, scores, settings, scratch, math),
                record: None,
            }
        }
    }
}

impl Optimizer for Ga {
    fn name(&self) -> &'static str {
        "ga"
    }

    fn boxed_clone(&self) -> Box<dyn Optimizer> {
        Box::new(Ga)
    }

    fn start(
        &mut self,
        seed: &Network,
        settings: &EvolutionSettings,
        rng: &mut TrainingRandom,
        math: MathProfile,
        trace: bool,
    ) -> Produced {
        let seed_result = [AgentResult {
            network: seed,
            metrics: [None; 14],
            update_count: 0,
        }];
        let scores = reward_values(&seed_result, &settings.rewards);
        Self::produce(&seed_result, scores, settings, rng, &mut Vec::new(), math, trace)
    }

    fn next(
        &mut self,
        agents: &[AgentResult],
        scores: Option<Vec<f64>>,
        settings: &EvolutionSettings,
        rng: &mut TrainingRandom,
        scratch: &mut Vec<f64>,
        math: MathProfile,
        trace: bool,
    ) -> Produced {
        let scores = scores.unwrap_or_else(|| reward_values(agents, &settings.rewards));
        Self::produce(agents, scores, settings, rng, scratch, math, trace)
    }
}
