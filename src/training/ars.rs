//! Augmented Random Search (Mania, Guy and Recht 2018, V2-t) with an elite pool.
//!
//! The search keeps one point `theta` in parameter space. Each population is
//!
//! ```text
//! [elite 0 .. elite E-1] [theta] [theta + nu*d0, theta - nu*d0] [theta + nu*d1, ...] ([theta])
//! ```
//!
//! The elites are the best distinct cars so far, run again unchanged; each
//! `d` is a standard normal direction and the trailing `theta` appears when
//! the population size leaves one slot over. After the generation runs, the
//! scores (`reward_values`, so rank based) steer one step: the `top_frac` best
//! pairs by `max(r+, r-)` move `theta` by
//! `alpha / (b * nu) * sum (r+ - r-) * d`. If those scores are all alike the
//! step is skipped.
//!
//! Scores are ranks within one generation, so they cannot be compared across
//! generations. The elite pool needs no such comparison: its members are in
//! every population, and the next pool is the best distinct cars of that
//! population, old elites and probes alike.

use super::optimizer::{Optimizer, Produced, Record};
use super::TrainingRandom;
use crate::math::profile::MathProfile;
use crate::nn::network::{parameter_count, Network};
use crate::training::evolution::{reward_values, AgentResult, ArsSettings, EvolutionSettings, Generation};
use rayon::prelude::*;

/// Rejects settings that `Ars` cannot run with.
pub(crate) fn validate(s: &ArsSettings, population: usize) -> Result<(), String> {
    let positive = |x: f64| x.is_finite() && x > 0.0;
    if !positive(s.nu) || !positive(s.alpha) {
        return Err("ars nu and alpha must be positive".into());
    }
    if !(s.top_frac > 0.0 && s.top_frac <= 1.0) {
        return Err("ars top_frac must be in (0, 1]".into());
    }
    if !s.max_weight.is_finite() || s.max_weight < 0.0 {
        return Err("ars max_weight must be 0 (off) or positive".into());
    }
    if population < s.elite_count.saturating_add(3) {
        return Err(format!(
            "ars needs population >= elite_count + 3 (the elites, the search point and one pair), got {population} with {} elites",
            s.elite_count
        ));
    }
    Ok(())
}

/// The settings that shape a population, as checkpoints store them.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Sampling {
    pub(crate) population: usize,
    pub(crate) elite_count: usize,
    pub(crate) nu: f64,
    pub(crate) max_weight: f64,
}

impl Sampling {
    fn of(s: &EvolutionSettings) -> Sampling {
        Sampling {
            population: s.population,
            elite_count: s.ars.elite_count,
            nu: s.ars.nu,
            max_weight: s.ars.max_weight,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Elite {
    pub(crate) params: Vec<f64>,
    /// The reward it had in the generation that selected it.
    pub(crate) score: f64,
}

/// Where each kind of car sits in the population `Ars` produced last.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    elites: usize,
    pairs: usize,
    size: usize,
}

#[derive(Clone, Default)]
pub(crate) struct Ars {
    shape: Vec<usize>,
    /// The search point; empty until the first population is made.
    theta: Vec<f64>,
    /// Best first.
    pool: Vec<Elite>,
    layout: Option<Layout>,
    /// The noise scale the last population was sampled with; `learn` divides
    /// by it, whatever the settings say by then.
    nu: f64,
    /// The pairs' directions of the last population, one row of `theta.len()` each.
    deltas: Vec<f64>,
}

/// An `Ars` before it sampled a population: sampling again from `rng` repeats it.
#[derive(Clone)]
pub(crate) struct ArsRecord {
    pub(crate) shape: Vec<usize>,
    pub(crate) theta: Vec<f64>,
    pub(crate) pool: Vec<Elite>,
    pub(crate) sampling: Sampling,
    /// The generator before the directions were drawn.
    pub(crate) rng: TrainingRandom,
    /// The math the game's streams draw the directions with.
    pub(crate) math: MathProfile,
}

impl ArsRecord {
    /// Whether `rebuild` can sample this record; a checkpoint may be corrupt.
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.shape.len() < 2 || self.shape.contains(&0) {
            return Err("the checkpoint has an invalid network shape".into());
        }
        let size = self.shape.windows(2).try_fold(0usize, |sum, w| {
            w[0].checked_add(1)?.checked_mul(w[1])?.checked_add(sum)
        });
        if Some(self.theta.len()) != size || self.pool.iter().any(|e| Some(e.params.len()) != size) {
            return Err("the checkpoint has the wrong number of parameters".into());
        }
        let s = &self.sampling;
        let positive = |x: f64| x.is_finite() && x > 0.0;
        if !positive(s.nu) || !s.max_weight.is_finite() || s.max_weight < 0.0 {
            return Err("the checkpoint's ars noise or weight limit is out of range".into());
        }
        if s.population < self.pool.len().min(s.elite_count).saturating_add(3) {
            return Err("the checkpoint's population is too small for its elites".into());
        }
        Ok(())
    }

    /// The sampled population, the generator after it, and the search state.
    /// Call `validate` first.
    pub(crate) fn rebuild(&self) -> (Vec<Network>, TrainingRandom, Ars) {
        let mut ars = Ars {
            shape: self.shape.clone(),
            theta: self.theta.clone(),
            pool: self.pool.clone(),
            ..Ars::default()
        };
        let mut rng = self.rng.clone();
        let networks = ars.sample(&self.sampling, &mut rng, self.math);
        (networks, rng, ars)
    }
}

impl Ars {
    /// Draws the directions and builds the population around `theta`.
    fn sample(&mut self, sampling: &Sampling, rng: &mut TrainingRandom, math: MathProfile) -> Vec<Network> {
        let n = self.theta.len();
        let elites = sampling.elite_count.min(self.pool.len());
        assert!(
            sampling.population >= elites + 3,
            "ars needs population >= elite_count + 3"
        );
        let pairs = (sampling.population - elites - 1) / 2;
        if sampling.max_weight > 0.0 {
            let limit = sampling.max_weight;
            let clamp_all = |v: &mut Vec<f64>| v.iter_mut().for_each(|x| *x = x.clamp(-limit, limit));
            clamp_all(&mut self.theta);
            self.pool.iter_mut().for_each(|e| clamp_all(&mut e.params));
        }
        self.nu = sampling.nu;
        rng.standard_normals_into(pairs * n, &mut self.deltas, math);
        let (theta, deltas, nu, limit) = (&self.theta, &self.deltas, sampling.nu, sampling.max_weight);
        let clamp = move |v: f64| if limit > 0.0 { v.clamp(-limit, limit) } else { v };
        let probes: Vec<[Vec<f64>; 2]> = (0..pairs)
            .into_par_iter()
            .map(|i| {
                let d = &deltas[i * n..(i + 1) * n];
                let side =
                    |sign: f64| -> Vec<f64> { theta.iter().zip(d).map(|(&t, &z)| clamp(t + sign * nu * z)).collect() };
                [side(1.0), side(-1.0)]
            })
            .collect();
        let mut cars: Vec<Vec<f64>> = Vec::with_capacity(sampling.population);
        cars.extend(self.pool[..elites].iter().map(|e| e.params.clone()));
        cars.push(theta.clone());
        cars.extend(probes.into_iter().flatten());
        while cars.len() < sampling.population {
            cars.push(theta.clone());
        }
        self.layout = Some(Layout {
            elites,
            pairs,
            size: sampling.population,
        });
        let shape = &self.shape;
        cars.into_par_iter().map(|p| Network::from_vector(shape, p)).collect()
    }

    /// Whether `agents` are the population of the last `sample`.
    fn matches(&self, agents: &[AgentResult]) -> bool {
        let (Some(layout), Some(first)) = (self.layout, agents.first()) else {
            return false;
        };
        layout.size == agents.len()
            && first.network.shape == self.shape
            && self.theta.len() == first.network.params.len()
            && self.deltas.len() == layout.pairs * self.theta.len()
    }

    /// The new elite pool and the step of `theta`, from the scored population.
    fn learn(&mut self, agents: &[AgentResult], scores: &[f64], settings: &EvolutionSettings) {
        let Layout { elites, pairs, .. } = self.layout.expect("learn follows sample");
        let ars = &settings.ars;

        let mut order: Vec<usize> = (0..agents.len()).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut pool: Vec<Elite> = Vec::with_capacity(ars.elite_count);
        for &i in &order {
            if pool.len() >= ars.elite_count {
                break;
            }
            let params = &agents[i].network.params;
            if !pool.iter().any(|e| e.params == *params) {
                pool.push(Elite {
                    params: params.clone(),
                    score: scores[i],
                });
            }
        }
        self.pool = pool;

        // Probe i is at `elites + 1 + 2i` (plus side) and the next slot (minus side).
        let mut ranked: Vec<(usize, f64, f64)> = (0..pairs)
            .map(|i| (i, scores[elites + 1 + 2 * i], scores[elites + 2 + 2 * i]))
            .collect();
        ranked.sort_by(|a, b| {
            let (ka, kb) = (a.1.max(a.2), b.1.max(b.2));
            kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal)
        });
        let used = ((ars.top_frac * pairs as f64).ceil() as usize).clamp(1, pairs);
        ranked.truncate(used);

        let count = (2 * used) as f64;
        let mean = ranked.iter().map(|r| r.1 + r.2).sum::<f64>() / count;
        let variance = ranked
            .iter()
            .map(|r| (r.1 - mean).powi(2) + (r.2 - mean).powi(2))
            .sum::<f64>()
            / count;
        if variance.sqrt() < 1e-6 {
            return;
        }
        let scale = ars.alpha / (used as f64 * self.nu);
        let n = self.theta.len();
        for (i, plus, minus) in ranked {
            let weight = scale * (plus - minus);
            for (t, &z) in self.theta.iter_mut().zip(&self.deltas[i * n..(i + 1) * n]) {
                *t += weight * z;
            }
        }
        if ars.max_weight > 0.0 {
            let limit = ars.max_weight;
            self.theta.iter_mut().for_each(|t| *t = t.clamp(-limit, limit));
        }
    }

    fn produce(
        &mut self,
        scores: Vec<f64>,
        settings: &EvolutionSettings,
        rng: &mut TrainingRandom,
        math: MathProfile,
        trace: bool,
    ) -> Produced {
        let sampling = Sampling::of(settings);
        let before = trace.then(|| ArsRecord {
            shape: self.shape.clone(),
            theta: self.theta.clone(),
            pool: self.pool.clone(),
            sampling: sampling.clone(),
            rng: rng.clone(),
            math,
        });
        let networks = self.sample(&sampling, rng, math);
        Produced {
            generation: Generation {
                networks,
                preserved_count: self.layout.map_or(0, |l| l.elites + 1),
                rewards: scores,
            },
            record: before.map(Record::Ars),
        }
    }
}

impl Optimizer for Ars {
    fn name(&self) -> &'static str {
        "ars"
    }

    fn boxed_clone(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }

    fn invalidate(&mut self) {
        // `matches` fails from here on, so the next turnover searches on from the best car.
        self.layout = None;
        self.deltas.clear();
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
        *self = Ars {
            shape: seed.shape.clone(),
            theta: seed.params.clone(),
            ..Ars::default()
        };
        debug_assert_eq!(self.theta.len(), parameter_count(&self.shape));
        self.produce(scores, settings, rng, math, trace)
    }

    fn next(
        &mut self,
        agents: &[AgentResult],
        scores: Option<Vec<f64>>,
        settings: &EvolutionSettings,
        rng: &mut TrainingRandom,
        _scratch: &mut Vec<f64>,
        math: MathProfile,
        trace: bool,
    ) -> Produced {
        let scores = scores.unwrap_or_else(|| reward_values(agents, &settings.rewards));
        if self.matches(agents) {
            self.learn(agents, &scores, settings);
        } else {
            // A new run, a restored population or a changed population size:
            // the directions no longer belong to these cars. Search on from
            // the best one instead.
            let best = (1..scores.len()).fold(0, |best, i| if scores[i] > scores[best] { i } else { best });
            let network = agents[best].network;
            *self = Ars {
                shape: network.shape.clone(),
                theta: network.params.clone(),
                ..Ars::default()
            };
        }
        self.produce(scores, settings, rng, math, trace)
    }
}

#[cfg(test)]
mod tests;
