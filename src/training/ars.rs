//! Augmented Random Search (Mania, Guy and Recht 2018, V2-t) with an elite pool.
//!
//! The search keeps `heads` points `theta` in parameter space (one by
//! default). Each population is
//!
//! ```text
//! [elite 0 .. elite E-1] [theta 0 .. theta H-1] [theta_0 + nu*d0, theta_0 - nu*d0] [theta_0 + nu*d1, ...] ... ([theta 0])
//! ```
//!
//! The elites are the best distinct cars so far, run again unchanged; each
//! `d` is a standard normal direction and the trailing `theta` appears when
//! the population size leaves one slot over. The probe pairs are split evenly
//! among the heads, the remainder going to head 0. After the generation runs,
//! the scores (`reward_values`, so rank based) steer one step of each head: its
//! `top_frac` best pairs by `max(r+, r-)` move its `theta` by
//! `alpha / (b * nu) * sum (r+ - r-) * d`. If those scores are all alike the
//! step is skipped.
//!
//! Each head adapts its own `alpha`. Every `alpha_adapt_every` generations the
//! fast and slow moving averages of its search point's raw reward are
//! compared: a clearly rising trend multiplies `alpha` by `alpha_adapt_up`, a
//! clearly falling one by `alpha_adapt_down`, within `[alpha_min, alpha_max]`.
//! The raw reward (`raw_reward_values`) is used because ranks cannot be
//! compared across generations.
//!
//! The elite pool needs no such comparison: its members are in every
//! population, and the next pool is the best distinct cars of that
//! population, old elites and probes alike.

use super::optimizer::{Optimizer, Produced, Record};
use super::TrainingRandom;
use crate::math::profile::MathProfile;
use crate::nn::network::{parameter_count, Network};
use crate::training::evolution::{
    raw_reward_values, reward_values, AgentResult, ArsSettings, EvolutionSettings, Generation,
};
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
    if s.heads == 0 {
        return Err("ars heads must be at least 1".into());
    }
    if !positive(s.alpha_min) || !positive(s.alpha_max) || s.alpha_min > s.alpha_max {
        return Err("ars alpha_min and alpha_max must be positive, with alpha_min <= alpha_max".into());
    }
    if s.alpha_adapt_every == 0 {
        return Err("ars alpha_adapt_every must be at least 1".into());
    }
    let decay = |x: f64| x > 0.0 && x <= 1.0;
    if !decay(s.alpha_adapt_fast) || !decay(s.alpha_adapt_slow) {
        return Err("ars alpha_adapt_fast and alpha_adapt_slow must be in (0, 1]".into());
    }
    if !s.alpha_adapt_threshold.is_finite() || s.alpha_adapt_threshold < 0.0 {
        return Err("ars alpha_adapt_threshold must be 0 or positive".into());
    }
    if !positive(s.alpha_adapt_up) || !positive(s.alpha_adapt_down) {
        return Err("ars alpha_adapt_up and alpha_adapt_down must be positive".into());
    }
    if population < s.elite_count.saturating_add(s.heads.saturating_mul(3)) {
        return Err(format!(
            "ars needs population >= elite_count + 3 * heads (the elites, then per head the search point and one pair), got {population} with {} elites and {} heads",
            s.elite_count, s.heads
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
    pub(crate) heads: usize,
}

impl Sampling {
    fn of(s: &EvolutionSettings) -> Sampling {
        Sampling {
            population: s.population,
            elite_count: s.ars.elite_count,
            nu: s.ars.nu,
            max_weight: s.ars.max_weight,
            heads: s.ars.heads,
        }
    }
}

/// One search point and the state that adapts its step size.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Head {
    pub(crate) theta: Vec<f64>,
    /// The adapted step size; `None` until the first step takes it from the settings.
    pub(crate) alpha: Option<f64>,
    /// Fast and slow moving averages of the search point's raw reward.
    pub(crate) fast: Option<f64>,
    pub(crate) slow: Option<f64>,
    /// Generations since the last trend check.
    pub(crate) since_check: u32,
}

impl Head {
    fn new(theta: Vec<f64>) -> Head {
        Head {
            theta,
            alpha: None,
            fast: None,
            slow: None,
            since_check: 0,
        }
    }

    /// Takes in the raw reward of this generation's search point and, on
    /// every `alpha_adapt_every`-th call, moves the step size along the trend.
    /// A reward that is not finite (a missing metric) leaves the averages alone.
    fn adapt(&mut self, reward: f64, ars: &ArsSettings) {
        let bound = |alpha: f64| alpha.clamp(ars.alpha_min, ars.alpha_max);
        let mut alpha = bound(self.alpha.unwrap_or(ars.alpha));
        if reward.is_finite() {
            let blend = |old: Option<f64>, decay: f64| Some(old.map_or(reward, |e| (1.0 - decay) * e + decay * reward));
            self.fast = blend(self.fast, ars.alpha_adapt_fast);
            self.slow = blend(self.slow, ars.alpha_adapt_slow);
        }
        self.since_check += 1;
        if self.since_check >= ars.alpha_adapt_every {
            self.since_check = 0;
            if let (Some(fast), Some(slow)) = (self.fast, self.slow) {
                let (trend, band) = (fast - slow, ars.alpha_adapt_threshold * slow.abs());
                if trend > band {
                    alpha = bound(alpha * ars.alpha_adapt_up);
                } else if trend < -band {
                    alpha = bound(alpha * ars.alpha_adapt_down);
                }
            }
        }
        self.alpha = Some(alpha);
    }
}

/// The probe pairs `[start, start + count)` of head `head` when `pairs` are
/// split among `heads`: evenly, the remainder going to head 0.
fn pair_range(pairs: usize, heads: usize, head: usize) -> (usize, usize) {
    let (base, extra) = (pairs / heads, pairs % heads);
    if head == 0 {
        (0, base + extra)
    } else {
        (extra + head * base, base)
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
    heads: usize,
    pairs: usize,
    size: usize,
}

#[derive(Clone, Default)]
pub(crate) struct Ars {
    shape: Vec<usize>,
    /// The search points; empty until the first population is made.
    heads: Vec<Head>,
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
    pub(crate) heads: Vec<Head>,
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
        if self.heads.is_empty() || self.heads.len() != self.sampling.heads {
            return Err("the checkpoint has the wrong number of search points".into());
        }
        if self.heads.iter().any(|h| Some(h.theta.len()) != size)
            || self.pool.iter().any(|e| Some(e.params.len()) != size)
        {
            return Err("the checkpoint has the wrong number of parameters".into());
        }
        if self.heads.iter().any(|h| h.theta.iter().any(|p| !p.is_finite()))
            || self
                .pool
                .iter()
                .any(|e| !e.score.is_finite() || e.params.iter().any(|p| !p.is_finite()))
        {
            return Err("non-finite checkpoint parameter or elite score".into());
        }
        let s = &self.sampling;
        let positive = |x: f64| x.is_finite() && x > 0.0;
        if !positive(s.nu) || !s.max_weight.is_finite() || s.max_weight < 0.0 {
            return Err("the checkpoint's ars noise or weight limit is out of range".into());
        }
        let adapted = |x: Option<f64>| x.is_none_or(f64::is_finite);
        if self
            .heads
            .iter()
            .any(|h| !h.alpha.is_none_or(positive) || !adapted(h.fast) || !adapted(h.slow))
        {
            return Err("the checkpoint's step size state is out of range".into());
        }
        if s.population
            < self
                .pool
                .len()
                .min(s.elite_count)
                .saturating_add(self.heads.len().saturating_mul(3))
        {
            return Err("the checkpoint's population is too small for its elites and search points".into());
        }
        Ok(())
    }

    /// The sampled population, the generator after it, and the search state.
    /// Call `validate` first.
    pub(crate) fn rebuild(&self) -> (Vec<Network>, TrainingRandom, Ars) {
        let mut ars = Ars {
            shape: self.shape.clone(),
            heads: self.heads.clone(),
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
        self.fit_heads(sampling.heads);
        let heads = self.heads.len();
        let n = self.heads[0].theta.len();
        let elites = sampling.elite_count.min(self.pool.len());
        assert!(
            sampling.population >= elites + 3 * heads,
            "ars needs population >= elite_count + 3 * heads"
        );
        let pairs = (sampling.population - elites - heads) / 2;
        if sampling.max_weight > 0.0 {
            let limit = sampling.max_weight;
            let clamp_all = |v: &mut Vec<f64>| v.iter_mut().for_each(|x| *x = x.clamp(-limit, limit));
            self.heads.iter_mut().for_each(|h| clamp_all(&mut h.theta));
            self.pool.iter_mut().for_each(|e| clamp_all(&mut e.params));
        }
        self.nu = sampling.nu;
        rng.standard_normals_into(pairs * n, &mut self.deltas, math);
        let (point, deltas, nu, limit) = (&self.heads, &self.deltas, sampling.nu, sampling.max_weight);
        let owner: Vec<usize> = (0..heads)
            .flat_map(|h| std::iter::repeat_n(h, pair_range(pairs, heads, h).1))
            .collect();
        let clamp = move |v: f64| if limit > 0.0 { v.clamp(-limit, limit) } else { v };
        let probes: Vec<[Vec<f64>; 2]> = (0..pairs)
            .into_par_iter()
            .map(|i| {
                let (d, theta) = (&deltas[i * n..(i + 1) * n], &point[owner[i]].theta);
                let side =
                    |sign: f64| -> Vec<f64> { theta.iter().zip(d).map(|(&t, &z)| clamp(t + sign * nu * z)).collect() };
                [side(1.0), side(-1.0)]
            })
            .collect();
        let mut cars: Vec<Vec<f64>> = Vec::with_capacity(sampling.population);
        cars.extend(self.pool[..elites].iter().map(|e| e.params.clone()));
        cars.extend(point.iter().map(|h| h.theta.clone()));
        cars.extend(probes.into_iter().flatten());
        while cars.len() < sampling.population {
            cars.push(point[0].theta.clone());
        }
        self.layout = Some(Layout {
            elites,
            heads,
            pairs,
            size: sampling.population,
        });
        let shape = &self.shape;
        cars.into_par_iter().map(|p| Network::from_vector(shape, p)).collect()
    }

    /// Makes `count` heads: the first ones stay, extra ones start from head 0's
    /// point with a step size of their own.
    fn fit_heads(&mut self, count: usize) {
        self.heads.truncate(count);
        while self.heads.len() < count {
            let theta = self.heads[0].theta.clone();
            self.heads.push(Head::new(theta));
        }
    }

    /// Whether `agents` are the population of the last `sample`.
    fn matches(&self, agents: &[AgentResult]) -> bool {
        let (Some(layout), Some(first)) = (self.layout, agents.first()) else {
            return false;
        };
        layout.size == agents.len()
            && layout.heads == self.heads.len()
            && first.network.shape == self.shape
            && self.heads[0].theta.len() == first.network.params.len()
            && self.deltas.len() == layout.pairs * self.heads[0].theta.len()
    }

    /// The new elite pool and the step of `theta`, from the scored population.
    fn learn(&mut self, agents: &[AgentResult], scores: &[f64], settings: &EvolutionSettings) {
        let Layout {
            elites, heads, pairs, ..
        } = self.layout.expect("learn follows sample");
        let ars = &settings.ars;
        let raw = raw_reward_values(agents, &settings.rewards);

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

        // Probe i is at `elites + heads + 2i` (plus side) and the next slot (minus side).
        let n = self.heads[0].theta.len();
        for h in 0..heads {
            let head = &mut self.heads[h];
            head.adapt(raw[elites + h], ars);
            let (start, count) = pair_range(pairs, heads, h);
            if count == 0 {
                continue;
            }
            let mut ranked: Vec<(usize, f64, f64)> = (start..start + count)
                .map(|i| (i, scores[elites + heads + 2 * i], scores[elites + heads + 1 + 2 * i]))
                .collect();
            ranked.sort_by(|a, b| {
                let (ka, kb) = (a.1.max(a.2), b.1.max(b.2));
                kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal)
            });
            let used = ((ars.top_frac * count as f64).ceil() as usize).clamp(1, count);
            ranked.truncate(used);

            let total = (2 * used) as f64;
            let mean = ranked.iter().map(|r| r.1 + r.2).sum::<f64>() / total;
            let variance = ranked
                .iter()
                .map(|r| (r.1 - mean).powi(2) + (r.2 - mean).powi(2))
                .sum::<f64>()
                / total;
            if variance.sqrt() < 1e-6 {
                continue;
            }
            let scale = head.alpha.expect("adapt sets alpha") / (used as f64 * self.nu);
            for (i, plus, minus) in ranked {
                let weight = scale * (plus - minus);
                for (t, &z) in head.theta.iter_mut().zip(&self.deltas[i * n..(i + 1) * n]) {
                    *t += weight * z;
                }
            }
            if ars.max_weight > 0.0 {
                let limit = ars.max_weight;
                head.theta.iter_mut().for_each(|t| *t = t.clamp(-limit, limit));
            }
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
        self.fit_heads(sampling.heads);
        let before = trace.then(|| ArsRecord {
            shape: self.shape.clone(),
            heads: self.heads.clone(),
            pool: self.pool.clone(),
            sampling: sampling.clone(),
            rng: rng.clone(),
            math,
        });
        let networks = self.sample(&sampling, rng, math);
        Produced {
            generation: Generation {
                networks,
                preserved_count: self.layout.map_or(0, |l| l.elites + l.heads),
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
            heads: vec![Head::new(seed.params.clone())],
            ..Ars::default()
        };
        debug_assert_eq!(self.heads[0].theta.len(), parameter_count(&self.shape));
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
                heads: vec![Head::new(network.params.clone())],
                ..Ars::default()
            };
        }
        self.produce(scores, settings, rng, math, trace)
    }
}

#[cfg(test)]
mod tests;
