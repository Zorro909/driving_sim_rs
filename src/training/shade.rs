//! SHADE: differential evolution with a success history of its parameters
//! (Tanabe and Fukunaga 2013), keeping the best car found so far.
//!
//! The search keeps `NP = population - 1` parents. Each population is
//!
//! ```text
//! [best car] [trial 0] [trial 1] .. [trial NP-1]
//! ```
//!
//! The best car runs again unchanged. Trial `i` is made from parent `i` by
//! `current-to-pbest/1`,
//! `v = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)`, where `x_pbest` is one
//! of the best `p` fraction of the parents, `x_r1` another parent and `x_r2`
//! a parent or an archived one that was replaced earlier, followed by binomial
//! crossover with rate `CR`. `F` and `CR` are drawn around a random cell of
//! the success history: `F` from a Cauchy and `CR` from a normal distribution.
//!
//! After the generation runs, trial `i` replaces parent `i` if it scores
//! higher; the replaced parent goes to the archive. The `F` and `CR` of the
//! replacing trials, weighted by how much they gained, give their Lehmer means,
//! which overwrite the next cell of the history. The best car changes only to
//! a parent that beats it by `promote_margin_rel`.
//!
//! Parents are not run again, so their scores come from the generation that
//! made them and are compared with the scores of later ones. The default
//! scores (`raw_reward_values`) are the rewards themselves, not ranks, for that
//! reason; scores handed in from outside are used as given.

use super::optimizer::{Optimizer, Produced, Record};
use super::TrainingRandom;
use crate::math::profile::MathProfile;
use crate::nn::network::Network;
use crate::training::evolution::{raw_reward_values, AgentResult, EvolutionSettings, Generation, ShadeSettings};

/// The smallest number of parents: `current-to-pbest/1` needs the car itself,
/// a best one and two others.
const MIN_PARENTS: usize = 4;
/// Normal variates drawn per refill.
const NORMAL_CHUNK: usize = 256;

/// Rejects settings that `Shade` cannot run with.
pub(crate) fn validate(s: &ShadeSettings, population: usize) -> Result<(), String> {
    if population < MIN_PARENTS + 1 {
        return Err(format!(
            "shade needs population >= {} (the best car and {MIN_PARENTS} parents), got {population}",
            MIN_PARENTS + 1
        ));
    }
    if s.memory_h == 0 {
        return Err("shade memory_h must be positive".into());
    }
    if !(s.p > 0.0 && s.p <= 1.0) {
        return Err("shade p must be in (0, 1]".into());
    }
    if !s.initial_sigma.is_finite() || s.initial_sigma <= 0.0 {
        return Err("shade initial_sigma must be positive".into());
    }
    if !s.promote_margin_rel.is_finite() || s.promote_margin_rel < 0.0 {
        return Err("shade promote_margin_rel must not be negative".into());
    }
    if !(0.0..1.0).contains(&s.rejump_frac) {
        return Err("shade rejump_frac must be in [0, 1)".into());
    }
    if !s.max_weight.is_finite() || s.max_weight < 0.0 {
        return Err("shade max_weight must be 0 (off) or positive".into());
    }
    Ok(())
}

/// The settings that shape a population, as checkpoints store them.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Sampling {
    pub(crate) population: usize,
    pub(crate) p: f64,
    pub(crate) max_weight: f64,
}

impl Sampling {
    fn of(s: &EvolutionSettings) -> Sampling {
        Sampling {
            population: s.population,
            p: s.shade.p,
            max_weight: s.shade.max_weight,
        }
    }
}

/// The best car found so far.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Best {
    pub(crate) params: Vec<f64>,
    /// `None` until it has been scored.
    pub(crate) score: Option<f64>,
    /// Generations since it last changed.
    pub(crate) stalled: usize,
}

/// Everything the search carries from one generation to the next.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Search {
    pub(crate) parents: Vec<Vec<f64>>,
    /// The score of each parent; `None` until it has been scored.
    pub(crate) scores: Vec<Option<f64>>,
    /// Parents that were replaced, oldest first.
    pub(crate) archive: Vec<Vec<f64>>,
    /// The success history: a location of `F` and of `CR` per cell.
    pub(crate) f_history: Vec<f64>,
    pub(crate) cr_history: Vec<f64>,
    /// The cell the next success overwrites.
    pub(crate) next_cell: usize,
    pub(crate) best: Option<Best>,
}

/// The trial made from one parent, and the `F` and `CR` that made it.
#[derive(Clone, Debug)]
struct Trial {
    params: Vec<f64>,
    f: f64,
    cr: f64,
}

#[derive(Clone, Default)]
pub(crate) struct Shade {
    shape: Vec<usize>,
    search: Search,
    /// The trials of the last population, in slot order after the best car;
    /// empty until it is sampled.
    trials: Vec<Trial>,
}

/// A `Shade` before it sampled a population: sampling again from `rng` repeats it.
#[derive(Clone)]
pub(crate) struct ShadeRecord {
    pub(crate) shape: Vec<usize>,
    pub(crate) search: Search,
    pub(crate) sampling: Sampling,
    /// The generator before the trials were drawn.
    pub(crate) rng: TrainingRandom,
    /// The math the game's streams draw with.
    pub(crate) math: MathProfile,
}

impl ShadeRecord {
    /// Whether `rebuild` can sample this record; a checkpoint may be corrupt.
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.shape.len() < 2 || self.shape.contains(&0) {
            return Err("the checkpoint has an invalid network shape".into());
        }
        let size = self.shape.windows(2).try_fold(0usize, |sum, w| {
            w[0].checked_add(1)?.checked_mul(w[1])?.checked_add(sum)
        });
        let s = &self.search;
        let Some(best) = &s.best else {
            return Err("the checkpoint has no best car".into());
        };
        let sized = |v: &Vec<f64>| Some(v.len()) == size;
        if !sized(&best.params) || !s.parents.iter().all(sized) || !s.archive.iter().all(sized) {
            return Err("the checkpoint has the wrong number of parameters".into());
        }
        let finite = |v: &Vec<f64>| v.iter().all(|p| p.is_finite());
        if !finite(&best.params)
            || !s.parents.iter().all(finite)
            || !s.archive.iter().all(finite)
            || best.score.is_some_and(|x| !x.is_finite())
            || s.scores.iter().flatten().any(|x| !x.is_finite())
        {
            return Err("non-finite checkpoint parameter or score".into());
        }
        if s.parents.len() < MIN_PARENTS
            || s.scores.len() != s.parents.len()
            || self.sampling.population != s.parents.len() + 1
        {
            return Err("the checkpoint's population does not match its parents".into());
        }
        let cells = s.f_history.len();
        if cells == 0
            || s.cr_history.len() != cells
            || s.next_cell >= cells
            || s.f_history.iter().any(|f| !(*f > 0.0 && *f <= 1.0))
            || s.cr_history.iter().any(|cr| !(0.0..=1.0).contains(cr))
        {
            return Err("the checkpoint's success history is out of range".into());
        }
        let m = &self.sampling;
        if !(m.p > 0.0 && m.p <= 1.0) || !m.max_weight.is_finite() || m.max_weight < 0.0 {
            return Err("the checkpoint's shade p or weight limit is out of range".into());
        }
        Ok(())
    }

    /// The sampled population, the generator after it, and the search state.
    /// Call `validate` first.
    pub(crate) fn rebuild(&self) -> (Vec<Network>, TrainingRandom, Shade) {
        let mut shade = Shade {
            shape: self.shape.clone(),
            search: self.search.clone(),
            ..Shade::default()
        };
        let mut rng = self.rng.clone();
        let networks = shade.sample(&self.sampling, &mut rng, self.math);
        (networks, rng, shade)
    }
}

/// Standard normal variates, drawn from the generator in chunks.
struct Normals {
    chunk: Vec<f64>,
    at: usize,
    math: MathProfile,
}

impl Normals {
    fn new(math: MathProfile) -> Normals {
        Normals {
            chunk: Vec::new(),
            at: 0,
            math,
        }
    }

    fn next(&mut self, rng: &mut TrainingRandom) -> f64 {
        if self.at == self.chunk.len() {
            rng.standard_normals_into(NORMAL_CHUNK, &mut self.chunk, self.math);
            self.at = 0;
        }
        self.at += 1;
        self.chunk[self.at - 1]
    }

    /// A standard Cauchy variate: the ratio of two standard normal ones.
    fn cauchy(&mut self, rng: &mut TrainingRandom) -> f64 {
        let (a, b) = (self.next(rng), self.next(rng));
        a / b
    }
}

/// `F` around `location`: redrawn while outside `(0, 1]`.
fn draw_f(location: f64, normals: &mut Normals, rng: &mut TrainingRandom) -> f64 {
    for _ in 0..100 {
        let f = location + 0.1 * normals.cauchy(rng);
        if f > 0.0 && f <= 1.0 {
            return f;
        }
    }
    location.clamp(1e-6, 1.0)
}

fn by_score(a: &Option<f64>, b: &Option<f64>) -> std::cmp::Ordering {
    a.unwrap_or(f64::NEG_INFINITY)
        .total_cmp(&b.unwrap_or(f64::NEG_INFINITY))
}

/// `seed` plus noise of `sigma` per parameter.
fn jittered(seed: &[f64], sigma: f64, normals: &mut Normals, rng: &mut TrainingRandom) -> Vec<f64> {
    seed.iter().map(|&p| p + sigma * normals.next(rng)).collect()
}

fn clamp_all(v: &mut [f64], limit: f64) {
    if limit > 0.0 {
        v.iter_mut().for_each(|x| *x = x.clamp(-limit, limit));
    }
}

impl Shade {
    /// A search around `seed`: parents drawn from a normal distribution about it,
    /// none scored yet, and `seed` as the best car.
    fn seeded(
        seed: &Network,
        population: usize,
        settings: &ShadeSettings,
        rng: &mut TrainingRandom,
        math: MathProfile,
    ) -> Shade {
        let parents = population - 1;
        let mut normals = Normals::new(math);
        Shade {
            shape: seed.shape.clone(),
            search: Search {
                parents: (0..parents)
                    .map(|_| jittered(&seed.params, settings.initial_sigma, &mut normals, rng))
                    .collect(),
                scores: vec![None; parents],
                archive: Vec::new(),
                f_history: vec![0.5; settings.memory_h],
                cr_history: vec![0.5; settings.memory_h],
                next_cell: 0,
                best: Some(Best {
                    params: seed.params.clone(),
                    score: None,
                    stalled: 0,
                }),
            },
            trials: Vec::new(),
        }
    }

    /// Draws the trials and builds the population around the parents.
    fn sample(&mut self, sampling: &Sampling, rng: &mut TrainingRandom, math: MathProfile) -> Vec<Network> {
        let parents = self.search.parents.len();
        assert!(
            sampling.population == parents + 1 && parents >= MIN_PARENTS,
            "shade needs population >= {}",
            MIN_PARENTS + 1
        );
        let limit = sampling.max_weight;
        let search = &mut self.search;
        search.parents.iter_mut().for_each(|p| clamp_all(p, limit));
        search.archive.iter_mut().for_each(|p| clamp_all(p, limit));
        let best = search.best.as_mut().expect("sampling follows seeding");
        clamp_all(&mut best.params, limit);

        let n = best.params.len();
        let mut order: Vec<usize> = (0..parents).collect();
        order.sort_by(|&a, &b| by_score(&search.scores[a], &search.scores[b]));
        let aimed = ((parents as f64 * sampling.p).ceil() as usize).clamp(1, parents);
        let top = &order[parents - aimed..];

        let cells = search.f_history.len();
        let mut normals = Normals::new(math);
        self.trials = (0..parents)
            .map(|i| {
                let cell = rng.below(cells);
                let f = draw_f(search.f_history[cell], &mut normals, rng);
                let cr = (search.cr_history[cell] + 0.1 * normals.next(rng)).clamp(0.0, 1.0);
                let pbest = &search.parents[top[rng.below(aimed)]];
                let r1 = loop {
                    let r = rng.below(parents);
                    if r != i {
                        break &search.parents[r];
                    }
                };
                let r2 = rng.below(parents + search.archive.len());
                let r2 = search.parents.get(r2).unwrap_or_else(|| &search.archive[r2 - parents]);
                let current = &search.parents[i];
                let forced = rng.below(n);
                let params = (0..n)
                    .map(|d| {
                        if rng.unit() < cr || d == forced {
                            let v = current[d] + f * (pbest[d] - current[d]) + f * (r1[d] - r2[d]);
                            if limit > 0.0 {
                                v.clamp(-limit, limit)
                            } else {
                                v
                            }
                        } else {
                            current[d]
                        }
                    })
                    .collect();
                Trial { params, f, cr }
            })
            .collect();
        let shape = &self.shape;
        std::iter::once(&best.params)
            .chain(self.trials.iter().map(|t| &t.params))
            .map(|p| Network::from_vector(shape, p.clone()))
            .collect()
    }

    /// Whether `agents` are the population of the last `sample`.
    fn matches(&self, agents: &[AgentResult]) -> bool {
        let Some(first) = agents.first() else {
            return false;
        };
        !self.trials.is_empty()
            && agents.len() == self.trials.len() + 1
            && first.network.shape == self.shape
            && self
                .search
                .best
                .as_ref()
                .is_some_and(|b| b.params.len() == first.network.params.len())
    }

    /// The parents the trials replace, the history, the best car and the archive,
    /// from the scored population.
    fn learn(&mut self, scores: &[f64], settings: &EvolutionSettings, rng: &mut TrainingRandom, math: MathProfile) {
        let shade = &settings.shade;
        let parents = self.search.parents.len();
        let search = &mut self.search;
        let best = search.best.as_mut().expect("learning follows sampling");
        if best.score.is_none() && scores[0].is_finite() {
            best.score = Some(scores[0]);
        }

        let mut gains = Vec::new();
        for (i, trial) in std::mem::take(&mut self.trials).into_iter().enumerate() {
            let score = scores[i + 1];
            let gain = match search.scores[i] {
                _ if !score.is_finite() => continue,
                Some(parent) if score <= parent => continue,
                Some(parent) => score - parent,
                None => 1.0,
            };
            search
                .archive
                .push(std::mem::replace(&mut search.parents[i], trial.params));
            search.scores[i] = Some(score);
            gains.push((gain, trial.f, trial.cr));
        }
        let keep = if shade.archive_size == 0 {
            parents
        } else {
            shade.archive_size
        };
        while search.archive.len() > keep {
            search.archive.remove(rng.below(search.archive.len()));
        }

        if search.f_history.len() != shade.memory_h {
            // The history changed size: every cell starts from the old mean.
            let mean = |h: &[f64]| h.iter().sum::<f64>() / h.len() as f64;
            let (f, cr) = (mean(&search.f_history), mean(&search.cr_history));
            search.f_history = vec![f; shade.memory_h];
            search.cr_history = vec![cr; shade.memory_h];
            search.next_cell = 0;
        }
        let lehmer = |value: fn(&(f64, f64, f64)) -> f64| {
            let numerator: f64 = gains.iter().map(|g| g.0 * value(g) * value(g)).sum();
            let denominator: f64 = gains.iter().map(|g| g.0 * value(g)).sum();
            (denominator > 1e-12).then(|| (numerator / denominator).min(1.0))
        };
        if !gains.is_empty() {
            if let Some(f) = lehmer(|g| g.1) {
                search.f_history[search.next_cell] = f;
            }
            if let Some(cr) = lehmer(|g| g.2) {
                search.cr_history[search.next_cell] = cr;
            }
            search.next_cell = (search.next_cell + 1) % shade.memory_h;
        }

        let margin = |score: f64| shade.promote_margin_rel * score.abs().max(1e-9);
        let mut promoted = false;
        for i in 0..parents {
            let Some(score) = search.scores[i] else { continue };
            if best.score.is_none_or(|b| score > b + margin(b)) {
                best.params = search.parents[i].clone();
                best.score = Some(score);
                best.stalled = 0;
                promoted = true;
            }
        }
        if !promoted {
            best.stalled += 1;
        }

        let jumping = (parents as f64 * shade.rejump_frac).floor() as usize;
        if shade.rejump_gens > 0 && !promoted && best.stalled >= shade.rejump_gens && jumping > 0 && jumping < parents {
            let mut order: Vec<usize> = (0..parents).collect();
            order.sort_by(|&a, &b| by_score(&search.scores[a], &search.scores[b]));
            let mut normals = Normals::new(math);
            for &i in &order[..jumping] {
                search.parents[i] = jittered(&best.params, shade.initial_sigma, &mut normals, rng);
                search.scores[i] = None;
            }
            best.stalled = 0;
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
        let before = trace.then(|| ShadeRecord {
            shape: self.shape.clone(),
            search: self.search.clone(),
            sampling: sampling.clone(),
            rng: rng.clone(),
            math,
        });
        let networks = self.sample(&sampling, rng, math);
        Produced {
            generation: Generation {
                networks,
                preserved_count: 1,
                rewards: scores,
            },
            record: before.map(Record::Shade),
        }
    }
}

impl Optimizer for Shade {
    fn name(&self) -> &'static str {
        "shade"
    }

    fn boxed_clone(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }

    fn invalidate(&mut self) {
        // `matches` fails from here on, so the next turnover searches on from the best car.
        self.trials.clear();
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
        let scores = raw_reward_values(&seed_result, &settings.rewards);
        *self = Shade::seeded(seed, settings.population, &settings.shade, rng, math);
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
        let scores = scores.unwrap_or_else(|| raw_reward_values(agents, &settings.rewards));
        if self.matches(agents) && agents.len() == settings.population {
            self.learn(&scores, settings, rng, math);
        } else {
            // A new run, a restored population or a changed population size:
            // the trials no longer belong to these cars. Search on from the
            // best one instead.
            let best = (1..scores.len()).fold(0, |best, i| if scores[i] > scores[best] { i } else { best });
            *self = Shade::seeded(agents[best].network, settings.population, &settings.shade, rng, math);
        }
        self.produce(scores, settings, rng, math, trace)
    }
}

#[cfg(test)]
mod tests;
