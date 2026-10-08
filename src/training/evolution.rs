//! Rewards, selection, crossover, and
//! mutation with CPython-compatible random draws.

use crate::math::profile::MathProfile;
use crate::nn::network::Network;
use crate::training::pyrandom::PyRandom;
use rayon::prelude::*;
use serde_json::{json, Value};

pub(crate) trait DecisionSource {
    fn random(&mut self) -> f64;
    fn randrange(&mut self, bound: usize) -> usize;
    fn game_arithmetic(&self) -> bool {
        false
    }
}
impl DecisionSource for PyRandom {
    fn random(&mut self) -> f64 {
        PyRandom::random(self)
    }
    fn randrange(&mut self, bound: usize) -> usize {
        PyRandom::randrange(self, bound)
    }
}
impl DecisionSource for crate::training::game_random::GameRandom {
    fn random(&mut self) -> f64 {
        crate::training::game_random::GameRandom::random(self)
    }
    fn randrange(&mut self, bound: usize) -> usize {
        crate::training::game_random::GameRandom::randrange(self, bound)
    }
    fn game_arithmetic(&self) -> bool {
        true
    }
}

pub const METRIC_NAMES: [&str; 14] = [
    "total_score",
    "total_speed",
    "total_abs_steering",
    "total_actual_distance",
    "total_drifting_amount",
    "total_momentum_change_frequency",
    "total_throttle_change_frequency",
    "total_braking_frequency",
    "total_distance_from_wall",
    "total_distance_from_center",
    "network_novelty",
    "collision_count",
    "best_lap_performance",
    "total_racing_line_efficiency",
];

pub(crate) type Metrics = [Option<f64>; 14];

const AVERAGE_METRICS: [&str; 10] = [
    "total_score",
    "total_speed",
    "total_abs_steering",
    "total_drifting_amount",
    "total_actual_distance",
    "total_distance_from_wall",
    "total_distance_from_center",
    "total_momentum_change_frequency",
    "total_throttle_change_frequency",
    "total_braking_frequency",
];
const AVERAGE_ONLY_METRICS: [&str; 6] = [
    "total_abs_steering",
    "total_distance_from_wall",
    "total_distance_from_center",
    "total_momentum_change_frequency",
    "total_throttle_change_frequency",
    "total_braking_frequency",
];

pub(crate) fn metric_index(name: &str) -> Option<usize> {
    METRIC_NAMES.iter().position(|&m| m == name)
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RewardSpec {
    pub metric: String,
    pub weight: i64,
    #[serde(rename = "type")]
    pub kind: String,
}

#[derive(Clone, Copy)]
pub struct AgentResult<'a> {
    pub network: &'a Network,
    pub metrics: Metrics,
    pub update_count: u64,
}

impl AgentResult<'_> {
    pub(crate) fn value(&self, spec: &RewardSpec) -> Option<f64> {
        let value = self.metrics[metric_index(&spec.metric)?]?;
        let metric = spec.metric.as_str();
        if (spec.kind == "average" || AVERAGE_ONLY_METRICS.contains(&metric)) && AVERAGE_METRICS.contains(&metric) {
            return Some(if self.update_count != 0 {
                value / (0.1 * self.update_count as f64)
            } else {
                0.0
            });
        }
        Some(value)
    }
}

/// Parameters of the `"ars"` algorithm (see `training::ars`).
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ArsSettings {
    /// Standard deviation of the probe noise: probes are `theta +- nu * delta`.
    pub nu: f64,
    /// Step size of the search point.
    pub alpha: f64,
    /// Fraction of the antithetic pairs, best first, that steer the step.
    pub top_frac: f64,
    /// Best distinct cars carried over verbatim and re-evaluated each generation.
    pub elite_count: usize,
    /// Parameters are clamped to `[-max_weight, max_weight]`; 0 turns it off.
    pub max_weight: f64,
}

impl Default for ArsSettings {
    fn default() -> Self {
        ArsSettings {
            nu: 0.05,
            alpha: 0.05,
            top_frac: 1.0,
            elite_count: 3,
            max_weight: 0.0,
        }
    }
}

/// Parameters of the `"shade"` algorithm (see `training::shade`).
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ShadeSettings {
    /// Cells of the success history that `F` and `CR` are drawn around.
    pub memory_h: usize,
    /// Fraction of the parents, best first, that `current-to-pbest` aims at.
    pub p: f64,
    /// Replaced parents kept for the difference vectors; 0 keeps as many as there are parents.
    pub archive_size: usize,
    /// Standard deviation of the noise around the seed in the first population, and around the best car on a re-jump.
    pub initial_sigma: f64,
    /// A parent must beat the best car by this fraction of its score to replace it.
    pub promote_margin_rel: f64,
    /// Generations without a new best car before the worst parents jump back to it; 0 turns it off.
    pub rejump_gens: usize,
    /// Fraction of the parents, worst first, that a re-jump replaces.
    pub rejump_frac: f64,
    /// Parameters are clamped to `[-max_weight, max_weight]`; 0 turns it off.
    pub max_weight: f64,
}

impl Default for ShadeSettings {
    fn default() -> Self {
        ShadeSettings {
            memory_h: 6,
            p: 0.1,
            archive_size: 0,
            initial_sigma: 0.05,
            promote_margin_rel: 0.0,
            rejump_gens: 0,
            rejump_frac: 0.25,
            max_weight: 0.0,
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EvolutionSettings {
    /// `"ga"` (selection, crossover and mutation), `"ars"` or `"shade"`.
    pub algorithm: String,
    pub ars: ArsSettings,
    pub shade: ShadeSettings,
    pub population: usize,
    pub selection_algorithm: String,
    pub selection_size: usize,
    pub crossover: String,
    pub mutation_rate: f64,
    pub adaptive_mutation: bool,
    pub weight_decay: f64,
    pub preserve_parents: String,
    pub preserve_parents_size: usize,
    pub rewards: Vec<RewardSpec>,
}

impl Default for EvolutionSettings {
    fn default() -> Self {
        EvolutionSettings {
            algorithm: "ga".into(),
            ars: ArsSettings::default(),
            shade: ShadeSettings::default(),
            population: 300,
            selection_algorithm: "best".into(),
            selection_size: 3,
            crossover: "none".into(),
            mutation_rate: 0.20000000298023224,
            adaptive_mutation: false,
            weight_decay: 0.0005000000237487257,
            preserve_parents: "on_selection_size".into(),
            preserve_parents_size: 3,
            rewards: vec![RewardSpec {
                metric: "total_score".into(),
                weight: 100,
                kind: "default".into(),
            }],
        }
    }
}

impl EvolutionSettings {
    /// `EvolutionSettings.from_mcp`.
    /// Panics on malformed settings; preserves trusted CLI/export coercions.
    pub fn from_mcp(data: &Value) -> EvolutionSettings {
        Self::try_from_mcp(data).unwrap_or_else(|e| panic!("{e}"))
    }

    /// `from_mcp`, reporting malformed settings instead of panicking.
    pub fn try_from_mcp(data: &Value) -> Result<EvolutionSettings, String> {
        let data = data.get("settings").unwrap_or(data);
        let mut s = EvolutionSettings::default();
        let bad = |key: &str| format!("invalid evolution settings: bad {key}");
        let int = |key: &str| -> Result<Option<usize>, String> {
            data.get(key)
                .map(|v| v.as_f64().map(|x| x as usize).ok_or_else(|| bad(key)))
                .transpose()
        };
        let text = |key: &str| -> Result<Option<String>, String> {
            data.get(key)
                .map(|v| v.as_str().map(String::from).ok_or_else(|| bad(key)))
                .transpose()
        };
        if let Some(v) = text("algorithm")? {
            s.algorithm = v;
        }
        if let Some(v) = data.get("ars") {
            s.ars = serde_json::from_value(v.clone()).map_err(|e| format!("invalid evolution settings: ars: {e}"))?;
        }
        if let Some(v) = data.get("shade") {
            s.shade =
                serde_json::from_value(v.clone()).map_err(|e| format!("invalid evolution settings: shade: {e}"))?;
        }
        if let Some(v) = int("population")? {
            s.population = v;
        }
        if let Some(v) = text("selection_algorithm")? {
            s.selection_algorithm = v;
        }
        if let Some(v) = int("selection_size")? {
            s.selection_size = v;
        }
        if let Some(v) = text("crossover")? {
            s.crossover = v;
        }
        if let Some(v) = data.get("mutation_rate") {
            s.mutation_rate = v.as_f64().ok_or_else(|| bad("mutation_rate"))?;
        }
        if let Some(v) = data.get("adaptive_mutation") {
            s.adaptive_mutation = v.as_bool().ok_or_else(|| bad("adaptive_mutation"))?;
        }
        if let Some(v) = data.get("weight_decay") {
            s.weight_decay = v.as_f64().ok_or_else(|| bad("weight_decay"))?;
        }
        if let Some(v) = text("preserve_parents")? {
            s.preserve_parents = v;
        }
        if let Some(v) = int("preserve_parents_size")? {
            s.preserve_parents_size = v;
        }
        if let Some(rewards) = data.get("rewards") {
            s.rewards = rewards
                .as_array()
                .ok_or_else(|| bad("rewards"))?
                .iter()
                .map(|item| {
                    Ok(RewardSpec {
                        metric: item["metric"].as_str().ok_or_else(|| bad("reward metric"))?.into(),
                        weight: item["weight"].as_i64().ok_or_else(|| bad("reward weight"))?,
                        kind: item.get("type").and_then(Value::as_str).unwrap_or("default").into(),
                    })
                })
                .collect::<Result<_, String>>()?;
        }
        Ok(s)
    }

    /// Validate creation settings after applying the embedding host's population override.
    pub(crate) fn try_from_mcp_with_population(
        data: &Value,
        population: Option<usize>,
    ) -> Result<EvolutionSettings, String> {
        let fields = data.get("settings").unwrap_or(data);
        for key in ["population", "selection_size", "preserve_parents_size"] {
            if let Some(value) = fields.get(key) {
                value
                    .as_f64()
                    .filter(|n| n.is_finite() && *n >= 0.0 && n.fract() == 0.0 && *n < usize::MAX as f64)
                    .ok_or_else(|| format!("invalid evolution settings: bad {key}"))?;
            }
        }
        let mut settings = Self::try_from_mcp(data)?;
        if let Some(population) = population {
            settings.population = population;
        }
        settings.validate()?;
        Ok(settings)
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        self.validate_algorithm()?;
        if self.population == 0 || self.selection_size == 0 {
            return Err("population and selection_size must be positive".into());
        }
        if !["best", "tournament", "roulette"].contains(&self.selection_algorithm.as_str())
            || !["none", "single_point", "uniform"].contains(&self.crossover.as_str())
            || !["off", "on_selection_size", "on_custom"].contains(&self.preserve_parents.as_str())
        {
            return Err("unknown selection, crossover or preservation mode".into());
        }
        if !self.mutation_rate.is_finite()
            || !(0.0..=10.0).contains(&self.mutation_rate)
            || !self.weight_decay.is_finite()
            || !(0.0..=1.0).contains(&self.weight_decay)
        {
            return Err("mutation_rate must be in 0..10 and weight_decay in 0..1".into());
        }
        if self.rewards.is_empty()
            || self.rewards.iter().any(|r| {
                crate::training::evolution::metric_index(&r.metric).is_none()
                    || !["default", "average"].contains(&r.kind.as_str())
            })
        {
            return Err("invalid reward metric or type".into());
        }
        self.rewards.iter().try_fold(0i64, |total, r| {
            r.weight
                .checked_abs()
                .and_then(|w| total.checked_add(w))
                .ok_or("reward weights overflow")
        })?;
        Ok(())
    }

    /// `dataclasses.asdict(settings)`.
    pub fn to_json(&self) -> Value {
        let mut value = self.base_json();
        if self.algorithm != "ga" {
            value["algorithm"] = json!(self.algorithm);
        }
        if self.algorithm == "shade" {
            let shade = &self.shade;
            value["shade"] = json!({
                "memory_h": shade.memory_h,
                "p": shade.p,
                "archive_size": shade.archive_size,
                "initial_sigma": shade.initial_sigma,
                "promote_margin_rel": shade.promote_margin_rel,
                "rejump_gens": shade.rejump_gens,
                "rejump_frac": shade.rejump_frac,
                "max_weight": shade.max_weight,
            });
        } else if self.algorithm != "ga" {
            let ars = &self.ars;
            value["ars"] = json!({
                "nu": ars.nu,
                "alpha": ars.alpha,
                "top_frac": ars.top_frac,
                "elite_count": ars.elite_count,
                "max_weight": ars.max_weight,
            });
        }
        value
    }

    /// Checks the fields of the chosen algorithm that the algorithms cannot
    /// repair themselves. Anything is accepted for the GA, as before.
    pub fn validate_algorithm(&self) -> Result<(), String> {
        match self.algorithm.as_str() {
            "ga" => Ok(()),
            "ars" => crate::training::ars::validate(&self.ars, self.population),
            "shade" => crate::training::shade::validate(&self.shade, self.population),
            other => Err(format!("unknown algorithm {other:?} (ga, ars or shade)")),
        }
    }

    fn base_json(&self) -> Value {
        json!({
            "population": self.population,
            "selection_algorithm": self.selection_algorithm,
            "selection_size": self.selection_size,
            "crossover": self.crossover,
            "mutation_rate": self.mutation_rate,
            "adaptive_mutation": self.adaptive_mutation,
            "weight_decay": self.weight_decay,
            "preserve_parents": self.preserve_parents,
            "preserve_parents_size": self.preserve_parents_size,
            "rewards": self.rewards.iter().map(|r| json!({"metric": r.metric, "weight": r.weight, "type": r.kind})).collect::<Vec<_>>(),
        })
    }
}

pub struct Generation {
    pub networks: Vec<Network>,
    pub preserved_count: usize,
    pub rewards: Vec<f64>,
}

fn rank_normalize(values: &[Option<f64>]) -> Vec<f64> {
    let mut distinct: Vec<f64> = values.iter().flatten().copied().collect();
    distinct.sort_by(|a, b| a.partial_cmp(b).expect("reward values must not be NaN"));
    distinct.dedup_by(|a, b| a == b);
    if distinct.is_empty() {
        return vec![0.0; values.len()];
    }
    if distinct.len() == 1 {
        return values.iter().map(|v| if v.is_some() { 0.5 } else { 0.0 }).collect();
    }
    let top = (distinct.len() - 1) as f64;
    values
        .iter()
        .map(|v| match v {
            Some(v) => {
                let i = distinct
                    .binary_search_by(|d| d.partial_cmp(v).unwrap())
                    .expect("value in distinct set");
                i as f64 / top
            }
            None => 0.0,
        })
        .collect()
}

/// Relative reward used for selection, with signed weights and rank ties.
pub fn reward_values(agents: &[AgentResult], specs: &[RewardSpec]) -> Vec<f64> {
    let mut result = vec![0.0; agents.len()];
    let total_weight: i64 = specs.iter().map(|s| s.weight.abs()).sum();
    if total_weight == 0 {
        return result;
    }
    for spec in specs {
        let values: Vec<Option<f64>> = agents.iter().map(|a| a.value(spec)).collect();
        let ranks = rank_normalize(&values);
        let ratio = spec.weight as f64 / total_weight as f64;
        for (index, rank) in ranks.into_iter().enumerate() {
            result[index] += rank * ratio;
        }
    }
    result
}

/// The weighted sum of the cars' reward metrics themselves, not their ranks:
/// scores that stay comparable from one generation to the next. A car without
/// a metric adds 0, as it ranks lowest in `reward_values`.
pub fn raw_reward_values(agents: &[AgentResult], specs: &[RewardSpec]) -> Vec<f64> {
    let total_weight: i64 = specs.iter().map(|s| s.weight.abs()).sum();
    if total_weight == 0 {
        return vec![0.0; agents.len()];
    }
    agents
        .iter()
        .map(|agent| {
            specs.iter().fold(0.0, |sum, spec| {
                sum + agent.value(spec).unwrap_or(0.0) * (spec.weight as f64 / total_weight as f64)
            })
        })
        .collect()
}

/// The adaptive mutation scale; its log is the C runtime's (.NET Math.Log).
pub fn mutation_factor(network: &Network, algorithm: &str, math: MathProfile) -> f64 {
    let total = network.params.len() as f64;
    let layers = network.shape.len() as f64;
    let log = math.log(total + 1.0);
    let factor = match algorithm {
        "xavier" => 0.516777 / (0.262982 + 0.050982 * log - 0.000139 * layers - 0.000278 * layers * layers),
        "gaussian" => 1.646116 / (0.088941 + 0.350773 * log - 0.03423 * layers + 0.001063 * layers * layers),
        "uniform" => 0.272879 / (-0.074226 + 0.07021 * log - 0.000777 * layers - 0.000321 * layers * layers),
        _ => return 1.0,
    };
    if factor > 0.0 {
        let low = if factor > 0.1 { factor } else { 0.1 };
        if low < 10.0 {
            low
        } else {
            10.0
        }
    } else {
        1.0
    }
}

#[cfg(test)]
pub(crate) fn mutate_xavier(network: &Network, rate: f64, rng: &mut PyRandom, normalize: bool) -> Network {
    if rate <= 0.0 {
        return network.clone();
    }
    let normals = rng.standard_normals(network.params.len());
    mutate_xavier_with(network.clone(), rate, normalize, &normals)
}

/// `mutate_xavier` with its `gauss` draws supplied as standard normals `z`
/// (`gauss(0, sigma) == 0.0 + z * sigma`), in parameter order.
// Index order and parameter updates mirror the reference reproduction loop.
#[allow(clippy::needless_range_loop)]
fn mutate_xavier_with(mut network: Network, rate: f64, normalize: bool, normals: &[f64]) -> Network {
    let scale = rate
        * if normalize {
            mutation_factor(&network, "xavier", MathProfile::Proton)
        } else {
            1.0
        };
    let mut position = 0;
    let shape = network.shape.clone();
    for w in shape.windows(2) {
        let (inputs, outputs) = (w[0], w[1]);
        let deviation = (2.0 / (inputs + outputs) as f64).sqrt() * scale;
        for index in position..position + inputs * outputs {
            network.params[index] += 0.0 + normals[index] * deviation;
        }
        position += inputs * outputs;
        let bias_deviation = 0.1 * scale;
        for index in position..position + outputs {
            network.params[index] += 0.0 + normals[index] * bias_deviation;
        }
        position += outputs;
    }
    network
}

/// `mutate_xavier_with` followed by `weight_decay` for a child that is a
/// plain copy of `source` (crossover "none"): the copy, the mutation and the
/// decay are one pass writing a fresh vector, instead of a clone and two
/// read-modify-write passes. Each parameter sees the same operations in the
/// same order: `p + (0.0 + z * deviation)`, then `* (1.0 - decay)` if decay
/// is positive.
fn mutated_copy(source: &Network, rate: f64, normalize: bool, normals: &[f64], decay: f64) -> Network {
    let scale = rate
        * if normalize {
            mutation_factor(source, "xavier", MathProfile::Proton)
        } else {
            1.0
        };
    let keep = 1.0 - decay;
    let finish = move |v: f64| if decay > 0.0 { v * keep } else { v };
    let mut params = Vec::with_capacity(source.params.len());
    let mut position = 0;
    for w in source.shape.windows(2) {
        let (inputs, outputs) = (w[0], w[1]);
        let deviation = (2.0 / (inputs + outputs) as f64).sqrt() * scale;
        let weights = position..position + inputs * outputs;
        params.extend(
            source.params[weights.clone()]
                .iter()
                .zip(&normals[weights])
                .map(|(&p, &z)| finish(p + (0.0 + z * deviation))),
        );
        position += inputs * outputs;
        let bias_deviation = 0.1 * scale;
        let biases = position..position + outputs;
        params.extend(
            source.params[biases.clone()]
                .iter()
                .zip(&normals[biases])
                .map(|(&p, &z)| finish(p + (0.0 + z * bias_deviation))),
        );
        position += outputs;
    }
    Network {
        shape: source.shape.clone(),
        params,
    }
}

pub(crate) fn weight_decay(mut network: Network, rate: f64) -> Network {
    if rate > 0.0 {
        let keep = 1.0 - rate;
        network.params.iter_mut().for_each(|v| *v *= keep);
    }
    network
}

/// Indices sorted by descending score; Python's reverse sort keeps ties in order.
fn ranked(scores: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
    indices
}

fn select(scores: &[f64], settings: &EvolutionSettings, rng: &mut impl DecisionSource) -> Vec<usize> {
    let count = settings.selection_size;
    let indices: Vec<usize> = match settings.selection_algorithm.as_str() {
        "best" => ranked(scores).into_iter().take(count).collect(),
        "tournament" => (0..count)
            .map(|_| {
                let mut best = rng.randrange(scores.len());
                for _ in 1..5 {
                    let i = rng.randrange(scores.len());
                    if scores[i] > scores[best] {
                        best = i;
                    }
                }
                best
            })
            .collect(),
        "roulette" => {
            let low = scores
                .iter()
                .copied()
                .fold(f64::INFINITY, |a, b| if b < a { b } else { a });
            let high = scores
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });
            let equal = if rng.game_arithmetic() {
                let (a, b) = (low as f32, high as f32);
                a == b || (a - b).abs() < (1e-6f32 * a.abs()).max(1e-6)
            } else {
                high == low
            };
            let weights: Vec<f64> = scores
                .iter()
                .map(|&s| if !equal { (s - low) / (high - low) } else { 1.0 })
                .collect();
            let total = if rng.game_arithmetic() {
                weights.iter().fold(0.0, |a, b| a + b)
            } else {
                crate::math::pymath::py_sum(weights.iter().copied())
            };
            (0..count)
                .map(|_| {
                    let draw = rng.random();
                    let mut cumulative = 0.0;
                    for (index, &weight) in weights.iter().enumerate() {
                        cumulative += weight / total;
                        if draw <= cumulative {
                            return index;
                        }
                    }
                    scores.len() - 1
                })
                .collect()
        }
        other => panic!("unknown selection algorithm: {other}"),
    };
    indices
}

pub(crate) const CROSSOVERS: [&str; 3] = ["none", "single_point", "uniform"];

/// The cars a reproduction breeds from, by index into the scored
/// population: `selected` in draw order (repeats kept), `preserved` best
/// first, and `best`, the highest-scoring car.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Choice {
    pub(crate) selected: Vec<usize>,
    pub(crate) preserved: Vec<usize>,
    pub(crate) best: usize,
}

fn preserve_size(settings: &EvolutionSettings) -> usize {
    match settings.preserve_parents.as_str() {
        "off" => 0,
        "on_selection_size" => settings.selection_size,
        "on_custom" => settings.preserve_parents_size,
        other => panic!("unknown parent preservation mode: {other}"),
    }
}

/// Selection and parent preservation: the first half of `reproduce`, and its
/// only random draws before crossover. `scores` must not be empty.
pub(crate) fn choose(scores: &[f64], settings: &EvolutionSettings, rng: &mut impl DecisionSource) -> Choice {
    let selected = select(scores, settings, rng);
    let ranked = ranked(scores);
    let preserved = ranked
        .iter()
        .copied()
        .take(preserve_size(settings).min(settings.population))
        .collect();
    Choice {
        selected,
        preserved,
        best: ranked[0],
    }
}

/// The settings `breed` and `breed_game` read, as checkpoints store them.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Breeding {
    pub(crate) population: usize,
    pub(crate) crossover: String,
    pub(crate) mutation_rate: f64,
    pub(crate) adaptive_mutation: bool,
    pub(crate) weight_decay: f64,
}

impl Breeding {
    pub(crate) fn of(s: &EvolutionSettings) -> Breeding {
        Breeding {
            population: s.population,
            crossover: s.crossover.clone(),
            mutation_rate: s.mutation_rate,
            adaptive_mutation: s.adaptive_mutation,
            weight_decay: s.weight_decay,
        }
    }

    /// Settings that breed like the ones `of` read; selection fields default.
    pub(crate) fn settings(&self) -> EvolutionSettings {
        EvolutionSettings {
            population: self.population,
            crossover: self.crossover.clone(),
            mutation_rate: self.mutation_rate,
            adaptive_mutation: self.adaptive_mutation,
            weight_decay: self.weight_decay,
            ..Default::default()
        }
    }
}

fn cross(parents: &[&Network], count: usize, algorithm: &str, rng: &mut impl DecisionSource) -> Vec<Network> {
    if count == 0 {
        return Vec::new();
    }
    assert!(!parents.is_empty(), "selection produced no parents");
    if algorithm == "none" {
        // No random decisions are needed; each child's copy is independent.
        // Indexed collection retains the parent order of the serial path.
        return (0..count)
            .into_par_iter()
            .map(|i| parents[i % parents.len()].clone())
            .collect();
    }
    (0..count)
        .map(|_| {
            let first = parents[rng.randrange(parents.len())];
            let second = parents[rng.randrange(parents.len())];
            let (a, b) = (&first.params, &second.params);
            let values = match algorithm {
                "single_point" => {
                    let cut = rng.randrange(a.len());
                    a[..cut].iter().chain(&b[cut..]).copied().collect()
                }
                "uniform" => a
                    .iter()
                    .zip(b)
                    .map(|(&x, &y)| if rng.random() < 0.5 { x } else { y })
                    .collect(),
                other => panic!("unknown crossover algorithm: {other}"),
            };
            Network::from_vector(&first.shape, values)
        })
        .collect()
}

/// `reproduce(agents, settings, rng)` without user-controlled agents.
pub fn reproduce(agents: &[AgentResult], settings: &EvolutionSettings, rng: &mut PyRandom) -> Generation {
    reproduce_with_scratch(agents, settings, rng, &mut Vec::new())
}

pub(crate) fn reproduce_with_scratch(
    agents: &[AgentResult],
    settings: &EvolutionSettings,
    rng: &mut PyRandom,
    normals: &mut Vec<f64>,
) -> Generation {
    reproduce_scored_with_scratch(agents, reward_values(agents, &settings.rewards), settings, rng, normals)
}

pub(crate) fn reproduce_scored_with_scratch(
    agents: &[AgentResult],
    scores: Vec<f64>,
    settings: &EvolutionSettings,
    rng: &mut PyRandom,
    normals: &mut Vec<f64>,
) -> Generation {
    validate_scores(agents, &scores);
    let mut profile = crate::training::training_profile::Profile::new("reproduce_python");
    if agents.is_empty() {
        return Generation {
            networks: Vec::new(),
            preserved_count: 0,
            rewards: Vec::new(),
        };
    }
    profile.mark("rewards");
    let choice = choose(&scores, settings, rng);
    let selected: Vec<&Network> = choice.selected.iter().map(|&i| agents[i].network).collect();
    let preserved: Vec<Network> = choice.preserved.iter().map(|&i| agents[i].network.clone()).collect();
    profile.mark("selection");
    let preserved_count = preserved.len();
    let networks = breed(&selected, preserved, settings, rng, normals, &mut profile);
    Generation {
        networks,
        preserved_count,
        rewards: scores,
    }
}

/// Crossover, mutation and weight decay: the second half of `reproduce`.
/// Returns `preserved` unchanged, then `settings.population` minus them
/// children of `selected`. Every draw comes from `rng`, so the same parents,
/// settings and generator state give the same networks.
pub(crate) fn breed(
    selected: &[&Network],
    preserved: Vec<Network>,
    settings: &EvolutionSettings,
    rng: &mut PyRandom,
    normals: &mut Vec<f64>,
    profile: &mut crate::training::training_profile::Profile,
) -> Vec<Network> {
    let count = settings.population.saturating_sub(preserved.len());
    // Crossover "none" copies parents in order without random decisions; that
    // copy is folded into the mutation pass below instead of materialized.
    // Other algorithms build the children here, consuming the stream as before.
    let copies = settings.crossover == "none";
    if copies && count > 0 {
        assert!(!selected.is_empty(), "selection produced no parents");
    }
    let crossed: Vec<Network> = if copies {
        Vec::new()
    } else {
        cross(selected, count, &settings.crossover, rng)
    };
    let source = |i: usize| -> &Network {
        if copies {
            selected[i % selected.len()]
        } else {
            &crossed[i]
        }
    };
    profile.mark("crossover");
    let mut networks = preserved;
    // Children draw their mutation noise one after another from the shared
    // stream; draw it all in order, then mutate the children in parallel.
    let mut draws = 0;
    let offsets: Vec<usize> = (0..count)
        .map(|i| {
            let start = draws;
            if settings.mutation_rate > 0.0 {
                draws += source(i).params.len();
            }
            start
        })
        .collect();
    rng.standard_normals_into(draws, normals);
    profile.mark("normal_draws");
    let mutate = |child: Network, start: usize| {
        let child = if settings.mutation_rate > 0.0 {
            let normals = &normals[start..start + child.params.len()];
            mutate_xavier_with(child, settings.mutation_rate, settings.adaptive_mutation, normals)
        } else {
            child
        };
        weight_decay(child, settings.weight_decay)
    };
    let mutated: Vec<Network> = if copies {
        (0..count)
            .into_par_iter()
            .zip(offsets)
            .map(|(i, start)| {
                let parent = source(i);
                if settings.mutation_rate > 0.0 {
                    let normals = &normals[start..start + parent.params.len()];
                    mutated_copy(
                        parent,
                        settings.mutation_rate,
                        settings.adaptive_mutation,
                        normals,
                        settings.weight_decay,
                    )
                } else {
                    weight_decay(parent.clone(), settings.weight_decay)
                }
            })
            .collect()
    } else {
        crossed
            .into_par_iter()
            .zip(offsets)
            .map(|(child, start)| mutate(child, start))
            .collect()
    };
    networks.extend(mutated);
    profile.mark("mutation");
    networks
}

/// Reproduction using the original game's independent decision/normal streams.
pub fn reproduce_game(
    agents: &[AgentResult],
    settings: &EvolutionSettings,
    rng: &mut crate::training::game_random::GameRandom,
    math: MathProfile,
) -> Generation {
    reproduce_game_scored(agents, reward_values(agents, &settings.rewards), settings, rng, math)
}

pub(crate) fn validate_scores(agents: &[AgentResult], scores: &[f64]) {
    assert_eq!(agents.len(), scores.len(), "fitness must match the ordered population");
    assert!(scores.iter().all(|score| score.is_finite()), "fitness must be finite");
}

pub(crate) fn reproduce_game_scored(
    agents: &[AgentResult],
    scores: Vec<f64>,
    settings: &EvolutionSettings,
    rng: &mut crate::training::game_random::GameRandom,
    math: MathProfile,
) -> Generation {
    validate_scores(agents, &scores);
    let mut profile = crate::training::training_profile::Profile::new("reproduce_game");
    if agents.is_empty() {
        return Generation {
            networks: Vec::new(),
            preserved_count: 0,
            rewards: Vec::new(),
        };
    }
    profile.mark("rewards");
    let choice = choose(&scores, settings, rng);
    let selected: Vec<&Network> = choice.selected.iter().map(|&i| agents[i].network).collect();
    let preserved: Vec<Network> = choice.preserved.iter().map(|&i| agents[i].network.clone()).collect();
    let preserved_count = preserved.len();
    profile.mark("selection");
    let networks = breed_game(&selected, preserved, settings, rng, math, &mut profile);
    Generation {
        networks,
        preserved_count,
        rewards: scores,
    }
}

/// `breed` with the original game's streams.
pub(crate) fn breed_game(
    selected: &[&Network],
    mut networks: Vec<Network>,
    settings: &EvolutionSettings,
    rng: &mut crate::training::game_random::GameRandom,
    math: MathProfile,
    profile: &mut crate::training::training_profile::Profile,
) -> Vec<Network> {
    let children = cross(
        selected,
        settings.population.saturating_sub(networks.len()),
        &settings.crossover,
        rng,
    );
    profile.mark("crossover");
    for child in children {
        let rate = settings.mutation_rate
            * if settings.adaptive_mutation {
                mutation_factor(&child, "xavier", math)
            } else {
                1.0
            };
        let mut child = child.mutate_xavier_game(rate, rng, math);
        if settings.weight_decay == 1.0 {
            child.params.fill(0.0);
        } else {
            child = weight_decay(child, settings.weight_decay);
        }
        networks.push(child);
    }
    profile.mark("mutation");
    networks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trusted_mcp_settings_keep_cli_coercions_and_population_override() {
        let data = json!({"settings":{
            "population":0,"selection_size":2.75,"preserve_parents_size":1.5,"rewards":[]
        }});
        let mut settings = EvolutionSettings::from_mcp(&data);
        assert_eq!(settings.population, 0);
        assert_eq!(settings.selection_size, 2);
        assert_eq!(settings.preserve_parents_size, 1);
        settings.population = 4; // The CLI applies --population after parsing.
        settings.validate_algorithm().unwrap();
        let network = Network::xavier(&[2, 2], &mut PyRandom::new(7));
        let agents = [
            AgentResult {
                network: &network,
                metrics: [Some(1.0); 14],
                update_count: 1,
            },
            AgentResult {
                network: &network,
                metrics: [Some(2.0); 14],
                update_count: 1,
            },
        ];
        let generation = reproduce(&agents, &settings, &mut PyRandom::new(8));
        assert_eq!(generation.networks.len(), 4);
        assert_eq!(generation.rewards, vec![0.0, 0.0]);

        let parsed = EvolutionSettings::try_from_mcp(&json!({
            "rewards":[{"metric":"total_score","weight":1,"type":"legacy-kind"}]
        }))
        .unwrap();
        assert_eq!(parsed.rewards[0].kind, "legacy-kind");
        assert_eq!(
            reward_values(&agents, &parsed.rewards),
            reward_values(&agents, &EvolutionSettings::default().rewards)
        );
    }

    #[test]
    fn session_mcp_settings_validate_after_overrides_without_cli_coercions() {
        assert_eq!(
            EvolutionSettings::try_from_mcp_with_population(&json!({"population":0}), Some(4))
                .unwrap()
                .population,
            4
        );
        for data in [
            json!({"population":0}),
            json!({"rewards":[]}),
            json!({"selection_size":2.75}),
            json!({"preserve_parents_size":1.5}),
            json!({"rewards":[{"metric":"total_score","weight":1,"type":"legacy-kind"}]}),
        ] {
            assert!(
                EvolutionSettings::try_from_mcp_with_population(&data, None).is_err(),
                "{data}"
            );
        }
    }

    /// Sequential reference: clone every child, then mutate and decay it in place.
    fn reproduce_reference(agents: &[AgentResult], settings: &EvolutionSettings, rng: &mut PyRandom) -> Generation {
        let scores = reward_values(agents, &settings.rewards);
        let selected: Vec<&Network> = select(&scores, settings, rng)
            .into_iter()
            .map(|i| agents[i].network)
            .collect();
        let size = preserve_size(settings);
        let preserved: Vec<Network> = ranked(&scores)
            .into_iter()
            .take(size.min(settings.population))
            .map(|i| agents[i].network.clone())
            .collect();
        let children = cross(
            &selected,
            settings.population.saturating_sub(preserved.len()),
            &settings.crossover,
            rng,
        );
        let preserved_count = preserved.len();
        let mut networks = preserved;
        for child in children {
            let child = if settings.mutation_rate > 0.0 {
                mutate_xavier(&child, settings.mutation_rate, rng, settings.adaptive_mutation)
            } else {
                child
            };
            networks.push(weight_decay(child, settings.weight_decay));
        }
        Generation {
            networks,
            preserved_count,
            rewards: scores,
        }
    }

    #[test]
    fn fused_copy_mutation_matches_sequential_reproduction() {
        let shape = [6, 5, 4];
        let mut seed = PyRandom::new(3);
        let networks: Vec<Network> = (0..37).map(|_| Network::xavier(&shape, &mut seed)).collect();
        let agents: Vec<AgentResult> = networks
            .iter()
            .enumerate()
            .map(|(i, network)| {
                let mut metrics: Metrics = [None; 14];
                metrics[0] = Some(((i * 7919) % 23) as f64 * 0.5);
                AgentResult {
                    network,
                    metrics,
                    update_count: 10 + i as u64,
                }
            })
            .collect();
        let mut variants = Vec::new();
        for crossover in ["none", "single_point", "uniform"] {
            for (mutation_rate, adaptive, weight_decay) in [
                (0.0, false, 0.0),
                (0.3, false, 0.0),
                (0.3, true, 0.0005000000237487257),
                (1e-9, true, 0.0),
            ] {
                for (selection_algorithm, preserve_parents, population) in [
                    ("best", "on_selection_size", 40),
                    ("tournament", "on_custom", 33),
                    ("roulette", "off", 5),
                ] {
                    variants.push(EvolutionSettings {
                        population,
                        selection_algorithm: selection_algorithm.into(),
                        selection_size: 3,
                        crossover: crossover.into(),
                        mutation_rate,
                        adaptive_mutation: adaptive,
                        weight_decay,
                        preserve_parents: preserve_parents.into(),
                        preserve_parents_size: 2,
                        ..Default::default()
                    });
                }
            }
        }
        let mut scratch = vec![f64::NAN; 5];
        for (v, settings) in variants.iter().enumerate() {
            let (mut a, mut b) = (PyRandom::new(11 + v as i64), PyRandom::new(11 + v as i64));
            a.gauss(1.0); // a cached normal must survive both paths identically
            b.gauss(1.0);
            let expected = reproduce_reference(&agents, settings, &mut a);
            let actual = reproduce_with_scratch(&agents, settings, &mut b, &mut scratch);
            assert_eq!(actual.preserved_count, expected.preserved_count, "variant {v}");
            assert_eq!(actual.rewards, expected.rewards, "variant {v}");
            assert_eq!(actual.networks.len(), expected.networks.len(), "variant {v}");
            for (i, (x, y)) in actual.networks.iter().zip(&expected.networks).enumerate() {
                assert_eq!(x.shape, y.shape);
                let (xb, yb): (Vec<u64>, Vec<u64>) = (
                    x.params.iter().map(|p| p.to_bits()).collect(),
                    y.params.iter().map(|p| p.to_bits()).collect(),
                );
                assert_eq!(xb, yb, "variant {v} network {i}");
            }
            assert_eq!(a.to_json(), b.to_json(), "variant {v}: generator state");
            assert_eq!(a.gauss(1.0).to_bits(), b.gauss(1.0).to_bits(), "variant {v}: next draw");
        }
    }

    fn scored_agents(networks: &[Network]) -> Vec<AgentResult<'_>> {
        networks
            .iter()
            .enumerate()
            .map(|(i, network)| {
                let mut metrics: Metrics = [None; 14];
                metrics[0] = Some(((i * 7919) % 23) as f64 * 0.5);
                AgentResult {
                    network,
                    metrics,
                    update_count: 10 + i as u64,
                }
            })
            .collect()
    }

    fn breeding_variants() -> Vec<EvolutionSettings> {
        let mut variants = Vec::new();
        for crossover in CROSSOVERS {
            for (mutation_rate, adaptive, weight_decay) in
                [(0.0, false, 0.0), (0.3, true, 0.0005000000237487257), (1e-9, true, 1.0)]
            {
                for (selection_algorithm, preserve_parents, population) in [
                    ("best", "on_selection_size", 40),
                    ("tournament", "on_custom", 33),
                    ("roulette", "off", 5),
                ] {
                    variants.push(EvolutionSettings {
                        population,
                        selection_algorithm: selection_algorithm.into(),
                        selection_size: 3,
                        crossover: crossover.into(),
                        mutation_rate,
                        adaptive_mutation: adaptive,
                        weight_decay,
                        preserve_parents: preserve_parents.into(),
                        preserve_parents_size: 2,
                        ..Default::default()
                    });
                }
            }
        }
        variants
    }

    fn bits(networks: &[Network]) -> Vec<Vec<u64>> {
        networks
            .iter()
            .map(|n| n.params.iter().map(|p| p.to_bits()).collect())
            .collect()
    }

    /// Breeding from copies of the chosen parents, with the generator state
    /// saved after selection and only the settings `Breeding` keeps, repeats
    /// `reproduce` exactly.
    #[test]
    fn breeding_replays_from_the_chosen_parents_and_generator_state() {
        let mut seed = PyRandom::new(3);
        let networks: Vec<Network> = (0..37).map(|_| Network::xavier(&[6, 5, 4], &mut seed)).collect();
        let agents = scored_agents(&networks);
        for (v, settings) in breeding_variants().iter().enumerate() {
            let scores = reward_values(&agents, &settings.rewards);
            let mut expected_rng = PyRandom::new(11 + v as i64);
            expected_rng.gauss(1.0);
            let mut chooser = expected_rng.clone();
            let expected = reproduce(&agents, settings, &mut expected_rng);
            let choice = choose(&scores, settings, &mut chooser);
            assert_eq!(
                choice.preserved.first().copied().unwrap_or(choice.best),
                choice.best,
                "variant {v}"
            );
            let selected: Vec<Network> = choice.selected.iter().map(|&i| agents[i].network.clone()).collect();
            let selected: Vec<&Network> = selected.iter().collect();
            let preserved = choice.preserved.iter().map(|&i| agents[i].network.clone()).collect();
            let mut replay = chooser.clone();
            let mut profile = crate::training::training_profile::Profile::new("test");
            let actual = breed(
                &selected,
                preserved,
                &Breeding::of(settings).settings(),
                &mut replay,
                &mut Vec::new(),
                &mut profile,
            );
            assert_eq!(bits(&actual), bits(&expected.networks), "variant {v}");
            assert_eq!(replay.to_json(), expected_rng.to_json(), "variant {v}: generator state");

            let mut game = crate::training::game_random::GameRandom::new([1, 2, 3, 4 + v as u64], 7 + v as i32);
            let mut game_chooser = game.clone();
            let math = MathProfile::ALL[v % 3];
            let expected = reproduce_game(&agents, settings, &mut game, math);
            let choice = choose(&scores, settings, &mut game_chooser);
            let selected: Vec<&Network> = choice.selected.iter().map(|&i| agents[i].network).collect();
            let preserved = choice.preserved.iter().map(|&i| agents[i].network.clone()).collect();
            let actual = breed_game(
                &selected,
                preserved,
                &Breeding::of(settings).settings(),
                &mut game_chooser,
                math,
                &mut profile,
            );
            assert_eq!(bits(&actual), bits(&expected.networks), "game variant {v}");
            assert_eq!(
                game_chooser.to_json(),
                game.to_json(),
                "game variant {v}: generator state"
            );
        }
    }
}
