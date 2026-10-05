//! Argument parsing and values shared by the CLI commands.

use super::io::load;
use altd_sim::track::training_tracks::RandomTrainingTrackSettings;
use altd_sim::training::evolution::RewardSpec;
use altd_sim::training::Mode;
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde_json::Value;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "altd-sim",
    about = "CPU game-fidelity simulator and trainer for AI Learns To Drive",
    version = super::platform::VERSION
)]
pub(super) struct Cli {
    /// Worker threads (default: all logical CPUs in the affinity mask).
    #[arg(long, global = true)]
    pub(super) threads: Option<usize>,
    /// Training execution: each car runs a whole window independently, or all
    /// cars advance one tick at a time like `TrainingRunner.step`.
    #[arg(long, global = true, value_enum, default_value_t = ModeArg::Independent)]
    pub(super) mode: ModeArg,
    #[command(subcommand)]
    pub(super) command: Command,
}

#[derive(Clone, Copy, ValueEnum)]
pub(super) enum ModeArg {
    Independent,
    Lockstep,
}

#[derive(Clone, Copy, PartialEq, ValueEnum)]
pub(super) enum TrackMode {
    Fixed,
    Random,
}

#[derive(Args)]
pub(super) struct TrackOptions {
    /// Use the scene's fixed track or a fresh CPU-generated track each generation.
    #[arg(long, value_enum, default_value_t = TrackMode::Fixed)]
    pub(super) track_mode: TrackMode,
    /// JSON TrackFactory settings with fixed values, min/max lengths, or choices.
    #[arg(long)]
    pub(super) random_track_settings: Option<PathBuf>,
    /// Number of prepared tracks queued by the CPU producer, including with --gpu.
    #[arg(long, default_value_t = 8)]
    pub(super) track_buffer_size: usize,
    /// Evaluate the unchanged population on this many random tracks before breeding.
    /// Resume restores the checkpoint value when omitted.
    #[arg(long)]
    pub(super) tracks_per_generation: Option<usize>,
}

impl TrackOptions {
    pub(super) fn count(&self, checkpoint: Option<&Value>) -> usize {
        let count = self.tracks_per_generation.unwrap_or_else(|| {
            checkpoint
                .and_then(|value| value["tracks_per_generation"].as_u64())
                .unwrap_or(1) as usize
        });
        assert!(count > 0, "--tracks-per-generation must be positive");
        assert!(
            self.track_mode == TrackMode::Random || count == 1,
            "--tracks-per-generation above 1 requires --track-mode random"
        );
        count
    }
    pub(super) fn settings(&self) -> Option<RandomTrainingTrackSettings> {
        if self.track_mode == TrackMode::Fixed {
            assert!(
                self.random_track_settings.is_none(),
                "--random-track-settings requires --track-mode random"
            );
            return None;
        }
        let settings: RandomTrainingTrackSettings = self
            .random_track_settings
            .as_deref()
            .map(|path| serde_json::from_value(load(path)).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
            .unwrap_or_default();
        settings
            .validate()
            .unwrap_or_else(|e| panic!("--random-track-settings: {e}"));
        assert!(self.track_buffer_size > 0, "--track-buffer-size must be positive");
        Some(settings)
    }
}

impl From<ModeArg> for Mode {
    fn from(mode: ModeArg) -> Mode {
        match mode {
            ModeArg::Independent => Mode::Independent,
            ModeArg::Lockstep => Mode::Lockstep,
        }
    }
}

/// Mutation rate interpolation between `--mutation-start` and `--mutation-end`.
#[derive(Clone, Copy, ValueEnum, PartialEq)]
pub(super) enum Schedule {
    /// Constant ratio per generation (equal time per halving).
    Geometric,
    Linear,
}

#[derive(Clone, Copy, ValueEnum)]
pub(super) enum ScratchReward {
    /// Distance along the track (total_score).
    Distance,
    /// Faster completed laps rank higher (best_lap_performance).
    BestLapTime,
}

impl ScratchReward {
    pub(super) fn spec(self) -> RewardSpec {
        let metric = match self {
            Self::Distance => "total_score",
            Self::BestLapTime => "best_lap_performance",
        };
        RewardSpec {
            metric: metric.into(),
            weight: 100,
            kind: "default".into(),
        }
    }
}

/// Metric watched by `--stop-plateau`.
#[derive(Clone, Copy, ValueEnum, PartialEq)]
pub(super) enum PlateauMetric {
    /// Fastest completed lap of a generation.
    Lap,
    /// Best total_score of a generation.
    Score,
}

#[derive(Subcommand)]
pub(super) enum Command {
    /// Time training ticks and reproduction for a population.
    Bench {
        #[arg(long)]
        scene: PathBuf,
        #[arg(long)]
        network: PathBuf,
        /// Vehicle model JSON (default: the built-in copy of assets/models/rally.json).
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long)]
        spawn_trace: PathBuf,
        #[arg(long, default_value_t = 1000)]
        population: usize,
        #[arg(long, default_value_t = 60)]
        ticks: u64,
        #[arg(long, default_value_t = 2)]
        warmup_ticks: u64,
        #[arg(long, default_value_t = 7)]
        seed: i64,
        #[arg(long, default_value_t = 0)]
        spawn_index: usize,
        #[arg(long, default_value_t = 2)]
        batch_count: usize,
        #[arg(long)]
        report: Option<PathBuf>,
        /// Write every car's final state, metrics, and the next generation's
        /// networks for deterministic comparisons.
        #[arg(long)]
        dump_state: Option<PathBuf>,
    },
    /// Run and evolve imported networks for several generations.
    Train {
        #[command(flatten)]
        tracks: TrackOptions,
        #[arg(long)]
        scene: PathBuf,
        #[arg(long)]
        network: PathBuf,
        #[arg(long)]
        model: PathBuf,
        #[arg(long, required_if_eq("track_mode", "fixed"))]
        spawn_trace: Option<PathBuf>,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        settings: Option<PathBuf>,
        #[arg(long)]
        population: Option<usize>,
        #[arg(long, default_value_t = 2)]
        generations: usize,
        #[arg(long)]
        ticks: Option<u64>,
        #[arg(long, default_value_t = 7)]
        seed: i64,
        /// Exact game RNG state JSON containing independent decisions and normals streams.
        #[arg(long)]
        game_rng_state: Option<PathBuf>,
        #[arg(long, default_value_t = 2)]
        batch_count: usize,
        #[arg(long, overrides_with = "no_eliminate_on_wall")]
        eliminate_on_wall: bool,
        #[arg(long)]
        no_eliminate_on_wall: bool,
        #[arg(long, overrides_with = "no_idle_eliminate")]
        idle_eliminate: bool,
        #[arg(long)]
        no_idle_eliminate: bool,
    },
    /// Train a network from a random Xavier initialization with a decaying
    /// mutation rate, logging lap times each generation and saving the network
    /// behind every new best lap. Defaults to Rally sensor and output metadata.
    TrainScratch {
        #[command(flatten)]
        tracks: TrackOptions,
        #[arg(long)]
        scene: PathBuf,
        /// Trace whose first frame is the spawn pose.
        #[arg(long, required_if_eq("track_mode", "fixed"))]
        spawn_trace: Option<PathBuf>,
        /// Network export supplying the input sensor and output control names; its weights are unused
        /// (default: the built-in copy of assets/networks/rally.json).
        #[arg(long)]
        network: Option<PathBuf>,
        /// Vehicle model JSON (default: the built-in copy of assets/models/rally.json).
        #[arg(long)]
        model: Option<PathBuf>,
        /// Run directory: log, best-lap networks, and checkpoints.
        #[arg(long, default_value = "runs/example")]
        out_dir: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "20,16,16,16,16,12,12,8,5")]
        shape: Vec<usize>,
        #[arg(long, default_value_t = 8192)]
        population: usize,
        #[arg(long, default_value_t = 50000)]
        generations: usize,
        /// Simulated ticks per generation (60 per second).
        #[arg(long, default_value_t = 5400)]
        ticks: u64,
        /// Mutation rate of generation 0; adaptive mutation then scales it by network shape.
        #[arg(long, default_value_t = 0.4)]
        mutation_start: f64,
        /// Mutation rate of the last generation.
        #[arg(long, default_value_t = 0.0125)]
        mutation_end: f64,
        #[arg(long, value_enum, default_value_t = Schedule::Geometric)]
        schedule: Schedule,
        /// Evolution settings JSON overriding the generalist defaults (population and mutation rate excluded).
        #[arg(long)]
        settings: Option<PathBuf>,
        /// Selection reward. Overrides settings-file rewards; otherwise defaults to distance.
        #[arg(long, value_enum)]
        reward: Option<ScratchReward>,
        #[arg(long, default_value_t = 1)]
        seed: i64,
        /// Exact game RNG state JSON containing independent decisions and normals streams.
        #[arg(long)]
        game_rng_state: Option<PathBuf>,
        /// Seed the first generation from this network export instead of a random Xavier network.
        #[arg(long)]
        init_network: Option<PathBuf>,
        /// Continue another run's population: the checkpoint.json of a run with the same shape,
        /// e.g. from another simulator. Takes its networks, generation (mutation
        /// schedule) and RNG; best laps start over.
        #[arg(long, conflicts_with = "init_network")]
        init_population: Option<PathBuf>,
        #[arg(long, default_value_t = 2)]
        batch_count: usize,
        /// Deactivate cars on wall contact. Overrides the settings file.
        #[arg(long, overrides_with = "no_eliminate_on_wall")]
        eliminate_on_wall: bool,
        #[arg(long, overrides_with = "eliminate_on_wall")]
        no_eliminate_on_wall: bool,
        /// Deactivate cars after sustained lack of forward progress. Overrides the settings file.
        #[arg(long, overrides_with = "no_idle_eliminate")]
        idle_eliminate: bool,
        #[arg(long, overrides_with = "idle_eliminate")]
        no_idle_eliminate: bool,
        /// Write a resumable checkpoint every N generations and when training ends (0 disables).
        #[arg(long, default_value_t = 100)]
        checkpoint_every: usize,
        /// Export the final complete batch fitness leader before reproduction.
        #[arg(long)]
        save_final_candidate: bool,
        /// Stop after a generation whose best total_score reaches this value.
        #[arg(long)]
        stop_score_above: Option<f64>,
        /// Stop after a generation whose fastest lap is at most this many seconds.
        #[arg(long)]
        stop_lap_below: Option<f64>,
        /// Stop after a generation in which at least this percentage of cars completed a lap.
        #[arg(long)]
        stop_lapped_percent: Option<f64>,
        /// Stop once --plateau-metric has not improved for this many generations of this invocation.
        #[arg(long)]
        stop_plateau: Option<usize>,
        #[arg(long, value_enum, default_value_t = PlateauMetric::Lap)]
        plateau_metric: PlateauMetric,
        /// Continue from the checkpoint in --out-dir, allowing new settings with the same network shape.
        #[arg(long)]
        resume: bool,
        /// Simulate the generations on the GPU (gpu/sim, bit-exact with the CPU).
        /// `gpu-info` shows which library this loads.
        #[arg(long)]
        gpu: bool,
    },
    /// Evaluate frozen weights on an immutable shared suite without breeding.
    Evaluate {
        #[arg(long)]
        network: PathBuf,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        suite: PathBuf,
        #[arg(long)]
        report: PathBuf,
    },
    /// Replay recorded controls and compare the resulting trajectory.
    CompareTrace {
        scene: PathBuf,
        trace: PathBuf,
        report: PathBuf,
        #[arg(long, default_value_t = 1)]
        start_index: usize,
        #[arg(long)]
        end_index: Option<usize>,
    },
    /// Compare one simulated transition from each recorded state.
    CompareOneStep {
        scene: PathBuf,
        trace: PathBuf,
        report: PathBuf,
    },
    /// Compare network inference and physics together against a recorded episode.
    CompareClosedLoop {
        scene: PathBuf,
        trace: PathBuf,
        network: PathBuf,
        model: PathBuf,
        report: PathBuf,
        #[arg(long, default_value_t = 2)]
        batch_count: usize,
        #[arg(long)]
        new_vehicle: bool,
    },
    /// Compare network inference on recorded sensors.
    CompareNetwork {
        network: PathBuf,
        trace: PathBuf,
        report: PathBuf,
        #[arg(long, default_value_t = 2)]
        first_tick: usize,
    },
    /// Compare sensors on recorded states.
    CompareSensors {
        scene: PathBuf,
        trace: PathBuf,
        model: PathBuf,
        trajectory_report: PathBuf,
        sensor_report: PathBuf,
        #[arg(long)]
        all_frames: bool,
    },
    /// Compare path scores on recorded positions.
    CompareScore {
        scene: PathBuf,
        trace: PathBuf,
        report: PathBuf,
    },
    /// Serve simulations to Drive Lab over a WebSocket on 127.0.0.1
    /// (docs/server.md).
    #[cfg(feature = "server")]
    Serve {
        /// TCP port on 127.0.0.1.
        #[arg(long, default_value_t = altd_sim::server::DEFAULT_PORT)]
        port: u16,
        /// A web origin (scheme://host[:port]) allowed to connect; repeat for
        /// several. Replaces the default https://drivinglab.jectrum.de.
        #[arg(long = "allow-origin", value_name = "ORIGIN", value_parser = altd_sim::server::protocol::normalize_origin)]
        allow_origin: Vec<String>,
    },
    /// Load the HIP simulator library used by --gpu and check that it matches this binary.
    /// Needs a ROCm runtime but no GPU.
    GpuInfo {
        /// Library path (default: ALTD_GPU_LIB, then beside the executable, then target/gpu).
        #[arg(long)]
        library: Option<PathBuf>,
    },
}

pub(super) fn flag(yes: bool, no: bool) -> Option<bool> {
    if yes {
        Some(true)
    } else if no {
        Some(false)
    } else {
        None
    }
}
