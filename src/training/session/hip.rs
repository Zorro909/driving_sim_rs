//! The HIP backend of `Session`: driving windows run on libaltd_gpu.so while
//! reproduction, checkpoints and every reader stay on the CPU runner, which
//! each window reads back into.

use crate::gpu::hip::{Gpu, GpuWorld, PreparedGpuWorld};
use crate::gpu::simulation::GpuSim;
use crate::track::world::World;
use crate::training::TrainingRunner;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

/// Ticks of the startup comparison with the CPU simulator.
const VERIFY_TICKS: u64 = 12;

/// The process's HIP library and device name, opened on first use and never
/// closed, so sessions can own `'static` device handles.
static GPU: OnceLock<Result<(Gpu, String), String>> = OnceLock::new();

/// Set while a session holds the device: one HIP session per process.
static IN_USE: AtomicBool = AtomicBool::new(false);

/// The HIP device and its name, or why it cannot be used.
pub fn device() -> Result<(&'static Gpu, &'static str), String> {
    let opened = GPU.get_or_init(|| {
        let gpu = Gpu::open(None)?;
        crate::gpu::simulation::layout_matches(&gpu)?;
        let name = gpu.device_name()?;
        Ok((gpu, name))
    });
    match opened {
        Ok((gpu, name)) => Ok((gpu, name.as_str())),
        Err(e) => Err(e.clone()),
    }
}

/// The device claim of one session, released when the session drops.
struct Claim;

impl Claim {
    fn take() -> Option<Claim> {
        IN_USE
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| Claim)
    }
}

impl Drop for Claim {
    fn drop(&mut self) {
        IN_USE.store(false, Ordering::Release);
    }
}

pub(super) struct HipState {
    // Declared before `world`, so it drops first: it refers to the world.
    sim: GpuSim<'static>,
    world: GpuWorld<'static>,
    capacity: usize,
    verified: bool,
    _claim: Claim,
}

fn upload_world(gpu: &'static Gpu, world: &World) -> Result<GpuWorld<'static>, String> {
    Ok(GpuWorld::from_prepared(gpu, PreparedGpuWorld::new(world)?))
}

impl HipState {
    /// A simulator for `runner`'s world, or why the runner needs the CPU.
    pub(super) fn new(runner: &TrainingRunner) -> Result<HipState, String> {
        let (gpu, _) = device()?;
        if runner.world.track.native_broadphase {
            return Err("the HIP simulator does not model the native broadphase".into());
        }
        let claim = Claim::take().ok_or("another session is using the GPU")?;
        let world = upload_world(gpu, &runner.world)?;
        let capacity = runner.settings.population.max(1);
        let sim = runner.gpu_sim(&world, capacity)?;
        Ok(HipState {
            sim,
            world,
            capacity,
            verified: false,
            _claim: claim,
        })
    }

    /// Uploads the networks again before the next window (after a start,
    /// restore or network change that kept the generation number).
    pub(super) fn invalidate_networks(&mut self) {
        self.sim.network_tag = None;
        self.verified = false;
    }

    /// Switches to the runner's new world and checks it again before the
    /// next window.
    #[cfg(any(test, feature = "server"))]
    pub(super) fn replace_world(&mut self, runner: &TrainingRunner) -> Result<(), String> {
        let world = upload_world(self.world.gpu(), &runner.world)?;
        self.sim.set_world(&world);
        // The simulator uses the new world now; the old one can go.
        self.world = world;
        self.verified = false;
        Ok(())
    }

    /// Rebuilds the simulator when the population outgrew it.
    fn fit(&mut self, runner: &TrainingRunner) -> Result<(), String> {
        let population = runner.agents.len();
        if population > self.capacity {
            self.sim = runner.gpu_sim(&self.world, population)?;
            self.capacity = population;
        }
        Ok(())
    }

    /// Whether the startup comparison still has to run.
    pub(super) fn needs_verify(&self) -> bool {
        !self.verified
    }

    /// Runs `VERIFY_TICKS` on the CPU and on HIP from the same state and
    /// compares every car and agent exactly; the runner ends as it started.
    pub(super) fn verify(&mut self, runner: &mut TrainingRunner) -> Result<(), String> {
        self.fit(runner)?;
        let surfaces = &self.world.arrays.surfaces;
        let (backup, tick, batch) = (runner.agents.clone(), runner.tick, runner.batch_index);
        runner.advance(VERIFY_TICKS, false);
        let expected = runner.canonical_state(surfaces);
        runner.agents = backup.clone();
        runner.tick = tick;
        runner.batch_index = batch;
        self.sim.network_tag = None;
        let actual = expected.and_then(|expected| {
            runner.advance_window_gpu(&mut self.sim, &self.world, VERIFY_TICKS, false, None)?;
            Ok((expected, runner.canonical_state(surfaces)?))
        });
        runner.agents = backup;
        runner.tick = tick;
        runner.batch_index = batch;
        let (expected, actual) = actual?;
        if expected != actual {
            return Err("HIP verification differs from the CPU simulator".into());
        }
        self.verified = true;
        Ok(())
    }

    pub(super) fn advance(&mut self, runner: &mut TrainingRunner, ticks: u64, stop: bool) -> Result<u64, String> {
        self.fit(runner)?;
        runner.advance_window_gpu(&mut self.sim, &self.world, ticks, stop, None)
    }

    pub(super) fn advance_generation(&mut self, runner: &mut TrainingRunner, limit: u64) -> Result<u64, String> {
        self.fit(runner)?;
        runner.advance_generation_gpu(&mut self.sim, &self.world, limit)
    }
}
