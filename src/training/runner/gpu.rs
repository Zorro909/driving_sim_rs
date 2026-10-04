//! GPU state transfers, driving windows, and population turnover.

use super::TrainingRunner;
#[cfg(not(target_arch = "wasm32"))]
use super::Turnover;
#[cfg(not(target_arch = "wasm32"))]
use crate::physics::car::Sensor;
#[cfg(not(target_arch = "wasm32"))]
use crate::training::evolution::{AgentResult, Generation};
use crate::training::TrainingAgent;
use rayon::prelude::*;

/// Fills `cars` and `out` with the GPU state of `agents` (`Car::gpu_export`
/// and `agent_export`) in place, resizing them to the population. Retaining
/// the vectors between windows keeps their pages mapped, and the indexed
/// parallel fill avoids the linked-list concatenation of a `Result` collect.
pub fn export_state_into(
    agents: &[TrainingAgent],
    vehicle: &crate::track::world::VehicleConfig,
    surfaces: &crate::gpu::simulation::SurfaceTable,
    cars: &mut Vec<crate::gpu::simulation::GpuCar>,
    out: &mut Vec<crate::gpu::simulation::GpuAgent>,
) -> Result<(), String> {
    cars.resize(agents.len(), crate::gpu::simulation::GpuCar::zeroed());
    out.resize(agents.len(), crate::gpu::simulation::GpuAgent::default());
    agents
        .par_iter()
        .zip(cars.par_iter_mut())
        .zip(out.par_iter_mut())
        .try_for_each(|((a, c), g)| {
            *c = a.car.gpu_export(vehicle, surfaces)?;
            *g = crate::gpu::simulation::agent_export(a)?;
            Ok::<(), String>(())
        })
}

impl TrainingRunner {
    /// A GPU simulator for this runner's world, vehicle and sensors, with
    /// room for `capacity` agents.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn gpu_sim<'a>(
        &self,
        world: &crate::gpu::hip::GpuWorld<'a>,
        capacity: usize,
    ) -> Result<crate::gpu::simulation::GpuSim<'a>, String> {
        let vehicle = crate::gpu::simulation::vehicle_desc(&self.world.vehicle)?;
        // The GPU path sensors sample the Curve2D only; without track.curve the
        // CPU falls back to the baked path, which the GPU does not implement.
        let path_sensor = self
            .layout
            .sensors
            .iter()
            .any(|s| matches!(s, Sensor::CorrectDirection | Sensor::TrackCurvature { .. }));
        if path_sensor && self.world.track.curve.is_none() && self.world.track.path.len() >= 2 {
            return Err(
                "the scene has no track.curve (Curve2D control points), which the correct_direction and \
                        track_curvature sensors need on the GPU; add the curve to the scene or train without --gpu"
                    .into(),
            );
        }
        let sensors: Vec<_> = self
            .layout
            .sensors
            .iter()
            .map(crate::gpu::simulation::sensor_desc)
            .collect();
        Ok(crate::gpu::simulation::GpuSim::new(world, &vehicle, &sensors, capacity))
    }

    /// `advance_generation` on the GPU.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn advance_generation_gpu(
        &mut self,
        sim: &mut crate::gpu::simulation::GpuSim,
        world: &crate::gpu::hip::GpuWorld,
        time_limit_ticks: u64,
    ) -> Result<u64, String> {
        assert_eq!(
            self.stats_phase, 0,
            "fresh game generations use reset statistics counters"
        );
        let bound = (time_limit_ticks / 6 + 2) * 6;
        self.advance_window_gpu(
            sim,
            world,
            bound.saturating_sub(self.tick),
            true,
            Some(time_limit_ticks as f64 / 60.0),
        )
    }

    /// `advance_window` on the GPU: agents and cars are uploaded, advanced in
    /// the tick-major order and read back. Networks are uploaded when
    /// `sim.network_tag` differs from the generation.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn advance_window_gpu(
        &mut self,
        sim: &mut crate::gpu::simulation::GpuSim,
        world: &crate::gpu::hip::GpuWorld,
        ticks: u64,
        stop_when_inactive: bool,
        time_limit: Option<f64>,
    ) -> Result<u64, String> {
        use crate::gpu::simulation::*;
        let mut profile = crate::training::training_profile::Profile::new("gpu_window");
        if ticks == 0 {
            return Ok(0);
        }
        if self.user_paused || self.physics.is_some() {
            return Err("the GPU runner does not model paused or native-broadphase windows".into());
        }
        if self.agents.is_empty() {
            return Err("no agents".into());
        }
        let surfaces = &world.arrays.surfaces;
        let vehicle = &self.world.vehicle;
        for agent in &mut self.agents {
            agent.deactivated_at = None;
        }
        if sim.network_tag != Some(self.generation) || sim.network_count() != self.agents.len() {
            let src = std::array::from_fn(|c| {
                self.outputs
                    .iter()
                    .rposition(|&slot| slot as usize == c)
                    .map_or(-1, |j| j as i32)
            });
            // Agents own the networks; the GPU uploader reuses its staging
            // allocation while copying them without cloning parameter vectors.
            sim.upload_agent_networks(&self.agents, src)?;
            profile.mark("networks");
            sim.network_tag = Some(self.generation);
        }
        // Retained staging buffers: filling them in place avoids allocating,
        // page-faulting and concatenating population-sized vectors per window.
        let (mut cars, mut agents) = sim.take_state_buffers();
        export_state_into(&self.agents, vehicle, surfaces, &mut cars, &mut agents)?;
        profile.mark("export_state");
        sim.upload(&cars, Some(&agents));
        profile.mark("upload_state");
        let args = WindowArgs {
            start_tick: self.tick,
            ticks,
            start_batch: self.batch_index as u32,
            batches_per_tick: self.batches_per_tick() as u32,
            stats_phase: self.stats_phase as u32,
            stop_when_inactive: stop_when_inactive as u32,
            eliminate_on_wall: self.eliminate_on_wall as u32,
            eliminate_when_idle: self.eliminate_when_idle as u32,
            has_time_limit: time_limit.is_some() as u32,
            pad: 0,
            time_limit: time_limit.unwrap_or(0.0),
        };
        let (executed, transition_without_drive) = sim.window(&args);
        profile.mark("window");
        // The upload has finished. Read back into the same allocations.
        sim.download(Some(&mut cars), Some(&mut agents));
        profile.mark("download_state");
        self.import_state(&cars, &agents, surfaces)?;
        profile.mark("import_state");
        sim.return_state_buffers(cars, agents);
        let bpt = self.batches_per_tick();
        self.tick += executed;
        self.batch_index = (self.batch_index + ((executed as usize - transition_without_drive as usize) % 8) * bpt) % 8;
        Ok(executed)
    }

    /// The rest of `gpu_import` for the whole population: `cars` and `agents`
    /// hold what `advance_window_gpu` read back.
    pub(crate) fn import_state(
        &mut self,
        cars: &[crate::gpu::simulation::GpuCar],
        agents: &[crate::gpu::simulation::GpuAgent],
        surfaces: &crate::gpu::simulation::SurfaceTable,
    ) -> Result<(), String> {
        self.agents
            .par_iter_mut()
            .zip(cars)
            .zip(agents)
            .try_for_each(|((a, c), g)| {
                a.car.gpu_import(c, surfaces)?;
                crate::gpu::simulation::agent_import(g, a);
                Ok::<(), String>(())
            })
    }

    /// Reproduce with the same RNG stream, then compute population novelty on
    /// the GPU. Uploaded offspring stay in place for the next driving window,
    /// and the new agents take over the offspring networks without a copy
    /// (read them from `agents`; only the selection summary is returned).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn next_generation_gpu(&mut self, sim: &mut crate::gpu::simulation::GpuSim) -> Result<Turnover, String> {
        self.next_generation_gpu_scored(sim, None)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn next_generation_gpu_with_fitness(
        &mut self,
        sim: &mut crate::gpu::simulation::GpuSim,
        scores: Vec<f64>,
    ) -> Result<Turnover, String> {
        self.next_generation_gpu_scored(sim, Some(scores))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn next_generation_gpu_scored(
        &mut self,
        sim: &mut crate::gpu::simulation::GpuSim,
        scores: Option<Vec<f64>>,
    ) -> Result<Turnover, String> {
        let mut profile = crate::training::training_profile::Profile::new("gpu_turnover");
        // The preceding upload is complete, so its host buffer can hold noise
        // until reproduction finishes. Refilling it then uploads the offspring.
        let mut scratch = sim.take_parameter_buffer();
        let results: Vec<AgentResult> = self.agents.iter().map(TrainingAgent::result).collect();
        let Generation {
            networks,
            preserved_count,
            rewards,
        } = match scores {
            Some(scores) => self
                .rng
                .reproduce_scored(&results, scores, &self.settings, &mut scratch),
            None => self.rng.reproduce_with_scratch(&results, &self.settings, &mut scratch),
        };
        sim.return_parameter_buffer(scratch);
        profile.mark("reproduce");
        let src = std::array::from_fn(|c| {
            self.outputs
                .iter()
                .rposition(|&slot| slot as usize == c)
                .map_or(-1, |j| j as i32)
        });
        sim.upload_networks(&networks, src)?;
        profile.mark("networks");
        let novelty = sim.novelty();
        profile.mark("novelty");
        self.install_with_novelty(networks, true, Some(&novelty));
        self.stats_phase = 0;
        self.generation += 1;
        sim.network_tag = Some(self.generation);
        profile.mark("install");
        Ok(Turnover {
            preserved_count,
            rewards,
        })
    }
}
