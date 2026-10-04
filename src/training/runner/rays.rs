//! Tick-by-tick windows with externally supplied ray hits.

#[cfg(any(test, all(target_arch = "wasm32", feature = "wasm")))]
use super::RayWindow;
use super::TrainingRunner;
#[cfg(any(test, all(target_arch = "wasm32", feature = "wasm")))]
use crate::physics::car::Sensor;
#[cfg(any(test, all(target_arch = "wasm32", feature = "wasm")))]
use rayon::prelude::*;

impl TrainingRunner {
    /// Starts a `RayWindow` of up to `ticks` ticks (`advance_window` with an
    /// external raycaster). Paused runners, native shared broadphase scenes,
    /// and ray sensors without a BSP ray tree are not supported.
    #[cfg(any(test, all(target_arch = "wasm32", feature = "wasm")))]
    pub(crate) fn begin_ray_window(
        &mut self,
        ticks: u64,
        stop_when_inactive: bool,
        time_limit: Option<f64>,
    ) -> Result<RayWindow, String> {
        if self.user_paused || self.physics.is_some() {
            return Err("external raycasting does not model paused or native-broadphase windows".into());
        }
        if self.layout.ray_count() > 0 && self.world.track.ray_tree().is_none() {
            return Err("the scene has no BSP raycaster; the ray sensors need its RayTree".into());
        }
        for agent in &mut self.agents {
            agent.deactivated_at = None;
        }
        Ok(RayWindow {
            start_tick: self.tick,
            start_batch: self.batch_index,
            ticks,
            stop_when_inactive,
            time_limit,
            executed: 0,
            finished: false,
            transition_without_drive: false,
            batch_index: self.batch_index,
            inferring: Vec::new(),
        })
    }

    /// Runs the statistics of the next tick and selects the agents that infer
    /// in it. Returns false once the window is complete or stops before
    /// driving (every car inactive, or the time limit reached).
    #[cfg(any(test, all(target_arch = "wasm32", feature = "wasm")))]
    pub(crate) fn prepare_ray_tick(&mut self, w: &mut RayWindow) -> bool {
        if w.finished || w.executed >= w.ticks {
            w.finished = true;
            return false;
        }
        w.executed += 1;
        let tick = w.tick();
        w.batch_index = (w.start_batch + ((w.executed - 1) as usize % 8) * self.batches_per_tick()) % 8;
        let mut agents = std::mem::take(&mut self.agents);
        let stop = {
            let schedule = self.schedule_for(agents.len());
            agents
                .par_iter_mut()
                .for_each(|agent| schedule.update_stats(agent, tick));
            let stop = tick % 6 == self.stats_phase && {
                let first = if self.stats_phase == 0 { 6 } else { self.stats_phase };
                let elapsed = ((tick - first) / 6) as f64 * 0.1;
                (w.stop_when_inactive && agents.iter().all(|a| !a.car.active))
                    || w.time_limit.is_some_and(|limit| elapsed >= limit)
            };
            if !stop {
                w.inferring.clear();
                w.inferring.extend(
                    agents
                        .iter()
                        .enumerate()
                        .filter(|(i, a)| schedule.infers(*i, w.batch_index) && a.car.active)
                        .map(|(i, _)| i),
                );
            }
            stop
        };
        self.agents = agents;
        if stop {
            w.transition_without_drive = true;
            w.finished = true;
        }
        !stop
    }

    /// The ray sensor segments of the inferring agents, `[start.x, start.y,
    /// end.x, end.y]` in float32 as `RayTree::raycast` takes them: for each
    /// agent of `w.inferring`, its `Sensor::Raycast` entries in layout order.
    #[cfg(any(test, all(target_arch = "wasm32", feature = "wasm")))]
    pub(crate) fn ray_queries(&self, w: &RayWindow, out: &mut Vec<[f32; 4]>) {
        use crate::math::godot_math::F2;
        out.clear();
        for &i in &w.inferring {
            let car = &self.agents[i].car;
            let start = F2::from(car.position);
            for sensor in &self.layout.sensors {
                if let Sensor::Raycast { degrees, length } = *sensor {
                    let end = F2::from(car.ray_end(degrees, length).0);
                    out.push([start.x, start.y, end.x, end.y]);
                }
            }
        }
    }

    /// Sets the inferring agents' controls from `hits` (one per query of
    /// `ray_queries`, `[hit.x, hit.y, hit != 0, _]`), then drives every car.
    #[cfg(any(test, all(target_arch = "wasm32", feature = "wasm")))]
    pub(crate) fn finish_ray_tick(&mut self, w: &RayWindow, hits: &[[f32; 4]]) -> Result<(), String> {
        let rays = self.layout.ray_count();
        if hits.len() != w.inferring.len() * rays {
            return Err(format!(
                "expected {} ray hits, got {}",
                w.inferring.len() * rays,
                hits.len()
            ));
        }
        let tick = w.tick();
        let mut agents = std::mem::take(&mut self.agents);
        {
            let schedule = self.schedule_for(agents.len());
            for (&i, hits) in w.inferring.iter().zip(hits.chunks(rays.max(1))) {
                let agent = &mut agents[i];
                let scratch = &mut agent.scratch;
                schedule.layout.read_into_with_hits(
                    schedule.world,
                    &agent.car,
                    &mut scratch.sensors,
                    &mut scratch.inputs,
                    hits,
                );
                schedule.set_controls(agent);
            }
            agents.par_iter_mut().for_each(|agent| schedule.step_agent(agent, tick));
        }
        self.agents = agents;
        Ok(())
    }

    /// Commits the window: `tick` and the inference cursor advance as after
    /// `advance`. Returns the executed ticks.
    #[cfg(any(test, all(target_arch = "wasm32", feature = "wasm")))]
    pub(crate) fn end_ray_window(&mut self, w: RayWindow) -> u64 {
        let bpt = self.batches_per_tick();
        self.tick = w.start_tick + w.executed;
        self.batch_index =
            (w.start_batch + ((w.executed as usize - w.transition_without_drive as usize) % 8) * bpt) % 8;
        w.executed
    }

    /// `CAR_STATE_STRIDE` values per agent (`CAR_STATE_FIELDS`), appended to `out`.
    pub(crate) fn car_states(&self, out: &mut Vec<f64>) {
        let steering = self.world.vehicle.wheels.iter().position(|w| w.steering);
        for a in &self.agents {
            let c = &a.car;
            out.extend_from_slice(&[
                c.position.x,
                c.position.y,
                c.rotation,
                c.velocity.x,
                c.velocity.y,
                c.angular_velocity,
                c.active as u8 as f64,
                a.stats.total_score,
                a.stats.score.lap_count as f64,
                a.stats.best_lap_time.unwrap_or(-1.0),
                c.collision_count as f64,
                c.boost_energy,
                steering.map_or(0.0, |i| c.wheels[i].angle_deg),
                a.stats.update_count as f64,
            ]);
        }
    }
}
