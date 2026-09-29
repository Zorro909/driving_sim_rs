//! Optional timings for population turnover and GPU transfers.
use std::sync::OnceLock;
use std::time::Instant;

pub(crate) struct Profile {
    name: &'static str,
    previous: Option<Instant>,
    phases: Vec<(&'static str, f64)>,
}

impl Profile {
    pub(crate) fn new(name: &'static str) -> Self {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        let enabled = *ENABLED.get_or_init(|| std::env::var("ALTD_TRAIN_PROFILE").is_ok_and(|v| v == "1"));
        Self { name, previous: enabled.then(Instant::now), phases: Vec::new() }
    }

    pub(crate) fn mark(&mut self, name: &'static str) {
        if let Some(previous) = &mut self.previous {
            let now = Instant::now();
            self.phases.push((name, now.duration_since(*previous).as_secs_f64()));
            *previous = now;
        }
    }
}

impl Drop for Profile {
    fn drop(&mut self) {
        if self.previous.is_some() {
            eprintln!("altd_train_profile {}", serde_json::json!({"stage": self.name, "seconds": self.phases}));
        }
    }
}
