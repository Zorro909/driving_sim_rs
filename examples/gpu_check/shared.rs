//! Deterministic inputs and mismatch reporting shared by the GPU checks.
pub(super) struct Rng(pub(super) u64);
impl Rng {
    pub(super) fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub(super) fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub(super) fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

#[derive(Default)]
pub(super) struct Tally {
    checked: usize,
    flagged: usize,
    pub(super) mismatches: usize,
}
impl Tally {
    pub(super) fn record(&mut self, err: u32, same: bool, describe: impl FnOnce() -> String) {
        if err != 0 {
            self.flagged += 1;
            return;
        }
        self.checked += 1;
        if !same {
            if self.mismatches < 8 {
                eprintln!("    mismatch: {}", describe());
            }
            self.mismatches += 1;
        }
    }
    pub(super) fn report(&self, name: &str) -> bool {
        println!(
            "  {name:<20} {:>9} checked  {:>7} flagged  {:>6} mismatches  {}",
            self.checked,
            self.flagged,
            self.mismatches,
            if self.mismatches == 0 { "OK" } else { "FAIL" }
        );
        self.mismatches == 0
    }
}

pub(super) fn ok_line(ok: bool, note: &str) -> bool {
    println!("    {note}");
    ok
}
