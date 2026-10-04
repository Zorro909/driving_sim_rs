//! Process statistics included in benchmark and training reports.

use serde_json::{json, Value};

pub(super) fn process_cpu_seconds() -> f64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

pub(super) fn peak_rss_kib() -> i64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    usage.ru_maxrss
}

pub(super) fn platform_json(threads: usize) -> Value {
    let logical = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
    json!({
        "implementation": format!("rust altd_sim {}", env!("CARGO_PKG_VERSION")),
        "platform": format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        "logical_cpus": logical,
        "affinity_cpus": std::thread::available_parallelism().map(|n| n.get()).ok(),
        "threads": threads,
    })
}
