//! Process statistics included in benchmark and training reports.

use serde_json::{json, Value};

/// `altd-sim --version`: the package version, or the CI build version of nightly packages.
pub(super) const VERSION: &str = match option_env!("ALTD_BUILD_VERSION") {
    Some(version) => version,
    None => env!("CARGO_PKG_VERSION"),
};

#[cfg(unix)]
pub(super) fn process_cpu_seconds() -> f64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

#[cfg(unix)]
pub(super) fn peak_rss_kib() -> i64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    // macOS reports bytes, Linux KiB.
    if cfg!(target_os = "macos") {
        usage.ru_maxrss / 1024
    } else {
        usage.ru_maxrss
    }
}

#[cfg(unix)]
fn logical_cpus() -> i64 {
    unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) }
}

#[cfg(windows)]
pub(super) fn process_cpu_seconds() -> f64 {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    let mut times = [FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    }; 4];
    let [created, exited, kernel, user] = &mut times;
    unsafe { GetProcessTimes(GetCurrentProcess(), created, exited, kernel, user) };
    // FILETIME counts 100 ns intervals.
    let ticks = |t: &FILETIME| ((t.dwHighDateTime as u64) << 32 | t.dwLowDateTime as u64) as f64;
    (ticks(kernel) + ticks(user)) * 1e-7
}

#[cfg(windows)]
pub(super) fn peak_rss_kib() -> i64 {
    use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, size) };
    (counters.PeakWorkingSetSize / 1024) as i64
}

#[cfg(windows)]
fn logical_cpus() -> i64 {
    use windows_sys::Win32::System::Threading::{GetActiveProcessorCount, ALL_PROCESSOR_GROUPS};
    unsafe { GetActiveProcessorCount(ALL_PROCESSOR_GROUPS) as i64 }
}

pub(super) fn platform_json(threads: usize) -> Value {
    json!({
        "implementation": format!("rust altd_sim {VERSION}"),
        "platform": format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        "logical_cpus": logical_cpus(),
        "affinity_cpus": std::thread::available_parallelism().map(|n| n.get()).ok(),
        "threads": threads,
    })
}
