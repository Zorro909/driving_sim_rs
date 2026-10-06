//! Bitwise differential checks of the GPU simulator parts (gpu/) against the CPU.
//! Usage: gpu_check [math] ...   (build the library first with gpu/build.sh)
use altd_sim::gpu::hip::Gpu;

mod benchmarks;
mod evolution;
mod fixtures;
mod math;
mod rays;
mod shared;
mod simulation;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help") {
        println!("usage: gpu_check [math|rays|sensors|infer|step|stats|window|schedules|reuse|novelty|turnover|benchfixed|benchtrain] ...");
        println!("Defaults use generated rally geometry and deterministic Xavier networks. Overrides: ALTD_GPU_SCENE, ALTD_GPU_SPAWN, ALTD_GPU_NETWORK, ALTD_GPU_MODEL, ALTD_GPU_CKPT, ALTD_GPU_LIB.");
        return;
    }
    let parts: Vec<&str> = if args.is_empty() {
        vec!["math"]
    } else {
        args.iter().map(String::as_str).collect()
    };
    // ALTD_LIBM=windows checks the Windows UCRT math on the CPU and the GPU.
    if let Some(libm) = altd_sim::math::libm_from_env().unwrap_or_else(|e| panic!("{e}")) {
        altd_sim::math::set_libm(libm).unwrap_or_else(|e| panic!("{e}"));
    }
    let gpu = Gpu::open(None).unwrap_or_else(|e| panic!("{e}"));
    let mut ok = true;
    for part in parts {
        ok &= match part {
            "math" => math::check_math(&gpu, 1 << 22, true),
            "rays" => rays::check_rays(&gpu, &fixtures::load_world(), 1 << 20),
            "sensors" => simulation::check_sensors(&gpu, 2048, 36, 150),
            "infer" => simulation::check_infer(&gpu, 2048, 36, 150),
            "step" => simulation::check_step(&gpu, 2048, 120, 400),
            "stats" => simulation::check_stats(&gpu, 2048, 3600),
            "window" => simulation::check_window(&gpu, 2048, 6000, 2, &[(true, false), (false, true)]),
            "bench" => benchmarks::bench_window(&gpu, 32768, 6000, false),
            "benchlive" => benchmarks::bench_window(&gpu, 32768, 6000, true),
            "bench2k" => benchmarks::bench_window(&gpu, 2048, 6000, false),
            "benchfixed" => benchmarks::bench_fixed(&gpu),
            "benchtrain" => benchmarks::bench_training(&gpu),
            "schedules" => evolution::check_schedules(&gpu),
            "reuse" => evolution::check_graph_reuse(&gpu),
            "novelty" => evolution::check_novelty(&gpu),
            "turnover" => evolution::check_turnover(&gpu),
            "window3" => simulation::check_window(&gpu, 2048, 6000, 2, &[(true, true)]),
            "settle" => evolution::settle_report(&gpu, 32768),
            other => panic!("unknown part {other}"),
        };
    }
    println!("{}", if ok { "ALL OK" } else { "FAILURES" });
    std::process::exit(if ok { 0 } else { 1 });
}
