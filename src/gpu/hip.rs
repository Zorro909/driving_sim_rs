//! Runtime loader for the HIP simulator parts in gpu/ (libaltd_gpu.so).
//! Loaded at runtime so the crate builds and runs without ROCm.
use std::ffi::{c_void, CStr};
use std::path::{Path, PathBuf};

/// Library file name, both in release archives and in gpu/build.sh output.
pub fn library_name() -> String {
    format!(
        "{}altd_gpu{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    )
}

/// Paths [`Gpu::open`] tries, in order. An explicit path or ALTD_GPU_LIB is the
/// only candidate; otherwise the library beside the executable (release
/// archives), then gpu/build.sh output in this source checkout.
pub fn library_candidates(explicit: Option<&Path>, env: Option<PathBuf>, exe_dir: Option<&Path>) -> Vec<PathBuf> {
    if let Some(path) = explicit.map(Path::to_path_buf).or(env) {
        return vec![path];
    }
    let name = library_name();
    let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/gpu").join(&name);
    exe_dir
        .map(|dir| dir.join(&name))
        .into_iter()
        .chain([checkout])
        .collect()
}

/// Explains a HIP status returned by the library.
pub(crate) fn status_message(status: i32) -> String {
    let hint = match status {
        35 => " (hipErrorInsufficientDriver: the GPU driver is missing or older than the runtime)",
        100 => " (hipErrorNoDevice: no supported GPU; check `rocminfo` and /dev/kfd permissions on AMD, `nvidia-smi` on NVIDIA)",
        209 => {
            " (hipErrorNoBinaryForGpu: the library has no kernels for this GPU; \
             rebuild it with GPU_ARCH set to the gfx name `rocminfo` reports, or CUDA_ARCH set to your sm_ target on NVIDIA)"
        }
        _ => "",
    };
    format!("HIP error {status}{hint}")
}

/// Math primitive selectors of `altd_gpu_math` (gpu/sim/altd_gpu.hip).
#[repr(i32)]
#[derive(Clone, Copy, Debug)]
pub enum MathOp {
    NativeSin = 0,
    NativeCos = 1,
    NativeAtan2 = 2,
    ManagedSinCos = 3,
    Exp = 4,
    Pow = 5,
    EngineSinCos = 6,
    GameTanh = 8,
}

pub struct Gpu {
    library: libloading::Library,
    path: PathBuf,
}

impl Gpu {
    /// Opens the first loadable path of [`library_candidates`] for `path`,
    /// ALTD_GPU_LIB, and the running executable.
    pub fn open(path: Option<&Path>) -> Result<Gpu, String> {
        let env = std::env::var_os("ALTD_GPU_LIB").map(PathBuf::from);
        let exe = std::env::current_exe().ok();
        let candidates = library_candidates(path, env, exe.as_deref().and_then(Path::parent));
        let mut errors = Vec::new();
        for candidate in candidates {
            match unsafe { load(&candidate) } {
                Ok(library) => {
                    return Ok(Gpu {
                        library,
                        path: candidate,
                    })
                }
                Err(error) => {
                    // libloading keeps the dlerror/LoadLibrary text in `source`.
                    let detail = std::error::Error::source(&error).map_or(error.to_string(), ToString::to_string);
                    let path = candidate.display().to_string();
                    let mut message = if detail.contains(&path) {
                        detail
                    } else {
                        format!("{path}: {detail}")
                    };
                    if message.contains("amdhip64") {
                        message += " (install a ROCm 7.x runtime that provides libamdhip64)";
                    }
                    errors.push(message);
                }
            }
        }
        Err(format!(
            "cannot load the GPU simulator library; build it with gpu/build.sh (AMD) or gpu/build-cuda.ps1 / gpu/build-cuda.sh (NVIDIA), or set ALTD_GPU_LIB\n  {}",
            errors.join("\n  ")
        ))
    }

    /// Path of the loaded library.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Function pointer `name` of type `F` (an `unsafe extern "C" fn`).
    pub(crate) fn symbol<F: Copy>(&self, name: &str) -> F {
        self.try_symbol(name).unwrap_or_else(|| {
            panic!(
                "{} lacks {name}; rebuild it with gpu/build.sh or gpu/build-cuda.*",
                self.path.display()
            )
        })
    }

    /// `symbol`, or `None` when the library lacks `name`.
    pub(crate) fn try_symbol<F: Copy>(&self, name: &str) -> Option<F> {
        assert_eq!(std::mem::size_of::<F>(), std::mem::size_of::<*mut c_void>());
        unsafe { self.library.get::<F>(name.as_bytes()) }.ok().map(|s| *s)
    }

    /// A required function pointer, reporting incompatible libraries without panicking.
    pub(crate) fn required_symbol<F: Copy>(&self, name: &str) -> Result<F, String> {
        self.try_symbol(name).ok_or_else(|| {
            format!(
                "{} lacks {name}; rebuild it with gpu/build.sh or gpu/build-cuda.*",
                self.path.display()
            )
        })
    }

    /// The runtime the library was built for: "HIP" (AMD) or "CUDA" (NVIDIA).
    /// Libraries without `altd_gpu_platform` predate the CUDA build and are HIP.
    pub fn platform(&self) -> String {
        let f: Option<unsafe extern "C" fn() -> *const std::ffi::c_char> = self.try_symbol("altd_gpu_platform");
        f.map_or_else(
            || "HIP".to_string(),
            |f| unsafe { CStr::from_ptr(f()) }.to_string_lossy().into_owned(),
        )
    }

    /// The device name, or the library path for older libraries. The legacy
    /// math entry point probes device availability when no name is exported.
    pub fn device_name(&self) -> Result<String, String> {
        let f: Option<unsafe extern "C" fn(*mut std::ffi::c_char, i32) -> i32> =
            self.try_symbol("altd_gpu_device_name");
        let Some(f) = f else {
            self.probe_kernels()?;
            return Ok(self.path.display().to_string());
        };
        let mut name = [0 as std::ffi::c_char; 256];
        let status = unsafe { f(name.as_mut_ptr(), name.len() as i32) };
        if status != 0 {
            return Err(format!("no usable HIP device ({})", status_message(status)));
        }
        Ok(unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned())
    }

    /// Runs one small kernel. A device can be present while the library has no
    /// code for it (built for another GPU_ARCH or CUDA_ARCH); this reports that
    /// before a simulation needs the device.
    pub fn probe_kernels(&self) -> Result<(), String> {
        // Asks before launching where the library can: a ROCm 7.1 launch without
        // kernels for the device crashes the process.
        let check: Option<unsafe extern "C" fn() -> i32> = self.try_symbol("altd_gpu_probe");
        if let Some(check) = check {
            let status = unsafe { check() };
            if status != 0 {
                return Err(format!("no usable HIP device ({})", status_message(status)));
            }
        }
        let probe: unsafe extern "C" fn(i32, i32, *const c_void, *mut c_void, *mut u32) -> i32 = self
            .try_symbol("altd_gpu_math")
            .ok_or("the HIP library lacks the device probe entry points")?;
        let input = 0.0f32;
        let mut output = 0.0f32;
        let mut error = 0u32;
        let status = unsafe {
            probe(
                MathOp::NativeSin as i32,
                1,
                std::ptr::from_ref(&input).cast(),
                std::ptr::from_mut(&mut output).cast(),
                &mut error,
            )
        };
        if status != 0 {
            return Err(format!("no usable HIP device ({})", status_message(status)));
        }
        Ok(())
    }

    pub(crate) fn check(status: i32, what: &str) {
        assert_eq!(status, 0, "GPU {what} failed with {}", status_message(status));
    }

    /// Panics unless the library's device structs match the Rust mirrors.
    /// Needs no GPU: the library only reports host-side sizes and offsets.
    pub fn check_layout(&self) {
        crate::gpu::simulation::check_layout(self);
    }

    /// Runs one math primitive. `input` and `output` hold `n` elements of the
    /// op's input and output layout (see gpu/sim/altd_gpu.hip); returns the error bits.
    pub fn math<I: Copy, O: Copy>(&self, op: MathOp, input: &[I], output: &mut [O]) -> Vec<u32> {
        assert_eq!(input.len(), output.len());
        let f: unsafe extern "C" fn(i32, i32, *const c_void, *mut c_void, *mut u32) -> i32 =
            self.symbol("altd_gpu_math");
        let mut err = vec![0u32; input.len()];
        let n = i32::try_from(input.len()).expect("batch too large");
        Self::check(
            unsafe {
                f(
                    op as i32,
                    n,
                    input.as_ptr().cast(),
                    output.as_mut_ptr().cast(),
                    err.as_mut_ptr(),
                )
            },
            "math",
        );
        err
    }

    /// Raw engine sine and cosine bits for the floats with bits `first..first + n`.
    pub fn engine_scan(&self, first: u32, out: &mut [[u32; 2]]) {
        let f: unsafe extern "C" fn(u32, u32, *mut [u32; 2]) -> i32 = self.symbol("altd_gpu_engine_scan");
        Self::check(unsafe { f(first, out.len() as u32, out.as_mut_ptr()) }, "engine scan");
    }
}

/// Opens `path`; on Unix with RTLD_NOW, so unresolved symbols fail at load time.
unsafe fn load(path: &Path) -> Result<libloading::Library, libloading::Error> {
    #[cfg(unix)]
    {
        use libloading::os::unix::{Library, RTLD_LOCAL, RTLD_NOW};
        unsafe { Library::open(Some(path), RTLD_NOW | RTLD_LOCAL) }.map(Into::into)
    }
    #[cfg(not(unix))]
    unsafe {
        libloading::Library::new(path)
    }
}

pub(crate) use crate::gpu::simulation::RayNode;

/// Query selectors of `altd_gpu_query`.
#[repr(i32)]
#[derive(Clone, Copy, Debug)]
pub enum Query {
    /// `(sx, sy, ex, ey)` -> `(hx, hy, hit, 0)`
    Raycast = 0,
    /// `(px, py, _, _)` -> `(cx, cy, 1, 0)`
    ClosestWall = 1,
}

/// Device copy of a `World`.
pub struct GpuWorld<'a> {
    gpu: &'a Gpu,
    handle: *mut c_void,
    free: unsafe extern "C" fn(*mut c_void),
    /// The exported track (surface table, nearest-segment grids).
    pub arrays: crate::gpu::simulation::TrackArrays,
}

impl Drop for GpuWorld<'_> {
    fn drop(&mut self) {
        unsafe { (self.free)(self.handle) };
    }
}

/// Owned host track arrays. Preparing these calls no GPU APIs and can run on
/// the CPU producer thread before a device is opened or a track is uploaded.
pub struct PreparedGpuWorld {
    arrays: crate::gpu::simulation::TrackArrays,
    rays: (Vec<RayNode>, Vec<[f32; 4]>, f32, usize),
}

impl PreparedGpuWorld {
    pub(crate) fn new(world: &crate::track::world::World) -> Result<Self, String> {
        let track = &world.track;
        if track.raycaster_present() == Some(false) {
            return Err("GPU port expects a BSP raycaster".into());
        }
        let (nodes, walls, magnitude, depth) = track.ray_tree().ok_or("GPU port needs the BSP RayTree")?.gpu_arrays();
        let nodes = nodes
            .into_iter()
            .map(|(links, own, subtree)| RayNode { links, own, subtree })
            .collect();
        Ok(Self {
            arrays: crate::gpu::simulation::TrackArrays::new(world)?,
            rays: (nodes, walls, magnitude, depth),
        })
    }
}

impl<'a> GpuWorld<'a> {
    pub fn new(gpu: &'a Gpu, world: &crate::track::world::World) -> GpuWorld<'a> {
        Self::from_prepared(
            gpu,
            PreparedGpuWorld::new(world).unwrap_or_else(|e| panic!("GPU track export: {e}")),
        )
    }

    /// Upload CPU-prepared geometry without rebuilding spatial query arrays.
    pub fn from_prepared(gpu: &'a Gpu, prepared: PreparedGpuWorld) -> GpuWorld<'a> {
        Self::try_from_prepared(gpu, prepared).unwrap_or_else(|e| panic!("{e}"))
    }

    /// `from_prepared`, returning unsupported geometry, missing symbols and upload failures.
    pub fn try_from_prepared(gpu: &'a Gpu, prepared: PreparedGpuWorld) -> Result<GpuWorld<'a>, String> {
        crate::gpu::simulation::layout_matches(gpu)?;
        let PreparedGpuWorld { arrays, rays } = prepared;
        arrays.validate_counts(&rays)?;
        let desc = arrays.desc(&rays);
        let create: unsafe extern "C" fn(*const crate::gpu::simulation::WorldDesc) -> *mut c_void =
            gpu.required_symbol("altd_gpu_world_create")?;
        let free = gpu.required_symbol("altd_gpu_world_free")?;
        // The library copies every array; `rays` only has to outlive the call.
        let handle = unsafe { create(&desc) };
        if handle.is_null() {
            return Err("altd_gpu_world_create failed to upload the track".into());
        }
        Ok(GpuWorld {
            gpu,
            handle,
            free,
            arrays,
        })
    }

    pub(crate) fn gpu(&self) -> &'a Gpu {
        self.gpu
    }

    pub(crate) fn handle(&self) -> *const c_void {
        self.handle
    }

    /// Runs one BSP query per element of `input`.
    pub fn query(&self, op: Query, input: &[[f32; 4]]) -> Vec<[f32; 4]> {
        let f: unsafe extern "C" fn(*mut c_void, i32, i32, *const [f32; 4], *mut [f32; 4]) -> i32 =
            self.gpu.symbol("altd_gpu_query");
        let mut out = vec![[0f32; 4]; input.len()];
        let n = i32::try_from(input.len()).expect("batch too large");
        Gpu::check(
            unsafe { f(self.handle, op as i32, n, input.as_ptr(), out.as_mut_ptr()) },
            "query",
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_library_paths_are_the_only_candidate() {
        let exe = Path::new("/opt/altd");
        let explicit = library_candidates(Some(Path::new("a.so")), Some("b.so".into()), Some(exe));
        assert_eq!(explicit, [PathBuf::from("a.so")]);
        assert_eq!(
            library_candidates(None, Some("b.so".into()), Some(exe)),
            [PathBuf::from("b.so")]
        );
    }

    #[test]
    fn default_library_search_prefers_the_executable_directory() {
        let name = library_name();
        let checkout = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/gpu").join(&name);
        let found = library_candidates(None, None, Some(Path::new("/opt/altd")));
        assert_eq!(found, [Path::new("/opt/altd").join(&name), checkout.clone()]);
        assert_eq!(library_candidates(None, None, None), [checkout]);
    }

    #[test]
    fn missing_libraries_list_every_attempt() {
        let error = Gpu::open(Some(Path::new("/nonexistent/libaltd_gpu.so"))).err().unwrap();
        assert!(error.contains("/nonexistent/libaltd_gpu.so"), "{error}");
        assert!(error.contains("gpu/build.sh"), "{error}");
    }
}
