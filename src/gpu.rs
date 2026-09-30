//! Runtime loader for the HIP simulator parts in gpu/ (libaltd_gpu.so).
//! Loaded with dlopen so the crate builds and runs without ROCm.
use std::ffi::{c_void, CStr, CString};

/// Default library location written by gpu/build.sh; override with ALTD_GPU_LIB.
pub const DEFAULT_LIBRARY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/target/gpu/libaltd_gpu.so");

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
    EngineRaw = 7,
    GameTanh = 8,
}

/// GPU error bits, as in gpu/sim/math.h.
pub const ERR_MANAGED_LARGE: u32 = 1;
pub const ERR_NATIVE_LARGE: u32 = 2;
pub const ERR_ENGINE_DOMAIN: u32 = 4;

pub struct Gpu {
    handle: *mut c_void,
}

// The library has no thread-affine state beyond the HIP runtime's own locking.
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe { libc::dlclose(self.handle) };
    }
}

impl Gpu {
    /// Opens `path`, or ALTD_GPU_LIB, or `DEFAULT_LIBRARY`.
    pub fn open(path: Option<&str>) -> Result<Gpu, String> {
        let env = std::env::var("ALTD_GPU_LIB").ok();
        let path = path.or(env.as_deref()).unwrap_or(DEFAULT_LIBRARY);
        let c_path = CString::new(path).map_err(|e| e.to_string())?;
        let handle = unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            let message = unsafe { CStr::from_ptr(libc::dlerror()) }.to_string_lossy().into_owned();
            return Err(format!("{path}: {message} (build it with gpu/build.sh)"));
        }
        Ok(Gpu { handle })
    }

    /// Function pointer `name` of type `F` (an `unsafe extern "C" fn`).
    pub fn symbol<F: Copy>(&self, name: &str) -> F {
        assert_eq!(std::mem::size_of::<F>(), std::mem::size_of::<*mut c_void>());
        let c_name = CString::new(name).unwrap();
        let pointer = unsafe { libc::dlsym(self.handle, c_name.as_ptr()) };
        assert!(!pointer.is_null(), "libaltd_gpu.so lacks {name}; rebuild it with gpu/build.sh");
        unsafe { std::mem::transmute_copy(&pointer) }
    }

    pub(crate) fn check(status: i32, what: &str) {
        assert_eq!(status, 0, "GPU {what} failed with HIP error {status}");
    }

    /// Runs one math primitive. `input` and `output` hold `n` elements of the
    /// op's input and output layout (see gpu/sim/altd_gpu.hip); returns the error bits.
    pub fn math<I: Copy, O: Copy>(&self, op: MathOp, input: &[I], output: &mut [O]) -> Vec<u32> {
        assert_eq!(input.len(), output.len());
        let f: unsafe extern "C" fn(i32, i32, *const c_void, *mut c_void, *mut u32) -> i32 = self.symbol("altd_gpu_math");
        let mut err = vec![0u32; input.len()];
        let n = i32::try_from(input.len()).expect("batch too large");
        Self::check(unsafe { f(op as i32, n, input.as_ptr().cast(), output.as_mut_ptr().cast(), err.as_mut_ptr()) }, "math");
        err
    }

    /// Raw engine sine and cosine bits for the floats with bits `first..first + n`.
    pub fn engine_scan(&self, first: u32, out: &mut [[u32; 2]]) {
        let f: unsafe extern "C" fn(u32, u32, *mut [u32; 2]) -> i32 = self.symbol("altd_gpu_engine_scan");
        Self::check(unsafe { f(first, out.len() as u32, out.as_mut_ptr()) }, "engine scan");
    }
}

/// `RayNode` of gpu/sim/world.h.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct RayNode {
    pub links: [u32; 4],
    pub own: [f32; 4],
    pub subtree: [f32; 4],
}

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
    /// The exported track (surface table, nearest-segment grids).
    pub arrays: crate::gpu_sim::TrackArrays,
}

impl Drop for GpuWorld<'_> {
    fn drop(&mut self) {
        let free: unsafe extern "C" fn(*mut c_void) = self.gpu.symbol("altd_gpu_world_free");
        unsafe { free(self.handle) };
    }
}

/// Owned host track arrays. Preparing these calls no GPU APIs and can run on
/// the CPU producer thread before a device is opened or a track is uploaded.
pub struct PreparedGpuWorld {
    arrays: crate::gpu_sim::TrackArrays,
    rays: (Vec<RayNode>, Vec<[f32; 4]>, f32, usize),
}

impl PreparedGpuWorld {
    pub fn new(world: &crate::world::World) -> Result<Self, String> {
        let track = &world.track;
        if track.raycaster_present() == Some(false) { return Err("GPU port expects a BSP raycaster".into()); }
        let (nodes, walls, magnitude, depth) = track.ray_tree().ok_or("GPU port needs the BSP RayTree")?.gpu_arrays();
        let nodes = nodes.into_iter().map(|(links, own, subtree)| RayNode { links, own, subtree }).collect();
        Ok(Self { arrays: crate::gpu_sim::TrackArrays::new(world)?, rays: (nodes, walls, magnitude, depth) })
    }
}

impl<'a> GpuWorld<'a> {
    pub fn new(gpu: &'a Gpu, world: &crate::world::World) -> GpuWorld<'a> {
        Self::from_prepared(gpu, PreparedGpuWorld::new(world).unwrap_or_else(|e| panic!("GPU track export: {e}")))
    }

    /// Upload CPU-prepared geometry without rebuilding spatial query arrays.
    pub fn from_prepared(gpu: &'a Gpu, prepared: PreparedGpuWorld) -> GpuWorld<'a> {
        crate::gpu_sim::check_layout(gpu);
        let PreparedGpuWorld { arrays, rays } = prepared;
        let desc = arrays.desc(&rays);
        let create: unsafe extern "C" fn(*const crate::gpu_sim::WorldDesc) -> *mut c_void = gpu.symbol("altd_gpu_world_create");
        // The library copies every array; `rays` only has to outlive the call.
        let handle = unsafe { create(&desc) };
        assert!(!handle.is_null(), "altd_gpu_world_create failed");
        GpuWorld { gpu, handle, arrays }
    }

    pub fn gpu(&self) -> &'a Gpu {
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
        Gpu::check(unsafe { f(self.handle, op as i32, n, input.as_ptr(), out.as_mut_ptr()) }, "query");
        out
    }
}
