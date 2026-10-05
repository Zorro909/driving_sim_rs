# Native simulator server for Drive Lab

Date: 2026-10-05
Status: implemented and reviewed

## Goal

Drive Lab (`driving_webapp_public`) trains in the browser on WASM, optionally with the WebGPU raycaster, and that stays the default. Users who also run the native `altd-sim` binary should be able to train faster with it: on native CPU threads without the browser's 8-thread cap and WASM overhead, or on an AMD GPU through the existing HIP simulator.

The browser talks to the native binary over a WebSocket on `127.0.0.1`. By default only `https://drivinglab.jectrum.de` may connect; a CLI argument replaces that list.

### Success criteria

- With `altd-sim serve` running and the setting enabled, a user can create a run with backend "Native CPU" or "Native GPU (HIP)". It trains, saves checkpoints and logs to IndexedDB, and pauses, resumes and changes stages exactly as WASM runs do.
- A native run's checkpoint resumes on WASM and the other way round. The format is shared; results are not bit-identical, see "Known limitation".
- No request reaches the server from an origin that is not allowed, or with a `Host` other than the loopback address and port.
- When the server is missing, unreachable or crashes, the user gets a clear message and the run falls back or resumes as the existing WebGPU and WASM-trap paths do.
- A HIP session matches a native CPU session exactly in car state, as the existing HIP checks require, or falls back to CPU with a reason.

### Non-goals

- The server does not own runs, storage or the stage, plateau and mutation-schedule logic; those stay in the webapp's coordinator worker.
- No remote or LAN access, TLS or authentication tokens.
- No HIP version of the parent-recording turnover (see "Turnover"); that is a later optimization.
- WASM/native bit-exactness is tracked separately in issue #3.

## Architecture

```
Page (main thread)            Coordinator worker                    altd-sim serve (127.0.0.1)
 ├─ setting + status probe ──► (hello only, then close) ──────────►  ├─ handshake: Origin + Host check
 └─ TrainingWorker ──────────► simulation.worker.ts                  ├─ one thread per connection
                                ├─ training loop (unchanged)          └─ one Session per connection
                                ├─ Engine interface                        ├─ backend: cpu (rayon)
                                │   ├─ WasmEngine (Simulation, WebGPU)     └─ backend: hip (libaltd_gpu.so)
                                │   └─ RemoteEngine ── WebSocket ──►
                                └─ IndexedDB (runs, logs, checkpoints)
```

The server works as a remote `Simulation`. The worker keeps its training loop and swaps the local `Simulation` for an object that sends the same calls over the socket.

## Part 1: HIP backend in `Session` (`driving_sim_rs`)

### Native access to Session methods

Some `Session` items exist only in WASM builds and must also be compiled for the server:

- `options` field (`src/training/session.rs:76`)
- `set_evolution_settings` (`:313`)
- `replace_track` (`:653`)
- the `network` field the WASM build keeps

Their cfg changes from "wasm" to "wasm or `server` feature". The server module lives in the library (`src/server/`) so it can use the `pub(crate)` methods directly.

### Backend option

`SessionOptions` gains `backend: "cpu" | "hip"` (camelCase JSON, default `"cpu"`). `deny_unknown_fields` stays.

- On wasm32, `"hip"` is an error: "the HIP backend is only available in the native simulator".
- On native, `Session` gains `hip: Option<HipState>`:

```rust
struct HipState {
    world: GpuWorld<'static>,
    sim: GpuSim<'static>,
}
static GPU: OnceLock<Result<Gpu, String>> = OnceLock::new(); // Gpu::open(None): ALTD_GPU_LIB or the default path
```

The `Gpu` handle is opened once per process and never dropped, so a `Session` can own `'static` GPU handles without self-borrowing.

### When HIP is refused

`Session::new` builds `HipState` only when all of these hold. Otherwise it keeps `hip = None` and records a reason that the server reports, and the session runs on CPU.

- `GPU` opened successfully.
- The track does not use the native broadphase (`runner.physics` would be `Some`); `advance_window_gpu` rejects it.
- `runner.gpu_sim(...)` succeeds. It already rejects path sensors without `track.curve`.

Population-dependent capacity is handled on the first window (below), so construction does not need the population.

### Routing

| Session call | CPU | HIP |
|---|---|---|
| `advance(ticks, stop)` | `runner.advance` | `runner.advance_window_gpu(sim, world, ticks, stop, None)` |
| `advance_generation(limit)` | `runner.advance_generation` | `runner.advance_generation_gpu(sim, world, limit)` |
| `next_generation()` | `next_generation_traced` | the same CPU path (see "Turnover") |
| `replace_track(scene)` | as today | as today, then `GpuWorld::new(gpu, &runner.world)` and `sim.set_world(&world)` |
| `start*`, `restore_checkpoint*`, `set_network_json` | as today | as today, then `sim.network_tag = None` |

Each HIP window reads car and agent state back into `runner.agents`, so `car_states`, `generation_summary`, `network_json`, `metrics`, `active_count` and checkpoints read the runner as they do now.

If the population is larger than `GpuSim`'s capacity, `GpuSim` is rebuilt with the new population before the window.

### Turnover

`next_generation_gpu` breeds on the GPU but does not produce the parent record (`Lineage`) that format 2 checkpoints store. A HIP session therefore breeds through `next_generation_traced` on the CPU, the same call CPU sessions make. The next window uploads the new networks because `network_tag` no longer matches. Breeding is small next to a generation of driving. A HIP turnover that also records parents can come later if profiling shows it matters.

### Startup check

A HIP session checks itself before its first real window, following `GpuSimulation::verify` in `src/wasm/sim/mod.rs:186-235`:

1. Back up `runner.agents`, `tick` and `batch_index`.
2. Advance 12 ticks on the CPU and record the canonical state (`canonical_state`, moved from `src/wasm/sim` into shared native and WASM code).
3. Restore the backup, run 12 ticks on HIP, and compare canonical states exactly.
4. Restore the backup again.

On a mismatch or HIP error, the session drops `HipState` and continues on CPU with the reason "HIP verification differs from the CPU simulator" (or the error). It runs once per session, on the first `advance` or `advance_generation` after `start*` or `restore*`.

`Session` exposes `backend() -> "cpu" | "hip"` and `backend_note() -> Option<String>`, so hosts can report what actually runs.

### Errors

HIP failures after the check, such as a failed window, are returned as `Err(String)` and not retried. The webapp fails the run, and resuming continues from the last checkpoint.

## Part 2: `altd-sim serve`

### Crate layout

- Feature `server = ["cli", "dep:tungstenite"]`, added to `default`. `tungstenite` is the synchronous WebSocket crate without TLS features, so there is no async runtime.
- `src/server/mod.rs`: listener, handshake checks and the per-connection loop.
- `src/server/protocol.rs`: message types and frame encoding.
- `src/server/dispatch.rs`: operations to `Session` calls.
- `src/bin/altd-sim`: a `Serve` subcommand that parses flags and calls `altd_sim::server::run(config)`.

### Flags

```
altd-sim [--threads N] serve [--port 47800] [--allow-origin ORIGIN]...
```

- `--port`: TCP port, default `47800`. The server binds `127.0.0.1:<port>` only.
- `--allow-origin`: repeatable. When given, the list replaces the default `https://drivinglab.jectrum.de`. Values are normalized as `scheme://host[:port]`, and invalid values are rejected at startup. Local development uses `--allow-origin http://127.0.0.1:5173`.
- `--threads`: the existing global flag; it sizes the rayon pool all CPU sessions share.

On startup the server prints the address, allowed origins, thread count and HIP availability.

### Handshake checks

`tungstenite::accept_hdr` with a callback that rejects with HTTP 403 and a short plain-text reason unless all hold:

- The path is `/v1`.
- `Origin` is present and equals an allowed origin exactly (scheme, host and port). `null` is rejected.
- `Host` is `127.0.0.1:<port>` or `localhost:<port>`.

### Greeting

Sent as the first text frame after the handshake:

```json
{"type":"hello","protocol":1,"version":"0.1.0","threads":32,
 "carStateStride":N,"carStateFields":["..."],
 "hip":{"available":true,"device":"..."} | {"available":false,"reason":"..."}}
```

Checking HIP availability calls `GPU.get_or_init` once per process. `device` comes from the HIP library if it exposes a name; otherwise it is the library path.

### Requests and replies

- Request (text frame): `{"id":7,"op":"advance","args":{"ticks":180,"stopWhenInactive":true}}`
- Success: `{"id":7,"ok":<result>,"state":State}`
- Failure: `{"id":7,"error":"<text>"}`
- `State` is `{"started","generation","tick","population","activeCount","checkpointGeneration","backend","backendNote"}`.

Requests are handled strictly in order on the connection's thread. Clients may send the next request before the reply arrives and match replies by `id`.

### Binary frames

Used by `restoreCheckpointBytes` (request), and by `checkpointBytes` and `carStates` (replies):

```
[u32 little-endian: JSON length][JSON header][raw payload]
```

The JSON header has the same shape as a text frame. `carStates` returns little-endian `f64` values. All other messages are text frames.

### Operations

| op | args | ok |
|---|---|---|
| `create` | `scene`, `network`, `model` (JSON strings), `options` (SessionOptions object, including `backend`) | `{backend, backendNote}` |
| `startWithShape` | `shape: number[]` | `null` |
| `restoreCheckpointBytes` | binary payload | `null` |
| `restoreCheckpointJson` | `json: string` | `null` |
| `advance` | `ticks`, `stopWhenInactive` | executed ticks |
| `advanceGeneration` | `timeLimitTicks` | executed ticks |
| `nextGeneration` | none | `{preservedCount, rewards: number[]}` |
| `setEvolutionSettings` | `json: string` | `null` |
| `replaceTrack` | `scene: string` | `null` |
| `generationSummary` | none | `{bestIndex, bestScore, lapped, active, lapIndex, lapTime}` |
| `networkJson` | `index` | string |
| `carStates` | none | binary `f64` payload |
| `checkpointBytes` | none | binary payload |

Rules:

- Any op before `create`, or a second `create`, is an error.
- Only one HIP session may exist per process at a time. A second `create` with `backend: "hip"` falls back to CPU with the note "another session is using the GPU".
- Unknown ops and malformed JSON are errors and leave the session usable. Malformed binary framing closes the connection.
- The session is dropped when the socket closes.

### Panics and timeouts

The release profile keeps `panic = "abort"`, so a panic ends the server. The client sees the socket close (see "Disconnect" below), and a session that panicked part-way would not be trustworthy anyway. There is no per-request timeout, because `advanceGeneration` can take minutes on large populations.

## Part 3: webapp client (`driving_webapp_public`)

### Setting and status

- In `#compute-choices` (`src/views.ts` `computeChoices`, wired in `src/main.ts` `bindCompute`), above the backend radios:
  - a "Use a local altd-sim" checkbox and a port input;
  - help text with the command `altd-sim serve` and the note that the browser may ask once for local network access.
- The setting is stored in `localStorage` (`drivelab.native` = `{enabled, port}`) and is per machine.
- When enabled, the page opens a WebSocket to `ws://127.0.0.1:<port>/v1`, reads `hello`, closes it, and shows one of:
  - "Connected: altd-sim 0.1.0 · 32 threads · HIP: <device>"
  - "Not reachable on port N"
  - "Update altd-sim: protocol N is not supported"

  It probes again when the setting changes and whenever the compute choices are drawn. The page timeout is 3 s.
- New radios:
  - "Native CPU · N threads": enabled when connected.
  - "Native GPU (HIP) · device": enabled when `hip.available`, otherwise disabled with the server's reason.
  - Both carry the note "Results differ slightly from in-browser training", linking to issue #3.
- When the selected native backend becomes unavailable while the user is editing a draft, the draft switches to `"cpu"`, as WebGPU does in `main.ts:115`.

### Domain and protocol

- `Setup.backend` becomes `"cpu" | "webgpu" | "native-cpu" | "native-hip"`.
- `validateSetup` accepts the new values; run summaries and the "Computing on" line show "Native CPU, N threads" and "Native GPU (HIP), <device>".
- `StartCommand` gains `native?: { port: number }`, set by the page from the setting.
- The `engine` event's `backend` gains `"native-cpu" | "native-hip"` and an optional `device`.

### Engine interface (`src/engine.ts`)

```ts
interface Engine {
  readonly started: boolean; readonly generation: number; readonly tick: number;
  readonly population: number; readonly activeCount: number;
  readonly checkpointGeneration: number | undefined;
  startWithShape(shape: Uint32Array): Promise<void>;
  restoreCheckpointBytes(bytes: Uint8Array): Promise<void>;
  restoreCheckpointJson(json: string): Promise<void>;
  advance(ticks: number, stopWhenInactive: boolean): Promise<number>;
  advanceGeneration(timeLimitTicks: number): Promise<number>;
  nextGeneration(): Promise<void>;
  setEvolutionSettings(json: string): Promise<void>;
  replaceTrack(sceneJson: string): Promise<void>;
  generationSummary(): Promise<Summary>;
  networkJson(index: number): Promise<string>;
  carStates(): Promise<Float64Array>;
  checkpointBytes(): Promise<Uint8Array>;
  free(): void;
}
```

- `WasmEngine` wraps `lib.Simulation` and holds the existing WebGPU logic (`prepareGpu`, `dropGpu`, the GPU-to-CPU fallback in `advance`).
- `RemoteEngine` opens the socket, checks `hello.protocol === 1`, sends `create`, and updates the cached getters from every reply's `state`. Its getters are synchronous because they return those cached values.
- `simulation.worker.ts` replaces `sim` and `gpu` with one `engine: Engine` and adds `await` to the calls that become asynchronous. The training loop, storage queue, stage logic and messages are otherwise unchanged.
- The WASM library still loads for every backend, because `trackScene` and `randomTrackScene` run in the worker.

### Fallback and disconnect

- **Server unreachable at `buildSession`, `hello` rejected, or `create` failing:** the run uses `WasmEngine` on CPU and emits an `engine` event with a note, for example "The local altd-sim could not be reached on port 47800, so this run trains in the browser. <reason>".
- **HIP refused or failing its check:** the server reports `backend: "cpu"` with `backendNote`, and the worker emits `engine` `native-cpu` with that note.
- **Disconnect during a run:** `RemoteEngine` rejects all pending and future calls with "The native simulator closed the connection. Resuming continues from the last saved checkpoint." The worker treats this like a WASM trap: it sets `trapped`, so `publishRun(true)` skips `checkpointBytes`, marks the run failed, stores it, and reports the error. Resuming goes through the same start path, including the fallback above.

### Content Security Policy

Add `ws://127.0.0.1:*` to `connect-src` in `public/_headers` and in `vite.config.ts`. `deploy/deploy.sh` reads `_headers`, and `scripts/serve-test.mjs` reads the same policy. COOP/COEP do not apply to WebSocket connections.

## Known limitation

WASM and native CPU are not bit-exact (issue #3). Native CPU and HIP are expected to be exact with each other, and the startup check enforces it. Cross-engine resume works through the shared checkpoint format, but continued training diverges. The UI says so next to the native choices.

## Testing

### `driving_sim_rs`

- **Unit:**
  - origin normalization and matching: exact, different port, different scheme, trailing slash, `null`, missing;
  - Host matching;
  - binary frame encode and decode, including truncated frames;
  - `SessionOptions` with `backend` (`"hip"` rejected on wasm by a cfg test).
- **Integration (`tests/server.rs`):** start the server in-process on port 0 and connect with a `tungstenite` client.
  - A disallowed origin and a wrong Host get 403.
  - `hello` contents.
  - `create → startWithShape → advance → advanceGeneration → nextGeneration → generationSummary → carStates → checkpointBytes → restoreCheckpointBytes` matches a direct `Session` with the same inputs bit for bit.
  - Ops before `create` and a second `create` return errors.
  - A dropped socket frees the session.
- **HIP (skipped when `Gpu::open` fails):**
  - a CPU and a HIP `Session` agree in canonical state through several `advance` windows, `advanceGeneration` and two turnovers;
  - a checkpoint saved under HIP restores under CPU, and the other way round;
  - fallback reasons for native broadphase and a missing curve;
  - a second concurrent HIP session falls back to CPU.
- Existing tests: `cargo test`, the WASM parity harness, and a WASM build that confirms the cfg changes do not leak native-only code.

### `driving_webapp_public`

- Domain tests: `validateSetup` with the new backends.
- Playwright (`tests/native.spec.ts`): build `altd-sim` from `SIM_ROOT` and start `altd-sim serve --port <free> --allow-origin http://127.0.0.1:<test port>`.
  - The setting shows "Connected".
  - A two-generation Native CPU run stores progress and a checkpoint, with the correct engine event.
  - Setting enabled with no server: the run falls back to CPU with a note.
  - Killing the server mid-run fails the run with the disconnect message; restarting the server and resuming continues from the checkpoint.
  - CSP permits the socket.
  - The HIP case is skipped unless the server reports HIP available.

## Documentation

- `docs/server.md`: purpose, flags, security model, the protocol (greeting, ops, frames) and fallback behavior.
- `docs/cli.md`: the `serve` subcommand.
- `docs/gpu.md`: HIP through `Session`, and CPU turnover with network re-upload.
- Webapp `README.md`: the setting, the CSP change and native testing.

## Delivery order

1. HIP backend in `Session`, with its tests. Independently useful; no server needed.
2. `altd-sim serve` and the protocol, with integration tests (CPU, then HIP).
3. Webapp client, CSP and Playwright tests.

Each phase lands with green tests before the next starts.
