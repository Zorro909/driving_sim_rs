# Local simulator server

`altd-sim serve` exposes native `Session` simulation over a WebSocket. A client can use Rayon CPU threads or request the AMD HIP backend. The server implements the protocol for Drive Lab's planned native client; the webapp integration is a separate change.

The server owns one session per connection. Clients own run storage, checkpoints, logs, stage changes and mutation schedules. Closing the socket drops its session.

## Starting the server

The default Cargo features include `server`. CPU use needs no ROCm installation. Build the HIP library separately using [the GPU guide](../gpu/README.md).

```sh
cargo build --release --offline
target/release/altd-sim --threads 8 serve
target/release/altd-sim serve --port 47800 \
  --allow-origin http://127.0.0.1:5173
```

| Flag | Default and effect |
| --- | --- |
| `--port N` | `47800`; binds only `127.0.0.1:N` |
| `--allow-origin ORIGIN` | `https://drivinglab.jectrum.de`; repeat to allow several origins, replacing the default list |
| `--threads N` | Global flag; sizes the Rayon pool shared by all CPU sessions, with CPU affinity as the default |
| `--math-profile PROFILE` | Global flag; the math profile of sessions whose options name none. Without it, the server detects the profile of its own machine |

Allowed origins use `http://host[:port]` or `https://host[:port]`, with an ASCII hostname or IP address. Startup normalizes scheme and host case, compresses IPv6 addresses, removes a trailing slash and omits default ports. Paths, credentials, query strings and fragments are rejected. Startup prints the listening URL, allowed origins, CPU thread count, HIP availability and the default math profile.

## Connection checks

Connect to `ws://127.0.0.1:47800/v1`, substituting the configured port. The handshake returns HTTP 403 unless all checks pass:

- The request path is `/v1`.
- `Origin` is present and exactly matches an allowed origin. Different schemes or ports do not match; `null` is rejected.
- `Host` names `127.0.0.1` or `localhost` with the listening port. Other hostnames are rejected to prevent DNS rebinding.

There is no LAN binding, TLS or authentication token. These checks restrict browser callers; a local program can supply its own headers. At most 64 connections (including incomplete handshakes) may be active. Excess sockets are closed before spawning a worker. Handshake socket reads/writes time out after 5 seconds; established socket reads/writes after 10 minutes of blocked I/O. Long-running generation computation is not subject to those socket timeouts. Connection-worker spawn failures reject that connection while retaining the accept loop. Requests run in order on a dedicated connection thread, with no per-request timeout. Incoming frames and assembled messages are each limited to 64 MiB, including binary checkpoint headers/payloads. Equal limits allow a browser to send an admissible message in one frame: JavaScript WebSocket.send cannot select continuation framing. Larger messages must be split into protocol-level operations; fragmenting a message does not bypass its assembled limit. A rejected oversized frame/message closes only that connection. A generation may take minutes.

## Protocol 1

The first frame is a JSON text greeting:

```json
{"type":"hello","protocol":1,"version":"1.2.0","threads":8,"carStateStride":14,"carStateFields":["x","y","..."],"hip":{"available":true,"device":"AMD Radeon RX 7900 XTX","platform":"HIP"},"mathProfile":"proton","mathProfiles":["proton","win10-fma3","win11-fma3"]}
```

Read `carStateStride` and `carStateFields` from the greeting rather than hard-coding them. The example's fields are abbreviated. `platform` is `"HIP"` for the AMD library and `"CUDA"` for the NVIDIA one. When HIP is unavailable, `hip` instead contains `{"available":false,"reason":"..."}`. Availability means the process opened the library and device; a particular session can still fall back to CPU. `mathProfile` is the default math profile and `mathProfiles` lists every profile a session can choose; [math profiles](../math/README.md) documents their runtime variants.

Requests use an unsigned integer `id`, an operation name and optional `args`:

```json
{"id":7,"op":"advance","args":{"ticks":180,"stopWhenInactive":true}}
```

A successful reply contains the same `id`, its result in `ok` and the current `state`. An error reply contains `id` and `error`, without `state`.

```json
{"id":7,"ok":180,"state":{"started":true,"generation":0,"tick":180,"population":256,"activeCount":256,"checkpointGeneration":0,"backend":"cpu","backendNote":null,"mathProfile":"proton"}}
```

`checkpointGeneration` identifies the restorable generation boundary, or is `null` before starting. It need not describe the current tick. `backend` reports what actually runs; `backendNote` explains a CPU fallback. `mathProfile` is the session's math profile. Clients can queue requests and match replies by `id`.

### Operations

| Operation | `args` | `ok` |
| --- | --- | --- |
| `create` | `scene`, `network`, `model` as JSON strings; optional `options` as a SessionOptions object | `{backend, backendNote, mathProfile}` |
| `start` | None; seeds training from the `create` network export | `null` |
| `startWithShape` | `shape` as an array of unsigned 32-bit integers | `null` |
| `restoreCheckpointBytes` | Binary payload, no arguments required | `null` |
| `restoreCheckpointJson` | `json` as a checkpoint JSON string | `null` |
| `advance` | `ticks`, `stopWhenInactive` | Executed ticks |
| `advanceGeneration` | `timeLimitTicks` | Executed ticks |
| `nextGeneration` | None | `{preservedCount, rewards}` |
| `setEvolutionSettings` | `json` as an evolution-settings JSON string; `algorithm` is `"ga"` or `"ars"` (see design notes) | `null` |
| `replaceTrack` | `scene` as a scene JSON string | `null` |
| `generationSummary` | None | `{bestIndex, bestScore, lapped, active, lapIndex, lapTime}`, optional `averageScore` and `worstScore` |
| `networkJson` | `index` | Network JSON string |
| `carStates` | None | Number of `f64` values, with a binary payload |
| `checkpointBytes` | None | `null`, with a binary payload |

Call `create` once, then start or restore before advancing. `options.backend` accepts `"cpu"`, the default, or `"hip"`. `options.mathProfile` accepts `"proton"`, `"win10-fma3"` or `"win11-fma3"` and defaults to the server's profile; checkpoints restore only into sessions of the profile they were made with. Other SessionOptions fields retain their regular defaults and unknown fields are rejected. A second successful `create` on the same connection is an error. Unknown operations and malformed JSON return errors while leaving the connection usable.

`start` uses the export's `weights` and `biases` as the seed, matching `train-scratch --init-network` and the browser simulator. GA breeds mutations of that seed; ARS samples around it as its initial search point. Without weights, `start` draws a Xavier seed from `shape` (or `summary.shape`). Without either weights or a shape, use `startWithShape`. Malformed weighted exports return an error before starting and leave the connection usable.

### Binary frames

`restoreCheckpointBytes` requests and `carStates` or `checkpointBytes` replies use a binary WebSocket frame:

```text
[u32 little-endian JSON byte length][UTF-8 JSON header][raw payload]
```

The header has the same request or reply shape as a text frame. `carStates` payloads contain little-endian `f64` values, grouped by the greeting's stride and field order. Checkpoint payloads use the shared Session checkpoint format. All other operations use text frames. Invalid binary framing closes the connection.

## HIP fallback and resume

Only one session can hold the HIP device per process. Another HIP request runs on CPU with `backendNote: "another session is using the GPU"`. Library/device errors, native broadphase tracks and unsupported path sensors also cause CPU fallback with a reason. HIP requires physics shapes and at least two path points. Networks support up to 12 layers, 32 inputs and 16 neurons in each later layer; wider CPU networks fall back during verification.

Before a HIP session's first driving window after a start or restore, it compares 12 ticks against CPU from the same state, then restores that state. A mismatch or verification error drops HIP and runs the requested window on CPU. Track replacement uploads the new world and repeats verification before driving. Read `state.backend` after advancing because it may change from the `create` result.

HIP cars and agents stay on the device between windows; an `advance` or `advanceGeneration` reply reads back only the active car count. `generationSummary`, `carStates`, turnover, restores and track replacement read the population back into the CPU runner first; checkpoints are saved from the generation boundary and need no readback. Turnover breeds on CPU to retain the parent records in format-2 checkpoints, then uploads the new networks and cars on the next HIP window. HIP libraries without `altd_gpu_sim_active_count` read the population back after every window. Checkpoints can move between native CPU, HIP and WASM sessions. Native and WASM training have known last-bit differences, so continued training may diverge after switching engines.

HIP errors after startup verification return request errors without retrying. The client should stop the run and resume from its last saved checkpoint. A closed connection loses the server's session; reconnect, create a new session and restore a saved checkpoint. Release builds use `panic = "abort"`, so a panic terminates the server and closes every connection.
