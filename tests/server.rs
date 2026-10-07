//! `altd-sim serve` over real sockets: handshake checks, the greeting, and
//! a remote session that matches a direct `Session` bit for bit.
#![cfg(all(feature = "server", not(target_arch = "wasm32")))]

use altd_sim::math::profile::MathProfile;
use altd_sim::nn::network::Network;
use altd_sim::server::protocol::{decode_binary, encode_binary};
use altd_sim::server::{Config, Server};
use altd_sim::training::session::{Session, SessionOptions};
use serde_json::{json, Value};
use std::net::{SocketAddr, TcpStream};
use std::sync::Mutex;
use tungstenite::client::IntoClientRequest;
use tungstenite::{Message, WebSocket};

#[path = "../examples/support/generated.rs"]
mod generated;

const ORIGIN: &str = "http://127.0.0.1:5173";

/// HIP sessions share one device claim per process.
static HIP_TESTS: Mutex<()> = Mutex::new(());

fn start() -> SocketAddr {
    let server = Server::bind(Config {
        port: 0,
        origins: vec![ORIGIN.to_string()],
        math_profile: Some(MathProfile::Proton),
    })
    .unwrap();
    let address = server.local_addr().unwrap();
    std::thread::spawn(move || server.run());
    address
}

fn handshake(
    address: SocketAddr,
    path: &str,
    origin: Option<&str>,
    host: Option<&str>,
) -> Result<WebSocket<TcpStream>, u16> {
    let mut request = format!("ws://{address}{path}").into_client_request().unwrap();
    let headers = request.headers_mut();
    if let Some(origin) = origin {
        headers.insert("origin", origin.parse().unwrap());
    }
    if let Some(host) = host {
        headers.insert("host", host.parse().unwrap());
    }
    let stream = TcpStream::connect(address).unwrap();
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(30)))
        .unwrap();
    stream
        .set_write_timeout(Some(std::time::Duration::from_secs(30)))
        .unwrap();
    match tungstenite::client::client(request, stream) {
        Ok((socket, _)) => Ok(socket),
        Err(tungstenite::HandshakeError::Failure(tungstenite::Error::Http(response))) => {
            Err(response.status().as_u16())
        }
        Err(e) => panic!("handshake: {e}"),
    }
}

struct Client {
    socket: WebSocket<TcpStream>,
    hello: Value,
    next: u64,
}

impl Client {
    fn connect(address: SocketAddr) -> Client {
        let mut socket = handshake(address, "/v1", Some(ORIGIN), None).unwrap();
        let hello = match socket.read().unwrap() {
            Message::Text(t) => serde_json::from_str(&t).unwrap(),
            other => panic!("expected hello, got {other:?}"),
        };
        Client { socket, hello, next: 1 }
    }

    fn read(&mut self) -> (Value, Vec<u8>) {
        match self.socket.read().unwrap() {
            Message::Text(t) => (serde_json::from_str(&t).unwrap(), Vec::new()),
            Message::Binary(b) => {
                let (header, payload) = decode_binary(&b).unwrap();
                (header, payload.to_vec())
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    /// Sends a request and returns the reply (header and binary payload).
    fn call_raw(&mut self, op: &str, args: Value, payload: Option<&[u8]>) -> (Value, Vec<u8>) {
        let id = self.next;
        self.next += 1;
        let request = json!({"id": id, "op": op, "args": args});
        let message = match payload {
            Some(p) => Message::binary(encode_binary(&request, p)),
            None => Message::text(request.to_string()),
        };
        self.socket.send(message).unwrap();
        let (reply, payload) = self.read();
        assert_eq!(reply["id"], json!(id));
        (reply, payload)
    }

    fn call(&mut self, op: &str, args: Value) -> Value {
        let (reply, _) = self.call_raw(op, args, None);
        assert!(reply.get("error").is_none(), "{op}: {}", reply["error"]);
        reply["ok"].clone()
    }

    fn error(&mut self, op: &str, args: Value) -> String {
        let (reply, _) = self.call_raw(op, args, None);
        reply["error"]
            .as_str()
            .unwrap_or_else(|| panic!("{op} succeeded: {reply}"))
            .to_owned()
    }

    fn create(&mut self, options: &Value) -> Value {
        self.call(
            "create",
            json!({
                "scene": generated::scene("formula", 0).to_string(),
                "network": generated::network("formula").to_string(),
                "model": generated::model("formula").to_string(),
                "options": options,
            }),
        )
    }
}

/// A local session with the options of a remote one, on the server's
/// default math profile unless the options name one.
fn direct(options: &Value) -> Session {
    let mut options = SessionOptions::from_json(&options.to_string()).unwrap();
    options.math_profile.get_or_insert(MathProfile::Proton);
    Session::new(
        &generated::scene("formula", 0),
        &generated::network("formula"),
        &generated::model("formula"),
        options,
    )
    .unwrap()
}

fn f64s(bytes: &[u8]) -> Vec<u64> {
    bytes
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn handshakes_need_an_allowed_origin_host_and_path() {
    let address = start();
    let port = address.port();
    let ok_host = format!("localhost:{port}");
    assert!(handshake(address, "/v1", Some(ORIGIN), Some(&ok_host)).is_ok());
    let cases = [
        ("/v1", None, None),
        ("/v1", Some("https://drivinglab.jectrum.de"), None),
        ("/v1", Some("http://127.0.0.1:5174"), None),
        ("/v1", Some("https://127.0.0.1:5173"), None),
        ("/v1", Some("null"), None),
        ("/v1", Some(ORIGIN), Some("evil.example:47800")),
        ("/v1", Some(ORIGIN), Some("127.0.0.1:1")),
        ("/v2", Some(ORIGIN), None),
        ("/", Some(ORIGIN), None),
    ];
    for (path, origin, host) in cases {
        assert_eq!(
            handshake(address, path, origin, host).err(),
            Some(403),
            "{path} {origin:?} {host:?}"
        );
    }
}

#[test]
fn hello_describes_the_server() {
    let client = Client::connect(start());
    let hello = &client.hello;
    assert_eq!(hello["type"], "hello");
    assert_eq!(hello["protocol"], 1);
    assert_eq!(hello["version"], altd_sim::VERSION);
    assert!(hello["threads"].as_u64().unwrap() >= 1);
    let stride = hello["carStateStride"].as_u64().unwrap() as usize;
    assert_eq!(hello["carStateFields"].as_array().unwrap().len(), stride);
    match hello["hip"]["available"].as_bool().unwrap() {
        true => assert!(hello["hip"]["device"].is_string()),
        false => assert!(hello["hip"]["reason"].is_string()),
    }
    assert_eq!(hello["mathProfile"], "proton");
    assert_eq!(hello["mathProfiles"], json!(["proton", "win10-fma3", "win11-fma3"]));
}

#[test]
fn server_configuration_normalizes_and_rejects_invalid_origins() {
    let server = Server::bind(Config {
        port: 0,
        origins: vec!["HTTP://127.0.0.1:5173/".to_string()],
        math_profile: None,
    })
    .unwrap();
    let address = server.local_addr().unwrap();
    std::thread::spawn(move || server.run());
    assert!(handshake(address, "/v1", Some(ORIGIN), None).is_ok());
    let error = Server::bind(Config {
        port: 0,
        origins: vec!["null".to_string()],
        math_profile: None,
    })
    .err()
    .expect("invalid origins must fail before listening");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
}

/// The remote session runs the same code as a direct one, so every reply
/// matches it exactly.
#[test]
fn a_remote_session_matches_a_direct_session() {
    let options = json!({"population": 8, "seed": 5, "eliminateOnWall": true,
        "settings": {"selection_size": 3, "mutation_rate": 0.3, "weight_decay": 0.0}});
    let address = start();
    let mut client = Client::connect(address);
    let mut session = direct(&options);
    assert_eq!(
        client.create(&options),
        json!({"backend": "cpu", "backendNote": null, "mathProfile": "proton"})
    );
    let shape = [20, 8, 5];
    let (reply, _) = client.call_raw("startWithShape", json!({"shape": shape}), None);
    session.start_with_shape(&shape).unwrap();
    assert_eq!(
        reply["state"],
        json!({"started": true, "generation": 0, "tick": 0, "population": 8, "activeCount": 8,
            "checkpointGeneration": 0, "backend": "cpu", "backendNote": null, "mathProfile": "proton"})
    );
    assert_eq!(
        client.call("advance", json!({"ticks": 90, "stopWhenInactive": false})),
        json!(session.advance(90, false).unwrap())
    );
    assert_eq!(
        client.call("advanceGeneration", json!({"timeLimitTicks": 300})),
        json!(session.advance_generation(300).unwrap())
    );
    assert_eq!(
        client.call("generationSummary", Value::Null),
        serde_json::to_value(session.generation_summary()).unwrap()
    );
    let (preserved, rewards) = session.next_generation().unwrap();
    let next = client.call("nextGeneration", Value::Null);
    assert_eq!(next["preservedCount"], json!(preserved));
    let remote_rewards: Vec<f64> = serde_json::from_value(next["rewards"].clone()).unwrap();
    assert_eq!(bits(&remote_rewards), bits(&rewards));
    let settings = r#"{"population": 6, "selection_size": 2, "mutation_rate": 0.1}"#;
    client.call("setEvolutionSettings", json!({"json": settings}));
    session.set_evolution_settings(settings).unwrap();
    let track = generated::scene("formula", 1);
    client.call("replaceTrack", json!({"scene": track.to_string()}));
    session.replace_track(&track).unwrap();
    client.call("advanceGeneration", json!({"timeLimitTicks": 240}));
    session.advance_generation(240).unwrap();
    let (reply, states) = client.call_raw("carStates", Value::Null, None);
    assert_eq!(f64s(&states), bits(&session.car_states()));
    assert_eq!(reply["ok"], json!(session.car_states().len()));
    let (reply, checkpoint) = client.call_raw("checkpointBytes", Value::Null, None);
    assert_eq!(reply["state"]["checkpointGeneration"], 1);
    assert_eq!(checkpoint, session.checkpoint_bytes().unwrap());
    let network: Value =
        serde_json::from_str(client.call("networkJson", json!({"index": 0})).as_str().unwrap()).unwrap();
    let network = Network::from_game_export(&network);
    assert_eq!(network.shape, session.runner.agents[0].network.shape);
    assert_eq!(bits(&network.params), bits(&session.runner.agents[0].network.params));

    // A new connection restores the checkpoint and replays the generation.
    let mut other = Client::connect(address);
    other.create(&options);
    let (reply, _) = other.call_raw("restoreCheckpointBytes", json!({}), Some(&checkpoint));
    assert!(reply.get("error").is_none(), "{reply}");
    assert_eq!(reply["state"]["generation"], 1);
    let mut replay = direct(&options);
    replay.restore_checkpoint_bytes(&checkpoint).unwrap();
    other.call("advanceGeneration", json!({"timeLimitTicks": 240}));
    replay.advance_generation(240).unwrap();
    assert_eq!(
        f64s(&other.call_raw("carStates", Value::Null, None).1),
        bits(&replay.car_states())
    );

    let (reply, _) = client.call_raw("nextGeneration", Value::Null, None);
    session.next_generation().unwrap();
    assert_eq!(reply["state"]["population"], 6);
    assert_eq!(
        f64s(&client.call_raw("carStates", Value::Null, None).1),
        bits(&session.car_states())
    );
    assert_eq!(
        client.call_raw("checkpointBytes", Value::Null, None).1,
        session.checkpoint_bytes().unwrap()
    );
}

/// A session's own math profile overrides the server's, and its
/// checkpoints only restore into sessions of the same profile.
#[test]
fn sessions_choose_their_math_profile() {
    let options = json!({"population": 4, "seed": 9, "mathProfile": "win11-fma3"});
    let address = start();
    let mut client = Client::connect(address);
    let mut session = direct(&options);
    assert_eq!(client.create(&options)["mathProfile"], "win11-fma3");
    client.call("startWithShape", json!({"shape": [20, 8, 5]}));
    session.start_with_shape(&[20, 8, 5]).unwrap();
    assert_eq!(
        client.call("advanceGeneration", json!({"timeLimitTicks": 240})),
        json!(session.advance_generation(240).unwrap())
    );
    let (reply, checkpoint) = client.call_raw("checkpointBytes", Value::Null, None);
    assert_eq!(reply["state"]["mathProfile"], "win11-fma3");
    assert_eq!(checkpoint, session.checkpoint_bytes().unwrap());

    let mut proton = Client::connect(address);
    proton.create(&json!({"population": 4}));
    let (reply, _) = proton.call_raw("restoreCheckpointBytes", Value::Null, Some(&checkpoint));
    assert!(reply["error"].as_str().unwrap().contains("win11-fma3"), "{reply}");
    let mut win11 = Client::connect(address);
    win11.create(&options);
    let (reply, _) = win11.call_raw("restoreCheckpointBytes", Value::Null, Some(&checkpoint));
    assert!(reply.get("error").is_none(), "{reply}");
    assert!(Client::connect(address)
        .error(
            "create",
            json!({"scene": "{}", "network": "{}", "model": "{}", "options": {"mathProfile": "win12"}})
        )
        .contains("options"));
}

#[test]
fn json_and_binary_checkpoints_resume_the_same_population_and_rng() {
    let options = json!({"population": 4, "seed": 27});
    let mut original = direct(&options);
    original.start_with_shape(&[20, 8, 5]).unwrap();
    original.advance_generation(120).unwrap();
    original.next_generation().unwrap();
    let json_checkpoint = json!({
        "generation": original.runner.generation,
        "shape": original.runner.agents[0].network.shape,
        "rng": original.runner.rng.to_json(),
        "networks": original.runner.agents.iter().map(|a| &a.network.params).collect::<Vec<_>>(),
    });
    let checkpoint = original.checkpoint_bytes().unwrap();
    let address = start();
    let mut from_json = Client::connect(address);
    let mut from_bytes = Client::connect(address);
    for client in [&mut from_json, &mut from_bytes] {
        client.create(&options);
    }
    let (json_reply, _) = from_json.call_raw(
        "restoreCheckpointJson",
        json!({"json": json_checkpoint.to_string()}),
        None,
    );
    let (binary_reply, _) = from_bytes.call_raw("restoreCheckpointBytes", Value::Null, Some(&checkpoint));
    assert!(json_reply.get("error").is_none(), "{json_reply}");
    assert!(binary_reply.get("error").is_none(), "{binary_reply}");
    assert_eq!(json_reply["state"], binary_reply["state"]);
    assert_eq!(json_reply["state"]["generation"], 1);
    assert_eq!(json_reply["state"]["tick"], 0);
    let executed = original.advance_generation(120).unwrap();
    for client in [&mut from_json, &mut from_bytes] {
        assert_eq!(
            client.call("advanceGeneration", json!({"timeLimitTicks": 120})),
            executed
        );
    }
    assert_eq!(
        from_json.call_raw("carStates", Value::Null, None),
        from_bytes.call_raw("carStates", Value::Null, None)
    );
    assert_eq!(
        from_json.call("nextGeneration", Value::Null),
        from_bytes.call("nextGeneration", Value::Null)
    );
    assert_eq!(
        from_json.call_raw("checkpointBytes", Value::Null, None),
        from_bytes.call_raw("checkpointBytes", Value::Null, None)
    );
}

#[test]
fn pipelined_requests_execute_in_order_and_keep_their_ids() {
    let mut client = Client::connect(start());
    client.create(&json!({"population": 4}));
    client.call("startWithShape", json!({"shape": [20, 5]}));
    for (id, ticks) in [(77, 3), (42, 7)] {
        client
            .socket
            .send(Message::text(
                json!({
                    "id": id, "op": "advance", "args": {"ticks": ticks, "stopWhenInactive": false},
                })
                .to_string(),
            ))
            .unwrap();
    }
    for (id, ticks, total) in [(77, 3, 3), (42, 7, 10)] {
        let (reply, payload) = client.read();
        assert_eq!(reply["id"], id);
        assert_eq!(reply["ok"], ticks);
        assert_eq!(reply["state"]["tick"], total);
        assert!(payload.is_empty());
    }
}

#[test]
fn errors_leave_the_connection_usable() {
    let options = json!({"population": 4});
    let mut client = Client::connect(start());
    assert!(client
        .error("advance", json!({"ticks": 1, "stopWhenInactive": false}))
        .contains("create"));
    assert!(client.error("checkpointBytes", Value::Null).contains("create"));
    assert!(client
        .error("create", json!({"scene": "{", "network": "{}", "model": "{}"}))
        .contains("scene"));
    assert!(client
        .error(
            "create",
            json!({"scene": "{}", "network": "{}", "model": "{}", "options": {"bogus": 1}})
        )
        .contains("options"));
    client.create(&options);
    assert!(client.error("create", json!({})).contains("already"));
    assert!(client
        .error("advance", json!({"ticks": 1, "stopWhenInactive": false}))
        .contains("started"));
    assert!(client
        .error("startWithShape", json!({"shape": [3, 5]}))
        .contains("inputs"));
    assert!(client
        .error("startWithShape", json!({"shape": [20, u64::MAX, 5]}))
        .contains("32-bit"));
    assert!(client.error("teleport", Value::Null).contains("unknown operation"));
    assert!(client
        .error("advance", json!({"ticks": -1, "stopWhenInactive": false}))
        .contains("ticks"));
    assert!(client.error("restoreCheckpointBytes", Value::Null).contains("binary"));
    let (reply, _) = client.call_raw("restoreCheckpointBytes", Value::Null, Some(b"invalid checkpoint"));
    assert!(reply["error"].as_str().unwrap().contains("checkpoint"));
    assert!(client
        .error("restoreCheckpointJson", json!({"json": "{}"}))
        .contains("shape"));
    let (reply, _) = client.call_raw("startWithShape", json!({"shape": [20, 5]}), Some(b"x"));
    assert!(reply["error"].as_str().unwrap().contains("binary"));
    // Malformed text requests get an error with the id, when there is one.
    client.socket.send(Message::text(r#"{"id": 99, "op": 5}"#)).unwrap();
    let (reply, _) = client.read();
    assert_eq!(reply["id"], 99);
    assert!(reply["error"].is_string());
    client.socket.send(Message::text("not json")).unwrap();
    assert_eq!(client.read().0["id"], Value::Null);
    client.call("startWithShape", json!({"shape": [20, 5]}));
    assert_eq!(
        client.call("advance", json!({"ticks": 6, "stopWhenInactive": false})),
        6
    );
}

#[test]
fn malformed_binary_frames_close_the_connection() {
    let mut client = Client::connect(start());
    client.socket.send(Message::binary(vec![200, 0, 0, 0, b'{'])).unwrap();
    loop {
        match client.socket.read() {
            Ok(Message::Close(_)) => {}
            Ok(other) => panic!("unexpected {other:?}"),
            Err(_) => break,
        }
    }
}

/// With a HIP device: a remote HIP session matches the CPU, and its device
/// claim ends with the connection.
#[test]
fn hip_sessions_end_with_their_connection() {
    let _hip = HIP_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    let address = start();
    let mut client = Client::connect(address);
    if client.hello["hip"]["available"] != true {
        eprintln!("skipping: {}", client.hello["hip"]["reason"]);
        return;
    }
    let options = json!({"population": 8, "seed": 2, "eliminateOnWall": true, "backend": "hip"});
    assert_eq!(
        client.create(&options),
        json!({"backend": "hip", "backendNote": null, "mathProfile": "proton"})
    );
    let mut second = Client::connect(address);
    assert_eq!(
        second.create(&options),
        json!({"backend": "cpu", "backendNote": "another session is using the GPU", "mathProfile": "proton"})
    );
    for c in [&mut client, &mut second] {
        c.call("startWithShape", json!({"shape": [20, 8, 5]}));
    }
    // Drive Lab's turnover: windows (the cars stay on the device between
    // them), the rest of the generation, its summary and the next generation.
    for generation in 0..2 {
        for ticks in [6, 30, 66] {
            let args = json!({"ticks": ticks, "stopWhenInactive": true});
            let (hip, _) = client.call_raw("advance", args.clone(), None);
            let (cpu, _) = second.call_raw("advance", args, None);
            assert_eq!(hip["ok"], cpu["ok"]);
            assert_eq!(hip["state"]["tick"], cpu["state"]["tick"]);
            assert_eq!(hip["state"]["activeCount"], cpu["state"]["activeCount"], "{hip}");
        }
        let generation_end = [&mut client, &mut second].map(|c| {
            let (reply, _) = c.call_raw("advanceGeneration", json!({"timeLimitTicks": 300}), None);
            (reply["ok"].clone(), reply["state"]["activeCount"].clone())
        });
        assert_eq!(generation_end[0], generation_end[1]);
        assert_eq!(
            client.call("generationSummary", Value::Null),
            second.call("generationSummary", Value::Null)
        );
        if generation == 0 {
            assert_eq!(
                client.call("nextGeneration", Value::Null),
                second.call("nextGeneration", Value::Null)
            );
        }
    }
    let (reply, states) = client.call_raw("carStates", Value::Null, None);
    assert_eq!(reply["state"]["backend"], "hip", "{reply}");
    assert_eq!(states, second.call_raw("carStates", Value::Null, None).1);
    client.socket.close(None).unwrap();
    while client.socket.read().is_ok() {}
    // The server drops the session once it sees the close; retry briefly.
    let claim = || {
        for attempt in 0.. {
            let mut client = Client::connect(address);
            let created = client.create(&options);
            if created["backend"] == "hip" {
                return client;
            }
            assert!(attempt < 50, "the GPU claim outlived its connection: {created}");
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        unreachable!()
    };
    let third = claim();
    // Dropping without the WebSocket closing handshake also releases HIP.
    drop(third);
    drop(claim());
}

#[test]
fn unsupported_hip_scenes_and_networks_fall_back_without_losing_the_connection() {
    let _hip = HIP_TESTS.lock().unwrap_or_else(|e| e.into_inner());
    let address = start();
    let mut client = Client::connect(address);
    if client.hello["hip"]["available"] != true {
        eprintln!("skipping: {}", client.hello["hip"]["reason"]);
        return;
    }
    let mut scene = generated::template("rally");
    scene["reset_position"] = json!([0, 0]);
    scene["reset_rotation"] = json!(0);
    scene["track"] = json!({
        "path": [[0, 0], [1000, 0], [1000, 1000]],
        "walls": [
            [[-200, -200], [1200, -200]], [[1200, -200], [1200, 1200]],
            [[1200, 1200], [-200, 1200]], [[-200, 1200], [-200, -200]]
        ],
        "polygons": [{"points": [[-200, -200], [1200, -200], [1200, 1200], [-200, 1200]]}],
        "physics_shapes": [], "tiles": []
    });
    let request = json!({
        "scene": scene.to_string(),
        "network": json!({"inputs": ["Speed"], "outputs": ["Acceleration"]}).to_string(),
        "model": json!({"vision": [], "sensors": ["speed"], "outputs": ["acceleration"]}).to_string(),
        "options": {"population": 2, "backend": "hip", "seed": 7}
    });
    let mut created = client.call("create", request.clone());
    // Another test's abrupt socket drop may still be reaching its server thread.
    for attempt in 0.. {
        if created["backendNote"] != "another session is using the GPU" {
            break;
        }
        assert!(attempt < 50, "the previous connection retained HIP: {created}");
        drop(client);
        std::thread::sleep(std::time::Duration::from_millis(20));
        client = Client::connect(address);
        created = client.call("create", request.clone());
    }
    assert_eq!(created["backend"], "cpu", "{created}");
    assert!(created["backendNote"].as_str().unwrap().contains("physics shapes"));
    client.call("startWithShape", json!({"shape": [1, 2, 1]}));
    assert_eq!(
        client.call("advance", json!({"ticks": 6, "stopWhenInactive": false})),
        6
    );
    assert!(!client.call_raw("checkpointBytes", Value::Null, None).1.is_empty());

    let mut wide = Client::connect(address);
    assert_eq!(
        wide.create(&json!({"population": 4, "backend": "hip", "seed": 3}))["backend"],
        "hip"
    );
    wide.call("startWithShape", json!({"shape": [20, 32, 5]}));
    let (reply, _) = wide.call_raw("advance", json!({"ticks": 30, "stopWhenInactive": false}), None);
    assert_eq!(reply["ok"], 30, "{reply}");
    assert_eq!(reply["state"]["backend"], "cpu", "{reply}");
    assert!(reply["state"]["backendNote"].as_str().unwrap().contains("16"));
    assert!(!wide.call_raw("checkpointBytes", Value::Null, None).1.is_empty());
}

#[test]
fn single_frame_requests_above_sixteen_mib_reach_dispatch() {
    let mut client = Client::connect(start());
    // Browser WebSocket.send cannot select continuation framing. Exercise
    // both text and binary as one frame, as Firefox sends them.
    let padding = " ".repeat(17 * 1024 * 1024);
    let request = format!("{}{}", json!({"id": 91, "op": "unknown", "args": {}}), padding);
    client.socket.send(Message::text(request)).unwrap();
    let (reply, _) = client.read();
    assert_eq!(reply["id"], 91);
    assert!(reply["error"].as_str().unwrap().contains("unknown operation"));
    let (reply, _) = client.call_raw("unknown", Value::Null, Some(padding.as_bytes()));
    assert!(reply["error"].as_str().unwrap().contains("takes no binary payload"));
}

#[test]
fn oversized_frame_is_rejected_before_its_payload_arrives() {
    use std::io::{Read, Write};
    let address = start();
    let mut client = Client::connect(address);
    // Masked binary frame with an oversized announced payload and no body.
    // The server must reject its header rather than wait/allocate for the body.
    let len = altd_sim::server::MAX_FRAME_BYTES as u64 + 1;
    let mut header = vec![0x82, 0xff];
    header.extend_from_slice(&len.to_be_bytes());
    header.extend_from_slice(&[0; 4]);
    client.socket.get_mut().write_all(&header).unwrap();
    let mut byte = [0];
    match client.socket.get_mut().read(&mut byte) {
        Ok(0) => {}
        Err(e) if e.kind() == std::io::ErrorKind::ConnectionReset => {}
        other => panic!("oversized frame did not close: {other:?}"),
    }
    assert_eq!(Client::connect(address).hello["protocol"], 1);
}

#[test]
fn fragmented_messages_obey_the_assembled_limit() {
    use std::io::Cursor;
    use tungstenite::protocol::{Role, WebSocketConfig};
    // Two individually allowed masked fragments exceed the assembled limit.
    let mut frames = vec![0x02, 0x83, 0, 0, 0, 0, 1, 2, 3];
    frames.extend_from_slice(&[0x80, 0x83, 0, 0, 0, 0, 4, 5, 6]);
    let config = WebSocketConfig::default()
        .max_frame_size(Some(4))
        .max_message_size(Some(5));
    let mut socket = WebSocket::from_raw_socket(Cursor::new(frames), Role::Server, Some(config));
    assert!(matches!(socket.read(), Err(tungstenite::Error::Capacity(_))));
}
/// Exercise the actual CLI process, not a thread with a test-only unwind profile.
/// `cargo test --release --test server malformed_nested_inputs` also runs the
/// production panic=abort executable.
#[test]
fn malformed_nested_inputs_return_errors_without_aborting_cli() {
    use std::io::{BufRead, BufReader};
    use std::process::{Child, Command, Stdio};
    struct Stop(Child);
    impl Drop for Stop {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let child = Command::new(env!("CARGO_BIN_EXE_altd-sim"))
        .args(["--threads", "1", "serve", "--port", "0", "--allow-origin", ORIGIN])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stop = Stop(child);
    let stderr = stop.0.stderr.take().unwrap();
    let mut stderr = BufReader::new(stderr);
    let address = (&mut stderr)
        .lines()
        .map(Result::unwrap)
        .find_map(|line| {
            line.strip_prefix("altd-sim serve: listening on ws://")
                .and_then(|s| s.strip_suffix("/v1"))
                .map(|s| s.parse::<SocketAddr>().unwrap())
        })
        .expect("server listening address");
    let scene = generated::scene("formula", 0);
    let network = generated::network("formula");
    let model = generated::model("formula");
    let request = |scene: &Value, network: &Value, model: &Value, options: Value| {
        json!({
            "scene": scene.to_string(), "network": network.to_string(),
            "model": model.to_string(), "options": options,
        })
    };
    let mut client = Client::connect(address);
    for bad in [
        json!({}),
        {
            let mut v = scene.clone();
            v["vehicle"]["shape_size"] = json!([1]);
            v
        },
        {
            let mut v = scene.clone();
            v["track"]["path_forward"] = json!([[0, 1]]);
            v
        },
        {
            let mut v = scene.clone();
            v["track"]["curve"]["points"] = json!([]);
            v
        },
    ] {
        assert!(!client
            .error("create", request(&bad, &network, &model, json!({})))
            .is_empty());
    }
    let mut bad_network = network.clone();
    bad_network["inputs"][0] = json!("unknown sensor");
    assert!(client
        .error("create", request(&scene, &bad_network, &model, json!({})))
        .contains("sensor"));
    let mut bad_outputs = network.clone();
    bad_outputs["outputs"][0] = json!("invalid output");
    assert!(client
        .error("create", request(&scene, &bad_outputs, &model, json!({})))
        .contains("control output"));
    assert!(client
        .error(
            "create",
            request(
                &scene,
                &network,
                &model,
                json!({
                    "settings": {"selection_algorithm": "unknown"}
                })
            )
        )
        .contains("selection"));
    client.create(&json!({"population": 4}));
    client.call("startWithShape", json!({"shape": [20, 8, 5]}));
    assert!(!client.error("replaceTrack", json!({"scene": "{}"})).is_empty());

    let original = direct(&json!({"population": 4}));
    let mut original = original;
    original.start_with_shape(&[20, 8, 5]).unwrap();
    let mut checkpoint = json!({
        "generation": 0, "shape": [20, 8, 5],
        "networks": original.runner.agents.iter().map(|a| &a.network.params).collect::<Vec<_>>(),
        "rng": original.runner.rng.to_json(),
    });
    let valid_checkpoint = checkpoint.clone();
    checkpoint["rng"]["state"] = json!([1]);
    assert!(client
        .error("restoreCheckpointJson", json!({"json": checkpoint.to_string()}))
        .contains("state"));

    // Both binary checkpoint formats carry the same JSON RNG inside an envelope.
    for parent_format in [false, true] {
        if parent_format {
            original.advance_generation(1).unwrap();
            original.next_generation().unwrap();
        }
        let bytes = if parent_format {
            original.checkpoint_bytes().unwrap()
        } else {
            client.call("restoreCheckpointJson", json!({"json": valid_checkpoint.to_string()}));
            client.call_raw("checkpointBytes", Value::Null, None).1
        };
        let word = |i: usize| u32::from_le_bytes(bytes[8 + i * 4..12 + i * 4].try_into().unwrap()) as usize;
        let layers = word(3);
        let rng_len = word(4);
        let rng_at = if parent_format {
            48 + 4 * (layers + word(6) + word(7))
        } else {
            28 + 4 * layers
        };
        let old_end = (rng_at + rng_len).next_multiple_of(8);
        let mut rng: Value = serde_json::from_slice(&bytes[rng_at..rng_at + rng_len]).unwrap();
        rng["state"] = json!([1]);
        let rng = rng.to_string();
        let mut bad = bytes[..rng_at].to_vec();
        bad[24..28].copy_from_slice(&(rng.len() as u32).to_le_bytes());
        bad.extend_from_slice(rng.as_bytes());
        bad.resize(bad.len().next_multiple_of(8), 0);
        bad.extend_from_slice(&bytes[old_end..]);
        let (reply, _) = client.call_raw("restoreCheckpointBytes", Value::Null, Some(&bad));
        assert!(reply["error"].as_str().unwrap().contains("state"), "{reply}");
        // A rejected restore did not destroy the existing population.
        assert!(!client.call_raw("checkpointBytes", Value::Null, None).1.is_empty());
    }
    assert!(stop.0.try_wait().unwrap().is_none());
    assert_eq!(Client::connect(address).hello["protocol"], 1);
}
