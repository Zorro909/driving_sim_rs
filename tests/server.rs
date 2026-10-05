//! `altd-sim serve` over real sockets: handshake checks, the greeting, and
//! a remote session that matches a direct `Session` bit for bit.
#![cfg(all(feature = "server", not(target_arch = "wasm32")))]

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

fn direct(options: &Value) -> Session {
    Session::new(
        &generated::scene("formula", 0),
        &generated::network("formula"),
        &generated::model("formula"),
        SessionOptions::from_json(&options.to_string()).unwrap(),
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
}

#[test]
fn server_configuration_normalizes_and_rejects_invalid_origins() {
    let server = Server::bind(Config {
        port: 0,
        origins: vec!["HTTP://127.0.0.1:5173/".to_string()],
    })
    .unwrap();
    let address = server.local_addr().unwrap();
    std::thread::spawn(move || server.run());
    assert!(handshake(address, "/v1", Some(ORIGIN), None).is_ok());
    let error = Server::bind(Config {
        port: 0,
        origins: vec!["null".to_string()],
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
    assert_eq!(client.create(&options), json!({"backend": "cpu", "backendNote": null}));
    let shape = [20, 8, 5];
    let (reply, _) = client.call_raw("startWithShape", json!({"shape": shape}), None);
    session.start_with_shape(&shape).unwrap();
    assert_eq!(
        reply["state"],
        json!({"started": true, "generation": 0, "tick": 0, "population": 8, "activeCount": 8,
            "checkpointGeneration": 0, "backend": "cpu", "backendNote": null})
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
    assert_eq!(client.create(&options), json!({"backend": "hip", "backendNote": null}));
    let mut second = Client::connect(address);
    assert_eq!(
        second.create(&options),
        json!({"backend": "cpu", "backendNote": "another session is using the GPU"})
    );
    for c in [&mut client, &mut second] {
        c.call("startWithShape", json!({"shape": [20, 8, 5]}));
        c.call("advanceGeneration", json!({"timeLimitTicks": 300}));
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
