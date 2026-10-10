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
        self.create_with(&generated::network("formula"), options)
    }

    fn create_with(&mut self, network: &Value, options: &Value) -> Value {
        self.call(
            "create",
            json!({
                "scene": generated::scene("formula", 0).to_string(),
                "network": network.to_string(),
                "model": generated::model("formula").to_string(),
                "options": options,
            }),
        )
    }
}

/// A local session with the options of a remote one, on the server's
/// default math profile unless the options name one.
fn direct(options: &Value) -> Session {
    direct_with(&generated::network("formula"), options)
}

fn direct_with(network: &Value, options: &Value) -> Session {
    let mut options = SessionOptions::from_json(&options.to_string()).unwrap();
    options.math_profile.get_or_insert(MathProfile::Proton);
    Session::new(
        &generated::scene("formula", 0),
        network,
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

/// `start` uses the supplied network for both optimizers, reproducing native
/// checkpoints and subsequent generations exactly.
#[test]
fn start_seeds_the_first_generation_from_the_created_network() {
    let export = generated::network_export("formula");
    let seed = Network::from_game_export(&export);
    let address = start();
    for (settings, magic) in [
        (
            json!({"algorithm": "ga", "selection_size": 2, "mutation_rate": 0.0, "weight_decay": 0.0}),
            b"ALTDCKP2",
        ),
        (
            json!({"algorithm": "ars", "ars": {"elite_count": 1, "nu": 0.02, "max_weight": 0.0}}),
            b"ALTDCKP3",
        ),
    ] {
        let options = json!({"population": 6, "seed": 3, "settings": settings});
        let mut client = Client::connect(address);
        let mut session = direct_with(&export, &options);
        client.create_with(&export, &options);
        let (reply, _) = client.call_raw("start", Value::Null, None);
        assert!(reply.get("error").is_none(), "{reply}");
        session.start().unwrap();
        assert_eq!(reply["state"]["started"], true);
        assert_eq!(reply["state"]["population"], 6);
        assert_eq!(reply["state"]["checkpointGeneration"], 0);
        for index in 0..6 {
            let network: Value =
                serde_json::from_str(client.call("networkJson", json!({"index": index})).as_str().unwrap()).unwrap();
            let network = Network::from_game_export(&network);
            assert_eq!(network.shape, seed.shape);
            assert_eq!(
                bits(&network.params),
                bits(&session.runner.agents[index].network.params)
            );
            // GA with mutation/decay disabled preserves every seed. ARS's
            // first car is the unchanged search point, before its probes.
            if settings["algorithm"] == "ga" || index == 0 {
                assert_eq!(bits(&network.params), bits(&seed.params));
            }
        }
        let checkpoint = client.call_raw("checkpointBytes", Value::Null, None).1;
        assert_eq!(&checkpoint[..8], magic);
        assert_eq!(checkpoint, session.checkpoint_bytes().unwrap());
        assert_eq!(
            client.call("advanceGeneration", json!({"timeLimitTicks": 120})),
            json!(session.advance_generation(120).unwrap())
        );
        let (preserved, rewards) = session.next_generation().unwrap();
        assert_eq!(
            client.call("nextGeneration", Value::Null),
            json!({"preservedCount": preserved, "rewards": rewards})
        );
        assert_eq!(
            client.call_raw("checkpointBytes", Value::Null, None).1,
            session.checkpoint_bytes().unwrap()
        );
    }
}

#[test]
fn start_uses_the_template_shape_and_recovers_when_it_is_missing() {
    let options = json!({"population": 6, "seed": 3});
    let address = start();
    for nested in [false, true] {
        let mut export = generated::network("formula");
        if nested {
            export["summary"] = json!({"shape": [20, 8, 5]});
        } else {
            export["shape"] = json!([20, 8, 5]);
        }
        let mut client = Client::connect(address);
        client.create_with(&export, &options);
        client.call("start", Value::Null);
        let mut session = direct(&options);
        session.start_with_shape(&[20, 8, 5]).unwrap();
        assert_eq!(
            client.call_raw("checkpointBytes", Value::Null, None).1,
            session.checkpoint_bytes().unwrap()
        );
    }
    let mut bare = Client::connect(address);
    bare.create(&options);
    let error = bare.error("start", Value::Null);
    assert!(error.contains("start_with_shape"), "{error}");
    bare.call("startWithShape", json!({"shape": [20, 8, 5]}));
    assert!(bare.call("generationSummary", Value::Null).is_object());
}

#[test]
fn start_rejects_malformed_weights_without_starting_or_consuming_rng() {
    let options = json!({"population": 6, "seed": 3});
    let export = generated::network_export("formula");
    let address = start();
    for (name, value) in [
        ("weights", Value::Null),
        ("weights", json!([])),
        ("biases", Value::Null),
        ("biases", json!([])),
        ("shape", json!([20, u64::MAX, 5])),
        // This fits the buffer arithmetic but must reject its tiny weight
        // arrays without trying to reserve the claimed multi-terabyte size.
        ("shape", json!([20, 1u64 << 40, 5])),
        ("shape", json!([19, 8, 5])),
    ] {
        let mut invalid = export.clone();
        invalid[name] = value;
        let mut client = Client::connect(address);
        client.create_with(&invalid, &options);
        let error = client.error("start", Value::Null);
        assert!(!error.is_empty(), "{name}: {invalid}");
        assert!(client
            .error("advance", json!({"ticks": 1, "stopWhenInactive": false}))
            .contains("not started"));
        client.call("startWithShape", json!({"shape": [20, 8, 5]}));
        let mut session = direct(&options);
        session.start_with_shape(&[20, 8, 5]).unwrap();
        assert_eq!(
            client.call_raw("checkpointBytes", Value::Null, None).1,
            session.checkpoint_bytes().unwrap()
        );
    }
    let mut invalid = export;
    invalid["weights"][0][0][0] = json!("bad parameter");
    let mut client = Client::connect(address);
    client.create_with(&invalid, &options);
    assert!(client
        .error("start", Value::Null)
        .contains("parameters must be numbers"));
    client.call("startWithShape", json!({"shape": [20, 8, 5]}));
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
        .args([
            "--threads",
            "1",
            "--math-profile",
            "proton",
            "serve",
            "--port",
            "0",
            "--allow-origin",
            ORIGIN,
        ])
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
    let mut bad_model = model.clone();
    bad_model["vision"][0]["length"] = json!(-1);
    assert!(client
        .error("create", request(&scene, &network, &bad_model, json!({})))
        .contains("ray"));
    let mut bad_wheels = scene.clone();
    for w in bad_wheels["vehicle"]["wheels"].as_array_mut().unwrap() {
        w["steering"] = json!(false);
    }
    assert!(client
        .error("create", request(&bad_wheels, &network, &model, json!({})))
        .contains("steering wheel"));

    for spawn in [
        json!({"position": [1e300, 0], "rotation": 0}),
        json!({"position": [0, -1e300], "rotation": 0}),
        json!({"position": [0, 0], "rotation": 1e300}),
    ] {
        assert!(client
            .error("create", request(&scene, &network, &model, json!({"spawn": spawn})))
            .contains("float32"));
    }

    let mut invalid_tiles = Vec::new();
    for coords in [
        json!([[-1e30, 0], [1e30, 0]]),
        json!([[-9e18, 0], [9e18, 0]]),
        json!([[0, 0], [4e9, 4e9]]),
        json!([[0, 0], [1e9, 1e9]]),
    ] {
        let mut bad = scene.clone();
        bad["track"]["tiles"] = json!([
            {"coords": coords[0], "surface": "asphalt"},
            {"coords": coords[1], "surface": "asphalt"},
        ]);
        invalid_tiles.push(bad);
    }
    let mut invalid_order = scene.clone();
    invalid_order["track"]["native_cell_order"] = json!([[-1e30, 0], [1e30, 0]]);
    invalid_tiles.push(invalid_order);
    for bad in &invalid_tiles {
        assert!(!client
            .error("create", request(bad, &network, &model, json!({})))
            .is_empty());
    }

    for (name, kind, maximum) in [
        ("AccF", "accelerationFront", "MaxAcceleration"),
        ("AccS", "accelerationSide", "MaxAcceleration"),
        ("Wall", "distanceFromWall", "MaxDistance"),
    ] {
        let mut sensor_network = network.clone();
        sensor_network["inputs"] = json!([name]);
        let mut sensor_model = model.clone();
        sensor_model["sensor_layout"] = json!({"names": [name], "sensors": [{"$type": kind}]});
        for value in [0.0, -0.0, -1.0, 1e-50] {
            sensor_model["sensor_layout"]["sensors"][0][maximum] = json!(value);
            assert!(client
                .error("create", request(&scene, &sensor_network, &sensor_model, json!({})))
                .contains(maximum));
        }
        sensor_model["sensor_layout"]["sensors"][0][maximum] = json!(100.0);
        let mut positive = Client::connect(address);
        positive.call(
            "create",
            request(&scene, &sensor_network, &sensor_model, json!({"population": 4})),
        );
        positive.call("startWithShape", json!({"shape": [1, 2, 5]}));
        positive.call("advance", json!({"ticks": 6, "stopWhenInactive": false}));
        let mut direct = Session::new(
            &scene,
            &sensor_network,
            &sensor_model,
            SessionOptions::from_json(r#"{"population":4,"mathProfile":"proton"}"#).unwrap(),
        )
        .unwrap();
        direct.start_with_shape(&[1, 2, 5]).unwrap();
        direct.advance(6, false).unwrap();
        assert!(direct.sensors(0).unwrap().iter().all(|v| v.is_finite()));
        assert_eq!(
            positive.call_raw("checkpointBytes", Value::Null, None).1,
            direct.checkpoint_bytes().unwrap()
        );
    }

    client.create(&json!({"population": 4}));
    client.call("startWithShape", json!({"shape": [20, 8, 5]}));
    assert!(!client.error("replaceTrack", json!({"scene": "{}"})).is_empty());
    let before = client.call_raw("checkpointBytes", Value::Null, None).1;
    let mut changed_wheels = scene.clone();
    let wheels = changed_wheels["vehicle"]["wheels"].as_array_mut().unwrap();
    wheels.push(wheels[0].clone());
    for bad in [&bad_wheels, &changed_wheels] {
        assert!(client
            .error("replaceTrack", json!({"scene": bad.to_string()}))
            .contains("keep the vehicle"));
        assert_eq!(client.call_raw("checkpointBytes", Value::Null, None).1, before);
    }
    for bad in &invalid_tiles {
        assert!(!client
            .error("replaceTrack", json!({"scene": bad.to_string()}))
            .is_empty());
        assert_eq!(client.call_raw("checkpointBytes", Value::Null, None).1, before);
    }
    let replacement = generated::scene("formula", 1);
    client.call("replaceTrack", json!({"scene": replacement.to_string()}));
    client.call("advance", json!({"ticks": 6, "stopWhenInactive": false}));

    let spawn = generated::spawn(&scene);
    let mut spawn_client = Client::connect(address);
    spawn_client.create(&json!({"population": 4, "spawn": spawn}));
    spawn_client.call("startWithShape", json!({"shape": [20, 8, 5]}));
    let mut spawn_direct = direct(&json!({"population": 4, "spawn": spawn}));
    spawn_direct.start_with_shape(&[20, 8, 5]).unwrap();
    assert_eq!(
        spawn_client.call_raw("checkpointBytes", Value::Null, None).1,
        spawn_direct.checkpoint_bytes().unwrap()
    );

    let mut negative_tiles = scene.clone();
    negative_tiles["track"]["tiles"] = json!([
        {"coords": [-1, -1], "surface": "asphalt"},
        {"coords": [1, 1], "surface": "asphalt"},
    ]);
    let mut tile_client = Client::connect(address);
    tile_client.call(
        "create",
        request(&negative_tiles, &network, &model, json!({"population": 4})),
    );
    tile_client.call("startWithShape", json!({"shape": [20, 8, 5]}));
    tile_client.call("advance", json!({"ticks": 6, "stopWhenInactive": false}));

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

    // All binary checkpoint formats carry the same JSON RNG inside an envelope.
    // Population options override settings before ARS's minimum-size validation.
    let mut ars = direct(&json!({
        "population": 16,
        "settings": {"algorithm": "ars", "population": 4},
    }));
    ars.start_with_shape(&[20, 8, 5]).unwrap();
    let ars_bytes = ars.checkpoint_bytes().unwrap();
    assert_eq!(&ars_bytes[..8], b"ALTDCKP3");
    let mut ars_client = Client::connect(address);
    ars_client.create(&json!({
        "population": 16,
        "settings": {"algorithm": "ars", "population": 4},
    }));
    ars_client.call("startWithShape", json!({"shape": [20, 8, 5]}));
    assert_eq!(ars_client.call_raw("checkpointBytes", Value::Null, None).1, ars_bytes);

    for format in [1, 2, 3] {
        if format == 2 {
            original.advance_generation(1).unwrap();
            original.next_generation().unwrap();
        }
        let bytes = match format {
            1 => {
                client.call("restoreCheckpointJson", json!({"json": valid_checkpoint.to_string()}));
                client.call_raw("checkpointBytes", Value::Null, None).1
            }
            2 => original.checkpoint_bytes().unwrap(),
            3 => ars_bytes.clone(),
            _ => unreachable!(),
        };
        let (reply, _) = client.call_raw("restoreCheckpointBytes", Value::Null, Some(&bytes));
        assert!(reply.get("error").is_none(), "{reply}");
        assert_eq!(client.call_raw("checkpointBytes", Value::Null, None).1, bytes);
        let word = |i: usize| u32::from_le_bytes(bytes[8 + i * 4..12 + i * 4].try_into().unwrap()) as usize;
        let layers = word(3);
        let rng_len = word(4);
        let rng_at = match format {
            1 => 28 + 4 * layers,
            2 => 48 + 4 * (layers + word(6) + word(7)),
            3 => 36 + 4 * layers,
            _ => unreachable!(),
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
        // A rejected restore preserves the exact checkpoint and a usable session.
        assert_eq!(client.call_raw("checkpointBytes", Value::Null, None).1, bytes);
        client.call("advance", json!({"ticks": 6, "stopWhenInactive": false}));
    }

    ars.advance_generation(1).unwrap();
    ars.next_generation().unwrap();
    let bytes = ars.checkpoint_bytes().unwrap();
    let word = |i: usize| u32::from_le_bytes(bytes[8 + i * 4..12 + i * 4].try_into().unwrap()) as usize;
    assert!(word(5) > 0, "the checkpoint must contain an elite pool");
    let size = ars.runner.agents[0].network.params.len();
    let floats_at = (36 + 4 * word(3) + word(4)).next_multiple_of(8);
    let theta_at = floats_at + 16;
    let elite_at = theta_at + size * 8;
    let score_at = elite_at + size * 8;
    let (reply, _) = ars_client.call_raw("restoreCheckpointBytes", Value::Null, Some(&bytes));
    assert!(reply.get("error").is_none(), "{reply}");
    assert_eq!(ars_client.call_raw("checkpointBytes", Value::Null, None).1, bytes);
    for at in [theta_at, elite_at, score_at] {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut bad = bytes.clone();
            bad[at..at + 8].copy_from_slice(&value.to_le_bytes());
            let (reply, _) = ars_client.call_raw("restoreCheckpointBytes", Value::Null, Some(&bad));
            assert!(
                reply["error"].as_str().is_some_and(|e| e.contains("non-finite")),
                "{reply}"
            );
            assert_eq!(ars_client.call_raw("checkpointBytes", Value::Null, None).1, bytes);
        }
    }
    ars_client.call("advance", json!({"ticks": 6, "stopWhenInactive": false}));
    assert!(stop.0.try_wait().unwrap().is_none());
    assert_eq!(Client::connect(address).hello["protocol"], 1);
}

#[test]
fn partial_settings_are_charged_before_mutation() {
    let mut client = Client::connect(start());
    client.create(&json!({"population":4, "backend":"cpu"}));
    client.call("startWithShape", json!({"shape":[20,64,64,5]}));
    let before = client.call_raw("checkpointBytes", Value::Null, None).1;
    assert!(client
        .error("setEvolutionSettings", json!({"json":"{\"mutation_rate\":0.1}"}))
        .contains("budget"));
    assert_eq!(before, client.call_raw("checkpointBytes", Value::Null, None).1);
    client.call("nextGeneration", Value::Null);
    assert_eq!(
        client.call_raw("generationSummary", Value::Null, None).0["state"]["population"],
        4
    );

    let mut small = Client::connect(start());
    small.create(&json!({"population":4, "backend":"cpu"}));
    small.call("startWithShape", json!({"shape":[20,8,5]}));
    small.call("setEvolutionSettings", json!({"json":"{\"mutation_rate\":0.1}"}));
    small.call("nextGeneration", Value::Null);
    assert_eq!(
        small.call_raw("generationSummary", Value::Null, None).0["state"]["population"],
        300
    );
    assert!(small
        .error("advance", json!({"ticks":3600,"stopWhenInactive":false}))
        .contains("work budget"));
}

#[test]
fn generation_admission_uses_remaining_window() {
    let mut client = Client::connect(start());
    client.create(&json!({"population":4,"backend":"cpu"}));
    client.call("startWithShape", json!({"shape":[20,8,5]}));
    let before = client.call_raw("checkpointBytes", Value::Null, None).1;
    assert!(client
        .error("advanceGeneration", json!({"timeLimitTicks":3600}))
        .contains("work budget"));
    assert!(client
        .error("advanceGeneration", json!({"timeLimitTicks":u64::MAX}))
        .contains("work budget"));
    assert_eq!(before, client.call_raw("checkpointBytes", Value::Null, None).1);
    client.call("advance", json!({"ticks":12,"stopWhenInactive":false}));
    client.call("advanceGeneration", json!({"timeLimitTicks":3600}));
}

#[test]
fn hip_grid_budget_is_checked_before_backend_preparation() {
    let mut client = Client::connect(start());
    let mut scene = generated::scene("formula", 0);
    let path: Vec<_> = (0..16384).map(|i| json!([i % 2, 0])).collect();
    scene["track"]["path"] = json!(path);
    let request = json!({
        "scene":scene.to_string(),"network":generated::network("formula").to_string(),
        "model":generated::model("formula").to_string(),"options":{"population":4,"backend":"hip"}
    });
    assert!(client.error("create", request).contains("grid-entry"));
    // No session was installed; a normal CPU request still succeeds.
    client.create(&json!({"population":4,"backend":"cpu"}));
}

#[test]
fn rejected_geometry_and_rng_headers_preserve_checkpoint() {
    let mut client = Client::connect(start());
    client.create(&json!({"population":4,"backend":"cpu"}));
    client.call("startWithShape", json!({"shape":[20,8,5]}));
    let before = client.call_raw("checkpointBytes", Value::Null, None).1;
    let mut scene = generated::scene("formula", 0);
    scene["track"] = json!({"path":[[0,0],[4000,4000]],
        "walls":vec![json!([[0,0],[1,1]]);2048],"raycaster_present":null,"polygons":[{"points":[]}]});
    assert!(client
        .error("replaceTrack", json!({"scene":scene.to_string()}))
        .contains("grid-entry"));
    assert_eq!(before, client.call_raw("checkpointBytes", Value::Null, None).1);
    for magic in [b"ALTDCKP1", b"ALTDCKP2"] {
        let mut bytes = before.clone();
        bytes[..8].copy_from_slice(magic);
        bytes[24..28].copy_from_slice(&(16 * 1024 * 1024 + 1u32).to_le_bytes());
        let (reply, _) = client.call_raw("restoreCheckpointBytes", Value::Null, Some(&bytes));
        assert!(reply["error"].as_str().unwrap().contains("RNG JSON"));
        assert_eq!(before, client.call_raw("checkpointBytes", Value::Null, None).1);
    }
}

#[test]
fn nonzero_statistics_phase_stays_fallible() {
    let mut client = Client::connect(start());
    client.create(&json!({"population":4,"backend":"cpu","statsPhase":1}));
    client.call("startWithShape", json!({"shape":[20,5]}));
    assert!(client
        .error("advanceGeneration", json!({"timeLimitTicks":12}))
        .contains("statistics phase"));
    client.call("advance", json!({"ticks":6,"stopWhenInactive":false}));
}

#[test]
fn resource_limits_reject_work_without_losing_the_session() {
    let address = start();
    let mut client = Client::connect(address);
    let request = json!({
        "scene":generated::scene("formula",0).to_string(),
        "network":generated::network("formula").to_string(),
        "model":generated::model("formula").to_string(),
        "options":{"population":u64::MAX},
    });
    assert!(client.error("create", request).contains("population"));
    client.create(&json!({"population":4}));
    client.call("startWithShape", json!({"shape":[20,8,5]}));
    let before = client.call_raw("checkpointBytes", Value::Null, None).1;
    assert!(client
        .error("startWithShape", json!({"shape":[20,u32::MAX,5]}))
        .contains("server"));
    assert!(client
        .error("advance", json!({"ticks":u64::MAX,"stopWhenInactive":false}))
        .contains("budget"));
    assert!(client
        .error("advanceGeneration", json!({"timeLimitTicks":u64::MAX}))
        .contains("budget"));
    assert!(client
        .error(
            "setEvolutionSettings",
            json!({"json":json!({"population":u64::MAX}).to_string()})
        )
        .contains("population"));
    let mut bad = generated::scene("formula", 0);
    bad["track"]["tiles"] = json!([
        {"coords":[0,0],"surface":"asphalt"},
        {"coords":[1000000,1000000],"surface":"asphalt"}
    ]);
    assert!(client
        .error("replaceTrack", json!({"scene":bad.to_string()}))
        .contains("budget"));
    let mut header = before.clone();
    header[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
    let (reply, _) = client.call_raw("restoreCheckpointBytes", Value::Null, Some(&header));
    assert!(reply["error"].as_str().unwrap().contains("population"));
    assert!(client
        .error(
            "restoreCheckpointJson",
            json!({"json":json!({"shape":[20,u64::MAX,5]}).to_string()})
        )
        .contains("32-bit"));
    assert_eq!(before, client.call_raw("checkpointBytes", Value::Null, None).1);
    assert_eq!(client.call("advance", json!({"ticks":1,"stopWhenInactive":false})), 1);
    assert_eq!(Client::connect(address).hello["protocol"], 1);
}

#[test]
fn ars_checkpoint_admission_restores_and_continues_exactly() {
    let options = json!({"population":16,"seed":4,"settings":{"algorithm":"ars","ars":{"elite_count":2}}});
    let mut source = direct(&options);
    source.start_with_shape(&[20, 8, 5]).unwrap();
    source.advance(12, false).unwrap();
    source.next_generation().unwrap();
    let saved = source.checkpoint_bytes().unwrap();
    assert_eq!(&saved[..8], b"ALTDCKP3");
    assert_eq!(u32::from_le_bytes(saved[28..32].try_into().unwrap()), 2);

    let mut client = Client::connect(start());
    client.create(&options);
    let (reply, _) = client.call_raw("restoreCheckpointBytes", Value::Null, Some(&saved));
    assert!(reply.get("error").is_none(), "{reply}");
    assert_eq!(saved, client.call_raw("checkpointBytes", Value::Null, None).1);
    for (word, value) in [(2, 32_769u32), (5, 32_769), (6, 32_769), (4, 16 * 1024 * 1024 + 1)] {
        let mut bad = saved.clone();
        bad[8 + word * 4..12 + word * 4].copy_from_slice(&value.to_le_bytes());
        let (reply, _) = client.call_raw("restoreCheckpointBytes", Value::Null, Some(&bad));
        assert!(reply["error"].as_str().is_some(), "{reply}");
        assert_eq!(saved, client.call_raw("checkpointBytes", Value::Null, None).1);
    }
    source.advance(12, false).unwrap();
    client.call("advance", json!({"ticks":12,"stopWhenInactive":false}));
    assert_eq!(
        f64s(&client.call_raw("carStates", Value::Null, None).1),
        bits(&source.car_states())
    );
    source.next_generation().unwrap();
    client.call("nextGeneration", Value::Null);
    assert_eq!(
        source.checkpoint_bytes().unwrap(),
        client.call_raw("checkpointBytes", Value::Null, None).1
    );
}

#[test]
fn retained_ars_settings_are_limited_before_session_mutation() {
    let oversized = json!({"algorithm":"ga","ars":{"elite_count":32_769}});
    let mut client = Client::connect(start());
    let request = json!({
        "scene":generated::scene("formula",0).to_string(),
        "network":generated::network("formula").to_string(),
        "model":generated::model("formula").to_string(),
        "options":{"population":16,"settings":oversized},
    });
    assert!(client.error("create", request).contains("elite_count"));
    client.create(&json!({"population":16}));
    client.call("startWithShape", json!({"shape":[20,8,5]}));
    let before = client.call_raw("checkpointBytes", Value::Null, None).1;
    assert!(client
        .error("setEvolutionSettings", json!({"json":oversized.to_string()}))
        .contains("elite_count"));
    assert_eq!(before, client.call_raw("checkpointBytes", Value::Null, None).1);
    client.call(
        "setEvolutionSettings",
        json!({"json":json!({"algorithm":"ars","population":16,"ars":{"elite_count":2}}).to_string()}),
    );
    client.call("nextGeneration", Value::Null);
    assert_eq!(
        &client.call_raw("checkpointBytes", Value::Null, None).1[..8],
        b"ALTDCKP3"
    );
}

#[test]
fn population_override_is_applied_before_server_admission() {
    for underlying in [0, 32_769] {
        let options = json!({
            "population":16,
            "settings":{"algorithm":"ars","population":underlying,"ars":{"elite_count":2}},
        });
        let mut client = Client::connect(start());
        client.create(&options);
        client.call("startWithShape", json!({"shape":[20,8,5]}));
        assert_eq!(
            client.call_raw("generationSummary", Value::Null, None).0["state"]["population"],
            16
        );
    }
    let mut client = Client::connect(start());
    for options in [
        json!({"population":32_769,"settings":{"population":4}}),
        json!({"population":16,"settings":{"population":0,"selection_size":32_769}}),
        json!({"population":16,"settings":{"population":0,"ars":{"elite_count":32_769}}}),
    ] {
        let request = json!({
            "scene":generated::scene("formula",0).to_string(),
            "network":generated::network("formula").to_string(),
            "model":generated::model("formula").to_string(),
            "options":options,
        });
        assert!(client.error("create", request).contains("limit"));
    }
    client.create(&json!({"population":16}));
}

#[test]
fn all_known_reward_metrics_fit_server_admission() {
    let mut rewards: Vec<_> = altd_sim::training::evolution::METRIC_NAMES
        .iter()
        .map(|metric| json!({"metric":metric,"weight":1,"type":"default"}))
        .collect();
    let mut client = Client::connect(start());
    client.create(&json!({"population":16,"settings":{"rewards":rewards}}));
    client.call("startWithShape", json!({"shape":[20,8,5]}));
    client.call("advance", json!({"ticks":6,"stopWhenInactive":false}));
    client.call("nextGeneration", Value::Null);
    let before = client.call_raw("checkpointBytes", Value::Null, None).1;
    rewards.push(rewards[0].clone());
    assert!(client
        .error(
            "setEvolutionSettings",
            json!({"json":json!({"population":16,"rewards":rewards}).to_string()})
        )
        .contains("too many server reward terms"));
    assert_eq!(before, client.call_raw("checkpointBytes", Value::Null, None).1);
}
