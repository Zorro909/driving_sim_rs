//! `altd-sim serve`: a WebSocket server on the loopback address that Drive
//! Lab uses as a remote simulation (docs/server.md). Each connection gets a
//! thread and one `Session`; requests run in order.

pub mod dispatch;
pub mod protocol;

use dispatch::{Connection, Reply};
use protocol::{decode_binary, encode_binary, host_allowed, normalize_origin, origin_allowed, Request, PROTOCOL};
use serde_json::{json, Value};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use tungstenite::handshake::server::{ErrorResponse, Request as HttpRequest, Response as HttpResponse};
use tungstenite::http::StatusCode;
use tungstenite::protocol::WebSocketConfig;
use tungstenite::{Message, WebSocket};

/// The port Drive Lab connects to unless the user changes it.
pub const DEFAULT_PORT: u16 = 47800;
/// The only origin allowed unless `--allow-origin` replaces it.
pub const DEFAULT_ORIGIN: &str = "https://drivinglab.jectrum.de";
/// The WebSocket endpoint.
pub const PATH: &str = "/v1";

pub struct Config {
    /// TCP port on 127.0.0.1; 0 picks a free one.
    pub port: u16,
    /// Allowed HTTP(S) origins; `bind` validates and normalizes them.
    pub origins: Vec<String>,
}

/// A bound server; `run` accepts connections.
pub struct Server {
    listener: TcpListener,
    origins: Arc<Vec<String>>,
}

/// HIP availability as `hello` reports it.
fn hip_status() -> Value {
    match crate::training::session::hip_device() {
        Ok((_, device)) => json!({"available": true, "device": device}),
        Err(reason) => json!({"available": false, "reason": reason}),
    }
}

fn hello() -> Value {
    json!({
        "type": "hello",
        "protocol": PROTOCOL,
        "version": crate::VERSION,
        "threads": rayon::current_num_threads(),
        "carStateStride": crate::training::CAR_STATE_STRIDE,
        "carStateFields": crate::training::CAR_STATE_FIELDS,
        "hip": hip_status(),
    })
}

fn reject(reason: &str) -> ErrorResponse {
    let mut response = ErrorResponse::new(Some(format!("{reason}\n")));
    *response.status_mut() = StatusCode::FORBIDDEN;
    response
}

/// The handshake checks of docs/server.md: path, `Origin` and `Host`.
fn check_request(request: &HttpRequest, origins: &[String], port: u16) -> Result<(), &'static str> {
    if request.uri().path() != PATH {
        return Err("unknown path");
    }
    let header = |name| request.headers().get(name).and_then(|v| v.to_str().ok());
    match header("origin") {
        Some(origin) if origin_allowed(origin, origins) => {}
        _ => return Err("origin not allowed (see altd-sim serve --allow-origin)"),
    }
    match header("host") {
        Some(host) if host_allowed(host, port) => Ok(()),
        _ => Err("host not allowed"),
    }
}

impl Server {
    pub fn bind(config: Config) -> std::io::Result<Server> {
        let origins = config
            .origins
            .iter()
            .map(|origin| normalize_origin(origin))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|reason| std::io::Error::new(std::io::ErrorKind::InvalidInput, reason))?;
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, config.port))?;
        Ok(Server {
            listener,
            origins: Arc::new(origins),
        })
    }

    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    /// Whether HIP sessions can run, as `hello` reports it.
    pub fn hip_status(&self) -> Value {
        hip_status()
    }

    /// Accepts connections until the listener fails.
    pub fn run(self) -> std::io::Result<()> {
        let port = self.local_addr()?.port();
        for stream in self.listener.incoming() {
            let stream = match stream {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("altd-sim serve: accept failed: {e}");
                    continue;
                }
            };
            let origins = self.origins.clone();
            std::thread::Builder::new()
                .name("altd-sim-connection".into())
                .spawn(move || serve_connection(stream, &origins, port))?;
        }
        Ok(())
    }
}

fn serve_connection(stream: TcpStream, origins: &[String], port: u16) {
    let peer = stream.peer_addr().map_or_else(|_| "?".into(), |a| a.to_string());
    let _ = stream.set_nodelay(true);
    let mut origin = String::new();
    // tungstenite's handshake callback requires an unboxed HTTP response.
    #[allow(clippy::result_large_err)]
    let callback = |request: &HttpRequest, response: HttpResponse| {
        origin = request
            .headers()
            .get("origin")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("none")
            .to_owned();
        match check_request(request, origins, port) {
            Ok(()) => Ok(response),
            Err(reason) => {
                eprintln!("altd-sim serve: rejected {peer} (origin {origin}): {reason}");
                Err(reject(reason))
            }
        }
    };
    // Scenes and large populations' checkpoints exceed the default limits.
    let config = WebSocketConfig::default()
        .max_message_size(Some(1 << 30))
        .max_frame_size(Some(1 << 30));
    let mut socket = match tungstenite::accept_hdr_with_config(stream, callback, Some(config)) {
        Ok(s) => s,
        Err(_) => return,
    };
    eprintln!("altd-sim serve: connected {peer} (origin {origin})");
    match converse(&mut socket) {
        Ok(()) => eprintln!("altd-sim serve: {peer} disconnected"),
        Err(e) => eprintln!("altd-sim serve: {peer} closed: {e}"),
    }
}

fn send(socket: &mut WebSocket<TcpStream>, reply: Reply) -> tungstenite::Result<()> {
    match reply {
        Reply::Text(v) => socket.send(Message::text(v.to_string())),
        Reply::Binary(header, payload) => socket.send(Message::binary(encode_binary(&header, &payload))),
    }
}

/// Greets the client and answers its requests until it closes. The
/// session drops with the connection.
fn converse(socket: &mut WebSocket<TcpStream>) -> Result<(), String> {
    let mut connection = Connection::default();
    socket
        .send(Message::text(hello().to_string()))
        .map_err(|e| e.to_string())?;
    loop {
        let message = match socket.read() {
            Ok(m) => m,
            Err(tungstenite::Error::ConnectionClosed | tungstenite::Error::AlreadyClosed) => return Ok(()),
            Err(e) => return Err(e.to_string()),
        };
        let reply = match message {
            Message::Text(text) => match serde_json::from_str::<Request>(&text) {
                Ok(r) => connection.handle(r.id, &r.op, &r.args, None),
                Err(e) => {
                    // Keep the id when there is one, so the client can match the error.
                    let id = serde_json::from_str::<Value>(&text)
                        .ok()
                        .and_then(|v| v.get("id").cloned())
                        .unwrap_or(Value::Null);
                    Reply::Text(json!({"id": id, "error": format!("invalid request: {e}")}))
                }
            },
            Message::Binary(frame) => {
                let request = decode_binary(&frame).and_then(|(header, payload)| {
                    serde_json::from_value::<Request>(header)
                        .map(|r| (r, payload))
                        .map_err(|e| format!("invalid binary frame header: {e}"))
                });
                match request {
                    Ok((r, payload)) => connection.handle(r.id, &r.op, &r.args, Some(payload)),
                    Err(e) => {
                        let _ = socket.close(None);
                        return Err(e);
                    }
                }
            }
            Message::Close(_) => continue,
            Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => continue,
        };
        send(socket, reply).map_err(|e| e.to_string())?;
    }
}
