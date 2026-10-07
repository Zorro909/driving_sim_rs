//! `altd-sim serve`: a WebSocket server on the loopback address that Drive
//! Lab uses as a remote simulation (docs/server.md). Each connection gets a
//! thread and one `Session`; requests run in order.

pub mod dispatch;
pub mod protocol;

use crate::math::profile::MathProfile;
use dispatch::{Connection, Reply};
use protocol::{decode_binary, encode_binary, host_allowed, normalize_origin, origin_allowed, Request, PROTOCOL};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
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
/// Finite transport ceilings, matching Tungstenite's supported defaults.
pub const MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;
/// Includes sockets that have not yet completed the handshake.
pub const MAX_CONNECTIONS: usize = 64;
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);
/// Applies to socket I/O, not computation between requests.
pub const IDLE_TIMEOUT: Duration = Duration::from_secs(600);

struct ConnectionPermit(Arc<AtomicUsize>);

impl ConnectionPermit {
    fn acquire(active: &Arc<AtomicUsize>) -> Option<Self> {
        active
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < MAX_CONNECTIONS).then_some(n + 1)
            })
            .ok()
            .map(|_| Self(active.clone()))
    }
}

impl Drop for ConnectionPermit {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

// A deadline across all handshake I/O prevents drip-fed bytes from resetting
// a per-read timeout forever. Established sockets retain their idle timeout.
struct ConnectionStream {
    stream: TcpStream,
    handshake_deadline: Option<Instant>,
}

impl ConnectionStream {
    fn timeout(&self) -> std::io::Result<Duration> {
        match self.handshake_deadline {
            Some(deadline) => deadline
                .checked_duration_since(Instant::now())
                .filter(|remaining| !remaining.is_zero())
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::TimedOut, "handshake deadline")),
            None => Ok(IDLE_TIMEOUT),
        }
    }
}

impl Read for ConnectionStream {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.stream.set_read_timeout(Some(self.timeout()?))?;
        self.stream.read(buffer)
    }
}

impl Write for ConnectionStream {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.stream.set_write_timeout(Some(self.timeout()?))?;
        self.stream.write(buffer)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.stream.flush()
    }
}

pub struct Config {
    /// TCP port on 127.0.0.1; 0 picks a free one.
    pub port: u16,
    /// Allowed HTTP(S) origins; `bind` validates and normalizes them.
    pub origins: Vec<String>,
    /// The math profile of sessions whose options name none; None detects it.
    pub math_profile: Option<MathProfile>,
}

/// A bound server; `run` accepts connections.
pub struct Server {
    listener: TcpListener,
    origins: Arc<Vec<String>>,
    math: MathProfile,
}

/// HIP availability as `hello` reports it.
fn hip_status() -> Value {
    match crate::training::session::hip_device() {
        Ok((gpu, device)) => json!({"available": true, "device": device, "platform": gpu.platform()}),
        Err(reason) => json!({"available": false, "reason": reason}),
    }
}

fn hello(math: MathProfile) -> Value {
    json!({
        "type": "hello",
        "protocol": PROTOCOL,
        "version": crate::VERSION,
        "threads": rayon::current_num_threads(),
        "carStateStride": crate::training::CAR_STATE_STRIDE,
        "carStateFields": crate::training::CAR_STATE_FIELDS,
        "hip": hip_status(),
        "mathProfile": math,
        "mathProfiles": MathProfile::ALL,
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
            math: config.math_profile.unwrap_or_else(MathProfile::detect),
        })
    }

    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    /// The math profile of sessions whose options name none.
    pub fn math_profile(&self) -> MathProfile {
        self.math
    }

    /// Whether HIP sessions can run, as `hello` reports it.
    pub fn hip_status(&self) -> Value {
        hip_status()
    }

    /// Accepts connections until the listener fails.
    pub fn run(self) -> std::io::Result<()> {
        let port = self.local_addr()?.port();
        let active = Arc::new(AtomicUsize::new(0));
        for stream in self.listener.incoming() {
            let stream = match stream {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("altd-sim serve: accept failed: {e}");
                    continue;
                }
            };
            let Some(permit) = ConnectionPermit::acquire(&active) else {
                drop(stream);
                continue;
            };
            let (origins, math) = (self.origins.clone(), self.math);
            if let Err(error) = std::thread::Builder::new()
                .name("altd-sim-connection".into())
                .spawn(move || {
                    let _permit = permit;
                    serve_connection(stream, &origins, port, math);
                })
            {
                // Dropping the failed spawn's closure releases its permit/socket.
                eprintln!("altd-sim serve: connection spawn failed: {error}");
            }
        }
        Ok(())
    }
}

fn serve_connection(stream: TcpStream, origins: &[String], port: u16, math: MathProfile) {
    let peer = stream.peer_addr().map_or_else(|_| "?".into(), |a| a.to_string());
    let _ = stream.set_nodelay(true);
    if stream.set_read_timeout(Some(HANDSHAKE_TIMEOUT)).is_err()
        || stream.set_write_timeout(Some(HANDSHAKE_TIMEOUT)).is_err()
    {
        return;
    }
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
    // Check frames and assembled messages before application deserialization.
    let config = WebSocketConfig::default()
        .max_message_size(Some(MAX_MESSAGE_BYTES))
        .max_frame_size(Some(MAX_FRAME_BYTES));
    let stream = ConnectionStream {
        stream,
        handshake_deadline: Some(Instant::now() + HANDSHAKE_TIMEOUT),
    };
    let mut socket = match tungstenite::accept_hdr_with_config(stream, callback, Some(config)) {
        Ok(s) => s,
        Err(_) => return,
    };
    socket.get_mut().handshake_deadline = None;
    eprintln!("altd-sim serve: connected {peer} (origin {origin})");
    match converse(&mut socket, math) {
        Ok(()) => eprintln!("altd-sim serve: {peer} disconnected"),
        Err(e) => eprintln!("altd-sim serve: {peer} closed: {e}"),
    }
}

fn send(socket: &mut WebSocket<ConnectionStream>, reply: Reply) -> tungstenite::Result<()> {
    match reply {
        Reply::Text(v) => socket.send(Message::text(v.to_string())),
        Reply::Binary(header, payload) => socket.send(Message::binary(encode_binary(&header, &payload))),
    }
}

/// Greets the client and answers its requests until it closes. The
/// session drops with the connection.
fn converse(socket: &mut WebSocket<ConnectionStream>, math: MathProfile) -> Result<(), String> {
    let mut connection = Connection::new(math);
    socket
        .send(Message::text(hello(math).to_string()))
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

#[cfg(test)]
mod connection_limit_tests {
    use super::*;

    #[test]
    fn overload_does_not_consume_a_permit_and_drop_releases_it() {
        let active = Arc::new(AtomicUsize::new(0));
        let mut permits: Vec<_> = (0..MAX_CONNECTIONS)
            .map(|_| ConnectionPermit::acquire(&active).unwrap())
            .collect();
        assert!(ConnectionPermit::acquire(&active).is_none());
        assert_eq!(active.load(Ordering::Acquire), MAX_CONNECTIONS);
        permits.pop();
        assert!(ConnectionPermit::acquire(&active).is_some());
        drop(permits);
        assert_eq!(active.load(Ordering::Acquire), 0);
    }

    #[test]
    fn rejected_or_incomplete_handshake_drops_its_socket() {
        use std::io::Read;
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        client.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let stream = listener.accept().unwrap().0;
        let worker = std::thread::spawn(move || serve_connection(stream, &[], 0, MathProfile::Proton));
        let mut byte = [0];
        assert_eq!(client.read(&mut byte).unwrap(), 0);
        worker.join().unwrap();
    }
}
