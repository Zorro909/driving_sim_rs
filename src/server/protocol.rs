//! Messages of `altd-sim serve` (docs/server.md): origin and host checks,
//! requests, and the binary frame layout.

use serde_json::Value;
use std::net::Ipv6Addr;

/// The protocol version `hello` announces.
pub const PROTOCOL: u32 = 1;

/// Normalizes an allowed origin given on the command line to the form
/// browsers send: lowercase `scheme://host[:port]` with the scheme's default
/// port left out. Only `http` and `https` origins without a path are valid.
pub fn normalize_origin(text: &str) -> Result<String, String> {
    let invalid = |why: &str| format!("invalid origin {text:?}: {why} (expected scheme://host[:port])");
    let (scheme, rest) = text.split_once("://").ok_or_else(|| invalid("no scheme"))?;
    let scheme = scheme.to_ascii_lowercase();
    let default_port = match scheme.as_str() {
        "http" => "80",
        "https" => "443",
        _ => return Err(invalid("the scheme must be http or https")),
    };
    let authority = rest.strip_suffix('/').unwrap_or(rest);
    if authority.is_empty() {
        return Err(invalid("no host"));
    }
    if authority.contains(['/', '?', '#', '@']) || authority.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return Err(invalid("only scheme, host and port are allowed"));
    }
    // An IPv6 host is bracketed; the port follows the closing bracket.
    let (host, port) = match authority.strip_prefix('[') {
        Some(inner) => {
            let (host, after) = inner.split_once(']').ok_or_else(|| invalid("unclosed IPv6 bracket"))?;
            let host = host.parse::<Ipv6Addr>().map_err(|_| invalid("invalid IPv6 host"))?;
            let port = match after {
                "" => None,
                p => Some(
                    p.strip_prefix(':')
                        .ok_or_else(|| invalid("garbage after the IPv6 host"))?,
                ),
            };
            // Browsers serialize mapped IPv4 addresses as hexadecimal hextets.
            let host = if host.to_ipv4_mapped().is_some() {
                let segments = host.segments();
                format!("::ffff:{:x}:{:x}", segments[6], segments[7])
            } else {
                host.to_string()
            };
            (format!("[{host}]"), port)
        }
        None => match authority.rsplit_once(':') {
            Some((host, port)) => (host.to_ascii_lowercase(), Some(port)),
            None => (authority.to_ascii_lowercase(), None),
        },
    };
    if host.is_empty()
        || !host.starts_with('[')
            && !host
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'-' | b'_'))
    {
        return Err(invalid("invalid host; use an ASCII hostname or IP address"));
    }
    let port = match port {
        None => None,
        Some(p) => {
            if p.is_empty() || !p.bytes().all(|c| c.is_ascii_digit()) {
                return Err(invalid("the port must be a number up to 65535"));
            }
            let n: u16 = p
                .parse()
                .map_err(|_| invalid("the port must be a number up to 65535"))?;
            (n.to_string() != default_port).then(|| n.to_string())
        }
    };
    Ok(match port {
        Some(port) => format!("{scheme}://{host}:{port}"),
        None => format!("{scheme}://{host}"),
    })
}

/// Whether an `Origin` header value is allowed: it must equal an allowed
/// origin exactly. Opaque origins (`null`) never are.
pub fn origin_allowed(origin: &str, allowed: &[String]) -> bool {
    origin != "null" && allowed.iter().any(|a| a == origin)
}

/// Whether a `Host` header names this server: the loopback address or
/// `localhost` with the listening port. Rejecting other names blocks DNS
/// rebinding.
pub fn host_allowed(host: &str, port: u16) -> bool {
    let Some((name, p)) = host.rsplit_once(':') else {
        return false;
    };
    p == port.to_string() && (name == "127.0.0.1" || name.eq_ignore_ascii_case("localhost"))
}

/// A request: `{"id": 7, "op": "advance", "args": {...}}`.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub id: u64,
    pub op: String,
    #[serde(default)]
    pub args: Value,
}

/// A binary frame: a little-endian u32 JSON length, the JSON header and the
/// raw payload.
pub fn encode_binary(header: &Value, payload: &[u8]) -> Vec<u8> {
    let json = header.to_string();
    let mut out = Vec::with_capacity(4 + json.len() + payload.len());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(json.as_bytes());
    out.extend_from_slice(payload);
    out
}

/// Splits a binary frame into its JSON header and payload.
pub fn decode_binary(frame: &[u8]) -> Result<(Value, &[u8]), String> {
    let length = frame
        .get(..4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
        .ok_or("binary frame shorter than its length prefix")?;
    let json = frame
        .get(4..4usize.saturating_add(length))
        .filter(|_| length <= frame.len() - 4)
        .ok_or("binary frame shorter than its JSON header")?;
    let header = serde_json::from_slice(json).map_err(|e| format!("invalid binary frame header: {e}"))?;
    Ok((header, &frame[4 + length..]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn origins_normalize_to_the_browser_form() {
        for (given, want) in [
            ("https://drivinglab.jectrum.de", "https://drivinglab.jectrum.de"),
            ("https://drivinglab.jectrum.de/", "https://drivinglab.jectrum.de"),
            ("HTTPS://DrivingLab.Jectrum.de:443", "https://drivinglab.jectrum.de"),
            ("http://127.0.0.1:5173", "http://127.0.0.1:5173"),
            ("http://localhost:80", "http://localhost"),
            ("https://example.com:8443", "https://example.com:8443"),
            ("http://[::1]:5173", "http://[::1]:5173"),
            ("http://[0:0:0:0:0:0:0:1]:5173", "http://[::1]:5173"),
            ("http://[::ffff:192.0.2.1]:5173", "http://[::ffff:c000:201]:5173"),
        ] {
            assert_eq!(normalize_origin(given).as_deref(), Ok(want), "{given}");
        }
        for bad in [
            "drivinglab.jectrum.de",
            "ftp://example.com",
            "https://",
            "https://example.com/path",
            "https://example.com?q",
            "https://user@example.com",
            "https://example.com:99999",
            "https://example.com:port",
            "null",
            "http://[::1",
            "http://[localhost]:5173",
            "http://[::1%eth0]:5173",
            "http://local\\host:5173",
            "http://example.com:+80",
            "https://example.com:",
            "https://example.com//",
        ] {
            assert!(normalize_origin(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn origins_match_exactly() {
        let allowed = vec![normalize_origin("https://drivinglab.jectrum.de").unwrap()];
        assert!(origin_allowed("https://drivinglab.jectrum.de", &allowed));
        for other in [
            "http://drivinglab.jectrum.de",
            "https://drivinglab.jectrum.de:8443",
            "https://drivinglab.jectrum.de/",
            "https://evil.drivinglab.jectrum.de",
            "https://drivinglab.jectrum.de.evil.com",
            "null",
            "",
        ] {
            assert!(!origin_allowed(other, &allowed), "{other}");
        }
        assert!(!origin_allowed("null", &["null".to_string()]));
    }

    #[test]
    fn hosts_must_name_the_loopback_port() {
        assert!(host_allowed("127.0.0.1:47800", 47800));
        assert!(host_allowed("localhost:47800", 47800));
        assert!(host_allowed("LocalHost:47800", 47800));
        for bad in [
            "127.0.0.1:47801",
            "127.0.0.1",
            "localhost",
            "evil.example:47800",
            "127.0.0.2:47800",
            "[::1]:47800",
            "localhost:+47800",
            "localhost:047800",
            "",
        ] {
            assert!(!host_allowed(bad, 47800), "{bad}");
        }
    }

    #[test]
    fn binary_frames_round_trip_and_reject_truncation() {
        let header = json!({"id": 3, "op": "restoreCheckpointBytes"});
        let frame = encode_binary(&header, b"payload");
        let (h, payload) = decode_binary(&frame).unwrap();
        assert_eq!((h, payload), (header.clone(), &b"payload"[..]));
        let empty = encode_binary(&header, b"");
        assert_eq!(decode_binary(&empty).unwrap().1, b"");
        assert!(decode_binary(&frame[..3]).is_err());
        assert!(decode_binary(&frame[..10]).is_err());
        let mut huge = frame.clone();
        huge[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode_binary(&huge).is_err());
        let mut garbage = frame;
        garbage[4] = b'x';
        assert!(decode_binary(&garbage).is_err());
    }

    #[test]
    fn requests_parse_with_optional_args() {
        let r: Request = serde_json::from_str(r#"{"id":1,"op":"nextGeneration"}"#).unwrap();
        assert_eq!((r.id, r.op.as_str(), r.args), (1, "nextGeneration", Value::Null));
        assert!(serde_json::from_str::<Request>(r#"{"op":"x"}"#).is_err());
        assert!(serde_json::from_str::<Request>(r#"{"id":1,"op":"x","extra":1}"#).is_err());
    }
}
