//! TCP wire protocol between the PoC client and server.
//!
//! Every message is a length-prefixed frame: 4 bytes big-endian length, then
//! that many payload bytes.
//!
//! Request payload:
//! ```text
//!   [cmd: u8] [body...]
//!   cmd = 1 (INFO)  -> body empty
//!   cmd = 2 (QUERY) -> body is the byte sequence produced by prepare_query
//!   cmd = 3 (DUMP)  -> body empty; server returns the gzipped zone
//! ```
//!
//! Response payload:
//! ```text
//!   INFO  -> [flags: u8]
//!            [num_entries: u32 BE]    (total PIR slots; 0 if PIR not
//!                                      available)
//!            [zone_name_len: u32 BE]
//!            [zone_name: utf-8]
//!            [dump_size: u32 BE]      (0 if dump not available)
//!            [hash_salt: u64 BE]      (salt chosen at build time; the
//!                                      client uses it for both the slot
//!                                      hash and the discriminator)
//!   QUERY -> the byte sequence accepted by process_reply, OR a single 0xFF
//!            byte if PIR isn't available on this server
//!   DUMP  -> the gzipped serialized zone text, OR a single 0xFF byte if no
//!            dump is available
//! ```
//!
//! Multiple request/response pairs may be exchanged over the same TCP
//! connection — `INFO` is typically followed by either `QUERY` or `DUMP`
//! on the same socket. The server reads frames in a loop and only closes
//! when the peer half-closes.
//!
//! `flags` is a bitmap: bit 0 = PIR available, bit 1 = DUMP available.

use std::cell::Cell;
use std::io::{self, Read, Write};

use tracing::instrument;

/// `CMD_INFO`: zero-body request that returns this server's [`InfoResponse`].
pub const CMD_INFO: u8 = 1;
/// `CMD_QUERY`: body is the byte sequence produced by
/// [`pir::pir_wrapper::prepare_query`]. Reply is the byte sequence
/// accepted by [`pir::pir_wrapper::process_reply`], or the
/// `RESP_UNAVAILABLE` sentinel if PIR is disabled on this server.
pub const CMD_QUERY: u8 = 2;
/// `CMD_DUMP`: zero-body request that returns the gzipped serialized zone
/// (or the `RESP_UNAVAILABLE` sentinel if the server isn't running in
/// dump-mode).
pub const CMD_DUMP: u8 = 3;

/// `InfoResponse::flags` bit indicating the server can answer `CMD_QUERY`.
pub const FLAG_PIR_AVAILABLE: u8 = 1 << 0;
/// `InfoResponse::flags` bit indicating the server can answer `CMD_DUMP`
/// (i.e. the zone fits under the dump-mode size threshold).
pub const FLAG_DUMP_AVAILABLE: u8 = 1 << 1;

// Per-thread byte counters incremented by every `write_frame` / `read_frame`
// call. The client reads these around its top-level `run()` to report
// per-resolution traffic; server worker threads have their own (independent)
// thread-local values, so the two never cross-contaminate.
thread_local! {
    static BYTES_SENT: Cell<u64> = const { Cell::new(0) };
    static BYTES_RECVD: Cell<u64> = const { Cell::new(0) };
}

/// Returns the number of bytes this thread has sent through `write_frame`
/// since the thread started. Monotonic; subtract two snapshots to get a delta.
pub fn bytes_sent() -> u64 {
    BYTES_SENT.with(|c| c.get())
}

/// Returns the number of bytes this thread has received through `read_frame`
/// since the thread started. Monotonic; subtract two snapshots to get a delta.
pub fn bytes_recvd() -> u64 {
    BYTES_RECVD.with(|c| c.get())
}

/// Single sentinel byte used as a response to `CMD_QUERY` or `CMD_DUMP` when
/// the server doesn't offer that capability. Cannot collide with a real
/// PIR reply (those are several KB) or a real gzip stream (starts with 0x1f).
pub const RESP_UNAVAILABLE: [u8; 1] = [0xFF];

/// Writes one length-prefixed frame: a big-endian `u32` length, then the
/// payload bytes. Flushes before returning. Updates the `BYTES_SENT`
/// thread-local counter on success.
#[instrument(skip_all)]
pub fn write_frame<W: Write>(mut w: W, payload: &[u8]) -> io::Result<()> {
    let len = u32::try_from(payload.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "frame too large"))?;
    w.write_all(&len.to_be_bytes())?;
    w.write_all(payload)?;
    w.flush()?;
    BYTES_SENT.with(|c| c.set(c.get() + 4 + payload.len() as u64));
    Ok(())
}

/// Reads one length-prefixed frame. Rejects frames larger than 64 MiB — a
/// hard cap well above any legitimate PIR reply or zone dump. Updates the
/// `BYTES_RECVD` thread-local counter on success.
#[instrument(skip_all)]
pub fn read_frame<R: Read>(mut r: R) -> io::Result<Vec<u8>> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len)?;
    let len = u32::from_be_bytes(len) as usize;
    // Bound the per-frame size to something well above the largest PIR
    // ciphertext (~MB-range) but far below "obviously hostile".
    if len > 64 * 1024 * 1024 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "frame too large"));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    BYTES_RECVD.with(|c| c.set(c.get() + 4 + len as u64));
    Ok(buf)
}

/// The server's response to `CMD_INFO`. Everything the client needs to know
/// before issuing a `CMD_QUERY` or deciding to fetch a `CMD_DUMP`.
#[derive(Debug, Clone)]
pub struct InfoResponse {
    /// Bitmap of available capabilities; see [`FLAG_PIR_AVAILABLE`] and
    /// [`FLAG_DUMP_AVAILABLE`]. Use [`InfoResponse::has_pir`] /
    /// [`InfoResponse::has_dump`] to read.
    pub flags: u8,
    /// Total number of PIR slots (plaintext-ring elements) behind this
    /// server. The client derives everything else from it: the
    /// `(secondary_idx, primary_idx)` slot of a name via `bucket::slot`,
    /// and the database shape (index groups, shard count) via
    /// `pir_wrapper::get_database_shape`. `0` when PIR is not available.
    pub num_entries: u32,
    /// Name of the zone this server is authoritative for, e.g. `"org"`.
    /// Empty string means the root zone (mixed-TLD).
    pub zone_name: String,
    /// Size of the gzipped zone the server would return on `CMD_DUMP`,
    /// in bytes. `0` when no dump is available.
    pub dump_size: u32,
    /// Salt fed into both `slot()` and `discriminator()`. Servers choose it
    /// at build time, possibly retrying with successive salts until no
    /// collision or overflow occurs.
    pub hash_salt: u64,
}

impl InfoResponse {
    /// True if this server can answer `CMD_QUERY` requests.
    pub fn has_pir(&self) -> bool {
        self.flags & FLAG_PIR_AVAILABLE != 0
    }
    /// True if this server can answer `CMD_DUMP` requests.
    pub fn has_dump(&self) -> bool {
        self.flags & FLAG_DUMP_AVAILABLE != 0
    }
}

/// Serializes an `InfoResponse` into the wire format described at the top
/// of this module.
#[instrument(skip_all)]
pub fn encode_info(resp: &InfoResponse) -> Vec<u8> {
    let z = resp.zone_name.as_bytes();
    let mut out = Vec::with_capacity(1 + 4 + 4 + z.len() + 4 + 8);
    out.push(resp.flags);
    out.extend_from_slice(&resp.num_entries.to_be_bytes());
    out.extend_from_slice(&(z.len() as u32).to_be_bytes());
    out.extend_from_slice(z);
    out.extend_from_slice(&resp.dump_size.to_be_bytes());
    out.extend_from_slice(&resp.hash_salt.to_be_bytes());
    out
}

/// Inverse of [`encode_info`]. Returns an `InvalidData` error if the
/// payload is shorter than the encoded shape demands or the zone name
/// isn't valid UTF-8.
#[instrument(skip_all)]
pub fn decode_info(payload: &[u8]) -> io::Result<InfoResponse> {
    if payload.len() < 1 + 4 + 4 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "short INFO reply"));
    }
    let flags = payload[0];
    let num_entries = u32::from_be_bytes(payload[1..5].try_into().unwrap());
    let zl = u32::from_be_bytes(payload[5..9].try_into().unwrap()) as usize;
    if payload.len() < 9 + zl + 4 + 8 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "short INFO reply"));
    }
    let zone_name = String::from_utf8(payload[9..9 + zl].to_vec())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad zone name utf-8"))?;
    let dump_size =
        u32::from_be_bytes(payload[9 + zl..9 + zl + 4].try_into().unwrap());
    let hash_salt =
        u64::from_be_bytes(payload[9 + zl + 4..9 + zl + 12].try_into().unwrap());
    Ok(InfoResponse {
        flags,
        num_entries,
        zone_name,
        dump_size,
        hash_salt,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn info_roundtrip() {
        let resp = InfoResponse {
            flags: FLAG_PIR_AVAILABLE | FLAG_DUMP_AVAILABLE,
            num_entries: 7 * 16384,
            zone_name: "org".into(),
            dump_size: 12345,
            hash_salt: 0xDEAD_BEEF_CAFE_F00D,
        };
        let bytes = encode_info(&resp);
        let back = decode_info(&bytes).unwrap();
        assert_eq!(back.flags, resp.flags);
        assert_eq!(back.num_entries, resp.num_entries);
        assert_eq!(back.zone_name, resp.zone_name);
        assert_eq!(back.dump_size, resp.dump_size);
        assert_eq!(back.hash_salt, resp.hash_salt);
        assert!(back.has_pir());
        assert!(back.has_dump());
    }

    #[test]
    fn info_empty_zone_name() {
        let resp = InfoResponse {
            flags: FLAG_DUMP_AVAILABLE,
            num_entries: 0,
            zone_name: "".into(),
            dump_size: 64,
            hash_salt: 0,
        };
        let back = decode_info(&encode_info(&resp)).unwrap();
        assert!(!back.has_pir());
        assert!(back.has_dump());
        assert_eq!(back.zone_name, "");
        assert_eq!(back.hash_salt, 0);
    }
}
