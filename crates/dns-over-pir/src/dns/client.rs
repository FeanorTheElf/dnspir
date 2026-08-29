//! DNS-over-PIR client: walks the DNS tree iteratively, issuing one or more
//! PIR queries per server hop until it gets a final A/AAAA record (or runs
//! out of next-hop servers it can reach).
//!
//! The iteration is structured around two trait objects so the logic is
//! exercisable without a real PIR engine or TCP socket — see the test
//! module at the bottom of this file.

use std::array::from_fn;
use std::fmt;
use std::net::{IpAddr, SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use feanor_math::ring::RingStore;
use feanor_math::rings::zn::ZnRing;
use feanor_math::rings::zn::zn_64::Zn;
use tracing::instrument;

use crate::dns::DEFAULT_PORT;
use crate::dns::bucket::{self, BUCKET_BYTES, Record, canonical};
use crate::dns::protocol::{
    CMD_DUMP, CMD_INFO, CMD_QUERY, InfoResponse, RESP_UNAVAILABLE, decode_info, read_frame,
    write_frame,
};
use crate::dns::zone::{self, Zone, dummy_ip};
use pir::double_pir::PRIMARY_PLAIN_MODULUS;
use pir::pir_wrapper::{prepare_query, process_reply};

/// Client-side cap on dumps it will accept. A misbehaving (or oversized) zone
/// server could otherwise force us to allocate arbitrarily much memory. The
/// server-side threshold for entering dump-only mode is ~100 KB, so this
/// leaves room for some headroom plus a safety factor.
const DUMP_FETCH_CAP_BYTES: usize = 2 * 1024 * 1024;

/// TCP connect timeout for one hop's socket. A campus-network RTT is well
/// under a second; we use a generous 5 s so that a transiently slow first
/// SYN-ACK doesn't kill the resolution.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Per-frame read/write timeout. Large PIR replies on a 128-shard `.com`
/// server are seconds in release mode; 30 s leaves ample headroom.
const IO_TIMEOUT: Duration = Duration::from_secs(30);
/// Hard cap on the number of server hops a single resolution will make
/// before giving up. Real iterative DNS rarely needs more than four
/// (root → TLD → 2nd-level → 3rd-level), so 16 is comfortably wasteful.
const MAX_HOPS: usize = 16;

/// An authoritative PIR-DNS server, as the client sees it. Real instances
/// talk to a remote process over TCP; tests use an in-memory fake.
pub trait DnsServer {
    /// The zone the server is authoritative for, e.g. `"org"`.
    fn zone_name(&self) -> &str;

    /// Returns every record whose discriminator matches `canon_name`, i.e.
    /// the records of `canon_name` itself (plus, in principle, any bucket
    /// collisions that happen to share the discriminator — the server
    /// rejects those at build time so in practice the result belongs to
    /// `canon_name` alone).
    fn query(&self, canon_name: &str) -> std::io::Result<Vec<Record>>;
}

/// Establishes a session with a server reachable at a given socket address.
/// Real production uses the TCP-backed [`HybridFactory`]; tests construct
/// their own in-process implementation.
pub trait DnsServerFactory {
    /// Opens a single-server session. Whatever transport this returns is
    /// expected to outlive any number of subsequent `query()` calls on it
    /// during one hop of the iterative walk.
    fn connect(&self, addr: SocketAddr) -> std::io::Result<Box<dyn DnsServer>>;
}

/// Outcome of [`one_hop`]: everything [`resolve_iteratively`] needs to
/// decide whether to stop, descend, or give up after one server is done
/// answering.
#[derive(Debug)]
pub enum HopOutcome {
    /// Definitive A/AAAA answer for the target.
    Answer(Vec<Record>),
    /// Server delegates downward; try the NS IPs in order.
    Delegation { from: String, ns_ips: Vec<IpAddr> },
    /// Walked the suffix chain to the zone apex with nothing useful.
    NotFound,
    /// Target name doesn't belong to this server's zone at all.
    OutOfZone,
}

/// Outcome of [`resolve_iteratively`] — the result of the entire iterative
/// walk, not just one hop.
#[derive(Debug)]
pub enum ResolutionResult {
    /// One or more A / AAAA records for the target name.
    Found(Vec<Record>),
    /// Walked the iterative chain to its end without finding the name.
    NotFound,
    /// Something else went wrong (no reachable server, hop limit, target
    /// out-of-zone, query error). The string is meant for human display.
    Failed(String),
}

impl fmt::Display for ResolutionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResolutionResult::Found(_) => f.write_str("found"),
            ResolutionResult::NotFound => f.write_str("not found"),
            ResolutionResult::Failed(why) => write!(f, "failed: {}", why),
        }
    }
}

/// Drops the leftmost label of a canonical DNS name (`"a.b.c" -> "b.c"`).
/// Returns `None` if there are no more labels to strip.
fn strip_leftmost(name: &str) -> Option<String> {
    let dot = name.find('.')?;
    Some(name[dot + 1..].to_owned())
}

/// Tests whether `target` is the same name as `zone` or a sub-name of it,
/// using DNS label semantics (the boundary must be a dot, not just a
/// string suffix). Empty `zone` means root and matches everything.
fn target_belongs_to_zone(target: &str, zone: &str) -> bool {
    if zone.is_empty() {
        return true;
    }
    if target == zone {
        return true;
    }
    target.len() > zone.len()
        && target.ends_with(zone)
        && target.as_bytes()[target.len() - zone.len() - 1] == b'.'
}

/// Performs the per-server portion of the iterative walk: query the full
/// target first, then walk shorter suffixes if and only if the previous
/// query came back empty. Stops at the first useful response (A/AAAA for
/// the target, or any NS delegation) or at the zone apex.
///
/// Querying the full name first is what makes glue lookups like
/// `dns2.fastdns24.org` work against the .org TLD: that record only
/// exists in the bucket keyed by its full name, not in the bucket for
/// the truncated suffix `fastdns24.org`.
#[instrument(skip_all)]
pub fn one_hop(server: &dyn DnsServer, target: &str) -> std::io::Result<HopOutcome> {
    let zone = server.zone_name();
    if !target_belongs_to_zone(target, zone) {
        return Ok(HopOutcome::OutOfZone);
    }

    let mut try_name = target.to_owned();
    loop {
        let records = server.query(&try_name)?;
        let mut a_recs = Vec::new();
        let mut ns_ips = Vec::new();
        for r in records {
            match r {
                Record::A(_) | Record::Aaaa(_) => a_recs.push(r),
                Record::Ns(ip) => ns_ips.push(ip),
            }
        }
        if try_name == target && !a_recs.is_empty() {
            return Ok(HopOutcome::Answer(a_recs));
        }
        if !ns_ips.is_empty() {
            return Ok(HopOutcome::Delegation {
                from: try_name,
                ns_ips,
            });
        }
        if try_name == zone {
            // Reached the zone apex with nothing useful — give up. (The root
            // zone has zone == "" and this branch is hit when the suffix walk
            // has already consumed every label of the target.)
            return Ok(HopOutcome::NotFound);
        }
        match strip_leftmost(&try_name) {
            Some(stripped) => try_name = stripped,
            None => {
                // No more labels to strip, but we haven't reached the zone
                // apex either. Only possible when the root zone (empty
                // zone_name) didn't know about any of the target's TLDs.
                return Ok(HopOutcome::NotFound);
            }
        }
    }
}

/// Top-level iterative resolver. Walks server-by-server, retrying through
/// every delegated NS IP until one is reachable.
#[instrument(skip_all)]
pub fn resolve_iteratively(
    target: &str,
    initial_candidates: &[SocketAddr],
    factory: &dyn DnsServerFactory,
    mut on_event: impl FnMut(ResolutionEvent),
) -> ResolutionResult {
    let target = canonical(target);
    if target.is_empty() {
        return ResolutionResult::Failed("empty domain name".into());
    }
    if initial_candidates.is_empty() {
        return ResolutionResult::Failed("no servers to try".into());
    }

    // Internal candidate pool is a stack popped from the back, so we push the
    // caller-supplied order in reverse to preserve "first listed = first tried".
    let mut candidates: Vec<SocketAddr> = initial_candidates.iter().rev().copied().collect();
    let mut tried_total = 0usize;
    let mut hops = 0usize;

    while !candidates.is_empty() {
        if hops >= MAX_HOPS {
            return ResolutionResult::Failed(format!("too many hops ({})", MAX_HOPS));
        }
        // Pop candidates until one connects — a delegation may list dead
        // glue (TEST-NET-1 placeholders for out-of-zone NS hostnames) ahead
        // of a real, reachable address.
        let mut server_and_addr = None;
        while let Some(addr) = candidates.pop() {
            tried_total += 1;
            match factory.connect(addr) {
                Ok(s) => {
                    server_and_addr = Some((addr, s));
                    break;
                }
                Err(e) => on_event(ResolutionEvent::ConnectFailed {
                    addr,
                    error: e.to_string(),
                }),
            }
        }
        let Some((addr, server)) = server_and_addr else {
            return ResolutionResult::Failed(format!(
                "no candidate server reachable (tried {})",
                tried_total
            ));
        };
        hops += 1;
        on_event(ResolutionEvent::HopStart {
            hop: hops,
            addr,
            zone: server.zone_name().to_owned(),
        });

        match one_hop(&*server, &target) {
            Ok(HopOutcome::Answer(recs)) => return ResolutionResult::Found(recs),
            Ok(HopOutcome::Delegation { from, ns_ips }) => {
                on_event(ResolutionEvent::Delegation {
                    from: from.clone(),
                    ns_ips: ns_ips.clone(),
                });
                // Reset the candidate pool to the new delegations, pushed
                // in reverse so the first listed NS gets popped first.
                candidates.clear();
                for ip in ns_ips.into_iter().rev() {
                    candidates.push(SocketAddr::new(ip, DEFAULT_PORT));
                }
            }
            Ok(HopOutcome::NotFound) => return ResolutionResult::NotFound,
            Ok(HopOutcome::OutOfZone) => {
                return ResolutionResult::Failed(format!(
                    "target {} doesn't belong to zone .{}",
                    target,
                    server.zone_name()
                ));
            }
            Err(e) => return ResolutionResult::Failed(format!("query error: {}", e)),
        }
    }

    ResolutionResult::Failed("no candidates left to try".into())
}

/// Progress notifications emitted by [`resolve_iteratively`]. The CLI uses
/// these for its on-screen output; tests can ignore them by passing
/// `|_| {}`.
#[derive(Debug, Clone)]
pub enum ResolutionEvent {
    /// A candidate server address was tried but unreachable.
    ConnectFailed { addr: SocketAddr, error: String },
    /// A successful TCP connection became the start of a new hop.
    HopStart { hop: usize, addr: SocketAddr, zone: String },
    /// The current hop returned a delegation; the listed NS IPs are about
    /// to be tried as candidates for the next hop.
    Delegation { from: String, ns_ips: Vec<IpAddr> },
}

/// CLI entry point used by `main`. Prints the resolution result plus a
/// per-resolution timing and byte-count summary at the end. Walks the
/// cached global zone first (if any) so the bootstrap PIR query is
/// skipped when possible, then runs [`resolve_iteratively`].
pub fn run(target: &str, cli_bootstrap: Option<SocketAddr>) -> std::io::Result<()> {
    let started = Instant::now();
    let sent_at_start = crate::dns::protocol::bytes_sent();
    let recvd_at_start = crate::dns::protocol::bytes_recvd();
    let result = run_inner(target, cli_bootstrap);
    let sent = crate::dns::protocol::bytes_sent() - sent_at_start;
    let recvd = crate::dns::protocol::bytes_recvd() - recvd_at_start;
    println!(
        "Resolution took {:.3} s; sent {} B, received {} B over the wire",
        started.elapsed().as_secs_f64(),
        sent,
        recvd,
    );
    result
}

/// Implementation of [`run`] minus the timing/byte-count instrumentation
/// wrapper.
#[instrument(skip_all)]
fn run_inner(target: &str, cli_bootstrap: Option<SocketAddr>) -> std::io::Result<()> {
    let canon = canonical(target);
    if canon.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "empty domain name",
        ));
    }
    println!("Resolving {} (iterative PIR-DNS lookup)", canon);

    let cache_path = default_cache_path();
    let on_event = |ev: ResolutionEvent| match ev {
        ResolutionEvent::ConnectFailed { addr, error } => {
            println!("  could not reach {}: {}", addr, error);
        }
        ResolutionEvent::HopStart { hop, addr, zone } => {
            println!("  hop {}: {} hosts zone .{}", hop, addr, zone);
        }
        ResolutionEvent::Delegation { from, ns_ips } => {
            println!("    delegation for {}:", from);
            for ip in &ns_ips {
                println!("      -> {}", ip);
            }
        }
    };
    let factory = HybridFactory::new(Some(cache_path.clone()));

    // Try the local cache for the global zone first. On a hit, we can do
    // the first hop entirely offline — avoiding the PIR round-trip the
    // user would otherwise pay every lookup.
    let mut initial_candidates: Vec<SocketAddr> = match load_cache(&cache_path) {
        Some(zone) if zone.zone_name.is_empty() => {
            println!("  global zone served from cache {:?}", cache_path);
            let local = LocalZoneServer::new(zone);
            match one_hop(&local, &canon) {
                Ok(HopOutcome::Answer(recs)) => {
                    // Extremely unusual but handle gracefully: the answer
                    // was already in the cached global zone.
                    print_answer(&canon, &recs);
                    return Ok(());
                }
                Ok(HopOutcome::Delegation { from, ns_ips }) => {
                    println!("    cache delegation for {}:", from);
                    for ip in &ns_ips {
                        println!("      -> {}", ip);
                    }
                    ns_ips
                        .into_iter()
                        .map(|ip| SocketAddr::new(ip, DEFAULT_PORT))
                        .collect()
                }
                Ok(HopOutcome::NotFound) | Ok(HopOutcome::OutOfZone) | Err(_) => {
                    println!("  cache lookup yielded nothing useful; falling back");
                    Vec::new()
                }
            }
        }
        _ => Vec::new(),
    };
    // Always keep `--server` as a fallback at the back of the candidate
    // list: if the cache is stale and all its delegated NS IPs are dead,
    // the explicit bootstrap is what gets us back on our feet.
    if let Some(bootstrap) = cli_bootstrap {
        if !initial_candidates.contains(&bootstrap) {
            initial_candidates.push(bootstrap);
        }
    }

    if initial_candidates.is_empty() {
        println!(
            "resolution failed: no global zone cached and no --server provided"
        );
        return Ok(());
    }

    let result = resolve_iteratively(&canon, &initial_candidates, &factory, on_event);

    match result {
        ResolutionResult::Found(recs) => print_answer(&canon, &recs),
        ResolutionResult::NotFound => println!("{} not found", canon),
        ResolutionResult::Failed(why) => println!("resolution failed: {}", why),
    }
    Ok(())
}

/// Prints the A / AAAA records of a resolution result to stdout in
/// human-readable form. NS records (which can occur in the rare "cached
/// global zone happens to know the target directly" case) are ignored —
/// the caller has already followed the delegation path or will.
fn print_answer(name: &str, recs: &[Record]) {
    for r in recs {
        match r {
            Record::A(ip) => println!("{} A {}", name, ip),
            Record::Aaaa(ip) => println!("{} AAAA {}", name, ip),
            Record::Ns(_) => {}
        }
    }
}

// --- Local zone (downloaded full-dump) DnsServer impl -----------------------

/// Wraps an in-memory [`Zone`] as a [`DnsServer`]. Used by the client when
/// the upstream server advertised a dump and we fetched it (or when the
/// global zone was loaded from cache).
pub struct LocalZoneServer {
    zone: Zone,
}

impl LocalZoneServer {
    /// Wraps a parsed [`Zone`] in a [`DnsServer`]. After construction the
    /// zone is owned by this server and the lookup path is purely
    /// in-memory — no network or PIR involved.
    pub fn new(zone: Zone) -> Self {
        LocalZoneServer { zone }
    }
}

impl DnsServer for LocalZoneServer {
    fn zone_name(&self) -> &str {
        &self.zone.zone_name
    }

    /// Mirrors the bucket-building logic in `server::build_buckets`: A and
    /// AAAA records pass through as-is, NS rdata is resolved to in-zone glue
    /// when available and a deterministic TEST-NET-1 placeholder otherwise.
    #[instrument(skip_all)]
    fn query(&self, canon_name: &str) -> std::io::Result<Vec<Record>> {
        let mut out = Vec::new();
        if let Some(rec) = self.zone.records.get(canon_name) {
            for ip in &rec.a {
                out.push(Record::A(*ip));
            }
            for ip in &rec.aaaa {
                out.push(Record::Aaaa(*ip));
            }
            for &ns_idx in &rec.ns {
                let nshost = self.zone.ns_hostname(ns_idx);
                let ip = self.zone.glue_for(nshost).unwrap_or_else(|| dummy_ip(nshost));
                out.push(Record::Ns(ip));
            }
        }
        Ok(out)
    }
}

// --- Hybrid factory: dump if offered + small, else PIR ----------------------

/// Wire-level factory that, on each `connect`, asks the server what it offers
/// (INFO), grabs the gzipped zone dump if it's available and reasonably
/// sized, and otherwise falls back to PIR queries over the same TCP socket.
/// Optionally stashes the *global* zone (`zone_name == ""`) on disk so the
/// next client invocation can skip the bootstrap server entirely.
pub struct HybridFactory {
    /// Where to cache the global zone, if anywhere. `None` disables caching.
    pub cache_path: Option<PathBuf>,
}

impl HybridFactory {
    /// Constructs a factory. Pass `Some(path)` to have the global zone
    /// stashed on disk after first download; `None` disables caching
    /// (the factory still resolves correctly, just without persistence).
    pub fn new(cache_path: Option<PathBuf>) -> Self {
        HybridFactory { cache_path }
    }
}

impl DnsServerFactory for HybridFactory {

    #[instrument(skip_all)]
    fn connect(&self, addr: SocketAddr) -> std::io::Result<Box<dyn DnsServer>> {
        // Single TCP connection per hop: open, do INFO, then keep reusing
        // the same socket for either CMD_DUMP or any number of CMD_QUERY
        // frames. The server's `handle` loops over frames until the peer
        // half-closes.
        let mut stream = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT)?;
        stream.set_read_timeout(Some(IO_TIMEOUT))?;
        stream.set_write_timeout(Some(IO_TIMEOUT))?;
        write_frame(&mut stream, &[CMD_INFO])?;
        let info_bytes = read_frame(&mut stream)?;
        let info = decode_info(&info_bytes)?;

        let want_dump =
            info.has_dump() && (info.dump_size as usize) <= DUMP_FETCH_CAP_BYTES;

        if want_dump {
            // Reuse the same socket for CMD_DUMP.
            write_frame(&mut stream, &[CMD_DUMP])?;
            let gz = read_frame(&mut stream)?;
            // Drop the stream; we don't need it again for this hop.
            drop(stream);
            if gz == RESP_UNAVAILABLE {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("server at {} advertised a dump but refused to send it", addr),
                ));
            }
            if gz.len() > DUMP_FETCH_CAP_BYTES {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("dump {} B exceeds cap {} B", gz.len(), DUMP_FETCH_CAP_BYTES),
                ));
            }
            // Cache iff this is the global zone — that's the only one we
            // ever look up by zone_name on disk.
            if info.zone_name.is_empty() {
                if let Some(path) = &self.cache_path {
                    if let Err(e) = save_cache(path, &gz) {
                        eprintln!("warn: could not cache global zone to {:?}: {}", path, e);
                    }
                }
            }
            let zone = zone::load_dump(&gz)?;
            return Ok(Box::new(LocalZoneServer::new(zone)));
        }
        if info.has_pir() {
            // Hand the live socket to TcpDnsServer; subsequent
            // CMD_QUERY frames run over it without re-handshaking TCP.
            return Ok(Box::new(TcpDnsServer {
                addr,
                info,
                stream: Mutex::new(stream),
            }));
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("server at {} offers neither PIR nor a fetchable dump", addr),
        ))
    }
}

// --- TCP-backed DnsServer implementation ------------------------------------

/// `DnsServer` returned by [`HybridFactory`] when the remote runs in PIR
/// mode. Holds the live TCP socket so the per-hop suffix walk reuses one
/// connection for all `CMD_QUERY` frames.
struct TcpDnsServer {
    #[allow(dead_code)]
    /// Remote address — kept around purely for debug-printing.
    addr: SocketAddr,
    /// Cached `INFO` response (zone name, shard count, hash salt, …)
    /// fetched by [`HybridFactory::connect`].
    info: InfoResponse,
    /// Live socket opened by `HybridFactory::connect`. Reused for every
    /// `query()` call so we don't pay a TCP handshake per suffix walk.
    stream: Mutex<TcpStream>,
}

impl DnsServer for TcpDnsServer {
    fn zone_name(&self) -> &str {
        &self.info.zone_name
    }

    #[instrument(skip_all)]
    fn query(&self, canon_name: &str) -> std::io::Result<Vec<Record>> {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let salt = self.info.hash_salt;
        let (secondary_idx, primary_idx) =
            bucket::slot(canon_name, self.info.num_entries as usize, salt);
        let mut rng = StdRng::from_entropy();
        let (query_bytes, sk_seed) =
            prepare_query(&mut rng, primary_idx, secondary_idx, self.info.num_entries as usize);

        let mut payload = Vec::with_capacity(1 + query_bytes.len());
        payload.push(CMD_QUERY);
        payload.extend_from_slice(&query_bytes);

        let reply = {
            let mut stream = self.stream.lock().unwrap();
            write_frame(&mut *stream, &payload)?;
            read_frame(&mut *stream)?
        };
        if reply == RESP_UNAVAILABLE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "server refused PIR query (no PIR mode)",
            ));
        }

        let coeffs = process_reply(reply.into_iter(), sk_seed);
        let bucket_bytes = coeffs_to_bucket(&coeffs);
        // The server's builder picked a per-bucket salt (because 14-bit
        // discriminators collide easily) and stashed it in the bucket
        // header; we need that same value to reproduce the discriminator
        // for our query name.
        let bucket_salt = bucket::read_bucket_salt(&bucket_bytes);
        let disc = bucket::discriminator(canon_name, salt, bucket_salt);
        Ok(bucket::decode_bucket_matching(&bucket_bytes, disc))
    }
}

// --- global-zone cache file -------------------------------------------------

/// On-disk location of the global-zone cache. Honors `XDG_CACHE_HOME`,
/// then `$HOME/.cache/`, then falls back to the system temp directory.
pub fn default_cache_path() -> PathBuf {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(|| std::env::temp_dir());
    base.join("dns-over-pir").join("global-zone.gz")
}

/// Reads the cache file (the gzipped global zone) from disk, if any.
/// Returns `None` on any IO or parse error — callers treat that as
/// "no cache" rather than as a hard failure.
pub fn load_cache(path: &Path) -> Option<Zone> {
    let bytes = std::fs::read(path).ok()?;
    zone::load_dump(&bytes).ok()
}

/// Atomically (-ish) writes the gzipped zone bytes to `path`, creating
/// the parent directory if needed. Used by [`HybridFactory::connect`]
/// after the first successful download of a global zone.
fn save_cache(path: &Path, gz_bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, gz_bytes)
}

/// Inverts the 10-bit-per-coefficient packing the server's `preprocess`
/// applied. Reads `bucket::N` plaintext coefficients, then unpacks them
/// four-at-a-time into `BUCKET_BYTES` of bucket payload via
/// [`bucket::unpack_coeffs_into_bytes`].
#[instrument(skip_all)]
fn coeffs_to_bucket(coeffs: &[feanor_math::ring::El<Zn>]) -> [u8; BUCKET_BYTES] {
    assert!(coeffs.len() >= bucket::N);
    let Zt = Zn::new(PRIMARY_PLAIN_MODULUS as u64);
    let coeffs_arr: [u16; bucket::N] = from_fn(|i| {
        Zt.get_ring().smallest_positive_lift(coeffs[i]) as u16
    });
    bucket::unpack_coeffs_into_bytes(&coeffs_arr)
}

// --- tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::net::{Ipv4Addr, Ipv6Addr};
    use std::sync::Mutex;

    use super::*;

    /// In-memory DnsServer impl backed by a map of canonical name to records.
    /// Used to exercise the iteration logic without standing up a real
    /// PIR server.
    struct FakeServer {
        zone: String,
        records: HashMap<String, Vec<Record>>,
    }

    impl DnsServer for FakeServer {
        fn zone_name(&self) -> &str {
            &self.zone
        }
        fn query(&self, name: &str) -> std::io::Result<Vec<Record>> {
            Ok(self.records.get(name).cloned().unwrap_or_default())
        }
    }

    fn server_with(zone: &str, records: &[(&str, Vec<Record>)]) -> FakeServer {
        FakeServer {
            zone: zone.to_owned(),
            records: records
                .iter()
                .map(|(k, v)| ((*k).to_owned(), v.clone()))
                .collect(),
        }
    }

    // -- one_hop tests ------------------------------------------------------

    #[test]
    fn one_hop_finds_direct_a_for_deep_name() {
        // Bug 2 in the user report: a glue A record three labels deep on the
        // .org TLD must be retrievable by the deep name itself, not via a
        // truncated parent suffix.
        let srv = server_with(
            "org",
            &[(
                "dns2.fastdns24.org",
                vec![Record::A(Ipv4Addr::new(178, 132, 200, 29))],
            )],
        );
        match one_hop(&srv, "dns2.fastdns24.org").unwrap() {
            HopOutcome::Answer(recs) => {
                assert_eq!(recs, vec![Record::A(Ipv4Addr::new(178, 132, 200, 29))]);
            }
            other => panic!("expected Answer, got {:?}", other),
        }
    }

    #[test]
    fn one_hop_walks_suffixes_when_full_name_empty() {
        let srv = server_with(
            "org",
            &[(
                "example.org",
                vec![Record::Ns(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)))],
            )],
        );
        match one_hop(&srv, "www.example.org").unwrap() {
            HopOutcome::Delegation { from, ns_ips } => {
                assert_eq!(from, "example.org");
                assert_eq!(ns_ips, vec![IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1))]);
            }
            other => panic!("expected Delegation, got {:?}", other),
        }
    }

    #[test]
    fn one_hop_returns_not_found_at_zone_apex() {
        let srv = server_with("org", &[]);
        match one_hop(&srv, "missing.org").unwrap() {
            HopOutcome::NotFound => {}
            other => panic!("expected NotFound, got {:?}", other),
        }
    }

    #[test]
    fn one_hop_prefers_a_over_ns_for_target() {
        let srv = server_with(
            "org",
            &[(
                "apex.org",
                vec![
                    Record::A(Ipv4Addr::new(1, 2, 3, 4)),
                    Record::Ns(IpAddr::V4(Ipv4Addr::new(9, 9, 9, 9))),
                ],
            )],
        );
        match one_hop(&srv, "apex.org").unwrap() {
            HopOutcome::Answer(recs) => {
                assert!(recs.contains(&Record::A(Ipv4Addr::new(1, 2, 3, 4))));
            }
            other => panic!("expected Answer, got {:?}", other),
        }
    }

    #[test]
    fn one_hop_returns_aaaa_too() {
        let srv = server_with(
            "org",
            &[(
                "host.org",
                vec![
                    Record::A(Ipv4Addr::new(1, 2, 3, 4)),
                    Record::Aaaa("2001:db8::1".parse::<Ipv6Addr>().unwrap()),
                ],
            )],
        );
        match one_hop(&srv, "host.org").unwrap() {
            HopOutcome::Answer(recs) => {
                assert_eq!(recs.len(), 2);
            }
            other => panic!("expected Answer, got {:?}", other),
        }
    }

    #[test]
    fn one_hop_out_of_zone() {
        let srv = server_with("org", &[]);
        match one_hop(&srv, "example.com").unwrap() {
            HopOutcome::OutOfZone => {}
            other => panic!("expected OutOfZone, got {:?}", other),
        }
    }

    /// Targets that *string-suffix* the zone name without a label
    /// boundary (e.g. `"thatsorg"` and `"org"`) are not in the zone.
    #[test]
    fn one_hop_rejects_string_suffix_without_label_boundary() {
        let srv = server_with("org", &[]);
        for bad in ["thatsorg", "horg", "evilorg"] {
            match one_hop(&srv, bad).unwrap() {
                HopOutcome::OutOfZone => {}
                other => panic!("expected OutOfZone for {:?}, got {:?}", bad, other),
            }
        }
    }

    #[test]
    fn target_belongs_to_zone_examples() {
        // Root zone accepts everything (including names with no dot).
        assert!(target_belongs_to_zone("anything", ""));
        assert!(target_belongs_to_zone("a.b.c", ""));
        // Exact apex match.
        assert!(target_belongs_to_zone("org", "org"));
        // Proper sub-name (dot-bounded).
        assert!(target_belongs_to_zone("example.org", "org"));
        assert!(target_belongs_to_zone("a.b.c.org", "org"));
        // String suffix without a label boundary — must be rejected.
        assert!(!target_belongs_to_zone("thatsorg", "org"));
        assert!(!target_belongs_to_zone("evilorg", "org"));
        // Different TLD entirely.
        assert!(!target_belongs_to_zone("example.com", "org"));
    }

    // -- resolve_iteratively tests ------------------------------------------

    /// Connection factory backed by a map of `addr -> FakeServer`. Addresses
    /// not in the map are treated as unreachable; this lets us simulate
    /// dead glue (a TEST-NET-1 placeholder) alongside a working server.
    struct FakeFactory {
        servers: Mutex<HashMap<SocketAddr, std::sync::Arc<FakeServer>>>,
        connect_log: Mutex<Vec<SocketAddr>>,
    }

    impl FakeFactory {
        fn new(map: HashMap<SocketAddr, FakeServer>) -> Self {
            FakeFactory {
                servers: Mutex::new(
                    map.into_iter().map(|(a, s)| (a, std::sync::Arc::new(s))).collect(),
                ),
                connect_log: Mutex::new(Vec::new()),
            }
        }
        fn log(&self) -> Vec<SocketAddr> {
            self.connect_log.lock().unwrap().clone()
        }
    }

    struct ArcFake(std::sync::Arc<FakeServer>);
    impl DnsServer for ArcFake {
        fn zone_name(&self) -> &str { self.0.zone_name() }
        fn query(&self, n: &str) -> std::io::Result<Vec<Record>> { self.0.query(n) }
    }

    impl DnsServerFactory for FakeFactory {
        fn connect(&self, addr: SocketAddr) -> std::io::Result<Box<dyn DnsServer>> {
            self.connect_log.lock().unwrap().push(addr);
            match self.servers.lock().unwrap().get(&addr) {
                Some(s) => Ok(Box::new(ArcFake(s.clone()))),
                None => Err(std::io::Error::new(
                    std::io::ErrorKind::ConnectionRefused,
                    format!("no fake server at {}", addr),
                )),
            }
        }
    }

    fn sa(ip: [u8; 4], port: u16) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::from(ip)), port)
    }

    #[test]
    fn resolve_returns_a_for_deep_glue() {
        // Bug 2 again, but exercising the whole resolve_iteratively path.
        let tld = server_with(
            "org",
            &[(
                "dns2.fastdns24.org",
                vec![Record::A(Ipv4Addr::new(178, 132, 200, 29))],
            )],
        );
        let mut servers = HashMap::new();
        servers.insert(sa([127, 0, 0, 1], DEFAULT_PORT), tld);
        let factory = FakeFactory::new(servers);

        let result = resolve_iteratively(
            "dns2.fastdns24.org",
            &[sa([127, 0, 0, 1], DEFAULT_PORT)],
            &factory,
            |_| {},
        );
        match result {
            ResolutionResult::Found(recs) => {
                assert_eq!(recs, vec![Record::A(Ipv4Addr::new(178, 132, 200, 29))]);
            }
            other => panic!("expected Found, got {:?}", other),
        }
    }

    #[test]
    fn resolve_retries_through_dead_glue_until_one_works() {
        // Bug 1 in the user report: when a delegation contains multiple NS
        // IPs and the first few are TEST-NET-1 placeholders for out-of-zone
        // hostnames, the client must keep trying instead of giving up at
        // the first unreachable address.
        //
        // The TLD server delegates apex.org to three NS IPs:
        //   - 192.0.2.50 (TEST-NET-1, unreachable)
        //   - 192.0.2.80 (TEST-NET-1, unreachable)
        //   - 10.0.0.1   (real glue, hosts apex.org's authoritative server)
        //
        // The authoritative server returns the final A.
        let tld_zone = "org";
        let tld = server_with(
            tld_zone,
            &[(
                "apex.org",
                vec![
                    Record::Ns(IpAddr::V4(Ipv4Addr::new(192, 0, 2, 50))),
                    Record::Ns(IpAddr::V4(Ipv4Addr::new(192, 0, 2, 80))),
                    Record::Ns(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1))),
                ],
            )],
        );
        let apex = server_with(
            "apex.org",
            &[("apex.org", vec![Record::A(Ipv4Addr::new(1, 2, 3, 4))])],
        );
        let mut servers = HashMap::new();
        servers.insert(sa([127, 0, 0, 1], DEFAULT_PORT), tld);
        servers.insert(sa([10, 0, 0, 1], DEFAULT_PORT), apex);
        let factory = FakeFactory::new(servers);

        let result = resolve_iteratively(
            "apex.org",
            &[sa([127, 0, 0, 1], DEFAULT_PORT)],
            &factory,
            |_| {},
        );
        match result {
            ResolutionResult::Found(recs) => {
                assert_eq!(recs, vec![Record::A(Ipv4Addr::new(1, 2, 3, 4))]);
            }
            other => panic!("expected Found, got {:?}", other),
        }
        let log = factory.log();
        // Must have tried the bootstrap, then walked through both dead
        // glue IPs, then reached the real one.
        assert_eq!(log.len(), 4, "tried addrs: {:?}", log);
        assert_eq!(log[0], sa([127, 0, 0, 1], DEFAULT_PORT));
        assert_eq!(log[3], sa([10, 0, 0, 1], DEFAULT_PORT));
    }

    #[test]
    fn resolve_fails_cleanly_when_all_delegations_dead() {
        let tld = server_with(
            "org",
            &[(
                "apex.org",
                vec![
                    Record::Ns(IpAddr::V4(Ipv4Addr::new(192, 0, 2, 50))),
                    Record::Ns(IpAddr::V4(Ipv4Addr::new(192, 0, 2, 80))),
                ],
            )],
        );
        let mut servers = HashMap::new();
        servers.insert(sa([127, 0, 0, 1], DEFAULT_PORT), tld);
        let factory = FakeFactory::new(servers);

        let result = resolve_iteratively(
            "apex.org",
            &[sa([127, 0, 0, 1], DEFAULT_PORT)],
            &factory,
            |_| {},
        );
        match result {
            ResolutionResult::Failed(_) => {}
            other => panic!("expected Failed, got {:?}", other),
        }
    }

    #[test]
    fn resolve_returns_not_found_for_missing_name() {
        let tld = server_with("org", &[]);
        let mut servers = HashMap::new();
        servers.insert(sa([127, 0, 0, 1], DEFAULT_PORT), tld);
        let factory = FakeFactory::new(servers);

        let result = resolve_iteratively(
            "ghost.org",
            &[sa([127, 0, 0, 1], DEFAULT_PORT)],
            &factory,
            |_| {},
        );
        match result {
            ResolutionResult::NotFound => {}
            other => panic!("expected NotFound, got {:?}", other),
        }
    }

    #[test]
    fn strip_leftmost_works() {
        assert_eq!(strip_leftmost("a.b.c"), Some("b.c".to_owned()));
        assert_eq!(strip_leftmost("a.b"), Some("b".to_owned()));
        assert_eq!(strip_leftmost("a"), None);
        assert_eq!(strip_leftmost(""), None);
    }

    // -- LocalZoneServer and HybridFactory tests ----------------------------

    fn zone_from(text: &str) -> Zone {
        Zone::parse(text.as_bytes()).unwrap()
    }

    #[test]
    fn local_zone_server_resolves_in_zone_glue() {
        // NS pointing to an in-zone host: LocalZoneServer must hand back
        // the real glue IP, exactly like the PIR-server bucket layer does.
        let zone = zone_from(
            "apex.org.\t3600\tin\tns\tns1.elsewhere.com.\n\
             apex.org.\t3600\tin\tns\tns2.inside.org.\n\
             ns2.inside.org.\t3600\tin\ta\t178.132.200.29\n",
        );
        let srv = LocalZoneServer::new(zone);
        let recs = srv.query("apex.org").unwrap();
        let ns_ips: Vec<IpAddr> = recs
            .iter()
            .filter_map(|r| match r {
                Record::Ns(ip) => Some(*ip),
                _ => None,
            })
            .collect();
        assert!(
            ns_ips.contains(&IpAddr::V4(Ipv4Addr::new(178, 132, 200, 29))),
            "expected real glue IP among {:?}",
            ns_ips
        );
        // The out-of-zone NS gets a TEST-NET-1 placeholder, not nothing.
        assert!(
            ns_ips.iter().any(|ip| matches!(
                ip,
                IpAddr::V4(v) if v.octets()[..3] == [192, 0, 2]
            )),
            "expected placeholder among {:?}",
            ns_ips
        );
    }

    #[test]
    fn local_zone_server_returns_a_and_aaaa() {
        let zone = zone_from(
            "host.org.\t3600\tin\ta\t1.2.3.4\n\
             host.org.\t3600\tin\taaaa\t2001:db8::1\n",
        );
        let srv = LocalZoneServer::new(zone);
        let recs = srv.query("host.org").unwrap();
        assert!(recs.contains(&Record::A(Ipv4Addr::new(1, 2, 3, 4))));
        assert!(recs.contains(&Record::Aaaa("2001:db8::1".parse().unwrap())));
    }

    #[test]
    fn one_hop_works_against_local_zone() {
        // Smoke test: the whole iteration machinery works just as well when
        // the server is a LocalZoneServer (no PIR involved).
        let zone = zone_from(
            "deep.glue.org.\t3600\tin\ta\t10.0.0.1\n",
        );
        let srv = LocalZoneServer::new(zone);
        match one_hop(&srv, "deep.glue.org").unwrap() {
            HopOutcome::Answer(recs) => {
                assert_eq!(recs, vec![Record::A(Ipv4Addr::new(10, 0, 0, 1))]);
            }
            other => panic!("expected Answer, got {:?}", other),
        }
    }

    #[test]
    fn cache_save_and_load_roundtrip() {
        let tmp = std::env::temp_dir()
            .join(format!("dns_over_pir_cache_test_{}.gz", std::process::id()));
        let zone = zone_from(
            "org.\t3600\tin\tns\ttld-org.example.\n\
             tld-org.example.\t3600\tin\ta\t1.2.3.4\n",
        );
        let gz = zone::dump(&zone).unwrap();
        save_cache(&tmp, &gz).unwrap();
        let loaded = load_cache(&tmp).expect("cache load");
        assert_eq!(loaded.records.len(), zone.records.len());
        assert_eq!(loaded.zone_name, zone.zone_name);
        std::fs::remove_file(&tmp).ok();
    }

    /// Bigger end-to-end: the global zone delegates to a TLD server (PIR),
    /// which in turn answers the query. The HybridFactory equivalent for
    /// tests is constructed by hand: the bootstrap address is served by a
    /// LocalZoneServer (mimicking what a real dump-only global server would
    /// have given us) and the delegated address is served by a FakeServer
    /// (mimicking the TLD's PIR backend).
    #[test]
    fn resolve_walks_from_local_global_into_tld_pir() {
        // Global (root) zone — no zone_name, knows about .org.
        let global = LocalZoneServer::new(zone_from(
            "org.\t3600\tin\tns\ttld-org.example.\n\
             tld-org.example.\t3600\tin\ta\t10.0.0.42\n",
        ));
        // TLD .org server — answers apex.org A directly.
        let tld = server_with(
            "org",
            &[("apex.org", vec![Record::A(Ipv4Addr::new(1, 2, 3, 4))])],
        );

        struct MixedFactory {
            global_addr: SocketAddr,
            tld_addr: SocketAddr,
            global: Mutex<Option<LocalZoneServer>>,
            tld: Mutex<Option<FakeServer>>,
        }
        impl DnsServerFactory for MixedFactory {
            fn connect(&self, addr: SocketAddr) -> std::io::Result<Box<dyn DnsServer>> {
                if addr == self.global_addr {
                    if let Some(z) = self.global.lock().unwrap().take() {
                        return Ok(Box::new(z));
                    }
                }
                if addr == self.tld_addr {
                    if let Some(t) = self.tld.lock().unwrap().take() {
                        return Ok(Box::new(t));
                    }
                }
                Err(std::io::Error::new(
                    std::io::ErrorKind::ConnectionRefused,
                    format!("no server at {}", addr),
                ))
            }
        }

        let factory = MixedFactory {
            global_addr: sa([127, 0, 0, 1], DEFAULT_PORT),
            tld_addr: sa([10, 0, 0, 42], DEFAULT_PORT),
            global: Mutex::new(Some(global)),
            tld: Mutex::new(Some(tld)),
        };

        let result = resolve_iteratively(
            "apex.org",
            &[sa([127, 0, 0, 1], DEFAULT_PORT)],
            &factory,
            |_| {},
        );
        match result {
            ResolutionResult::Found(recs) => {
                assert_eq!(recs, vec![Record::A(Ipv4Addr::new(1, 2, 3, 4))]);
            }
            other => panic!("expected Found, got {:?}", other),
        }
    }
}
