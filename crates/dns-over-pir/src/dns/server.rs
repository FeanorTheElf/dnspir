//! DNS-over-PIR server: loads a zone file, decides between dump-only and
//! PIR mode, preprocesses one or more [`PIRDatabase`] shards if PIR is
//! needed, and serves requests over TCP.
//!
//! Top-level entry point is [`run`]; the heavy startup work happens in
//! [`build_state`] and the per-connection handler is [`handle`]. See the
//! repository-root `CLAUDE.md` for the architectural overview and the
//! wire-format details in [`crate::dns::protocol`].

use std::collections::HashSet;
use std::net::{IpAddr, TcpListener, TcpStream};
use std::sync::Arc;

use feanor_math::ring::RingStore;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::rings::zn::ZnRing;
use feanor_math::rings::zn::zn_64::Zn;
use feanor_math::seq::VectorFn;
use memmap2::{MmapMut, MmapOptions};
use tracing::instrument;

use pir::base_pir::{FIXED_Q, LOG2_N, PIRDatabase, SIMD_COUNT, UsedSeeds};
use pir::bfv::{CipherRing, PlainRing};
use pir::double_pir::PRIMARY_PLAIN_MODULUS;
use pir::pir_wrapper;
use crate::dns::bucket::{
    self, BUCKET_BYTES, FIRST_ENTRY_OFFSET, KIND_A, KIND_AAAA, KIND_NS4,
    KIND_NS6, Record,
};
use crate::dns::protocol::{
    CMD_DUMP, CMD_INFO, CMD_QUERY, FLAG_DUMP_AVAILABLE, FLAG_PIR_AVAILABLE, InfoResponse,
    RESP_UNAVAILABLE, encode_info, read_frame, write_frame,
};
use crate::dns::zone::{self, Zone, dummy_ip};
use pir::pir_wrapper::process_query;
use pir::simd_zn::CompressedZnx8El;

/// All PIR engine objects the server needs at runtime. Held inside
/// [`ServerState`]; `None` indicates dump-only mode.
struct PirState {
    /// The fully-preprocessed PIR shards. The fleet is read-only after
    /// [`preprocess`] returns, so it's safe to lend out via `Arc`.
    dbs: Vec<PIRDatabase<'static>>,
    /// Total number of PIR slots the buckets were hashed into — the value
    /// advertised via `INFO`, from which clients re-derive the database
    /// shape (`pir_wrapper::get_database_shape`) and the slot mapping.
    num_entries: usize,
}

/// Threshold below which a zone is small enough to be served as a single
/// compressed dump instead of via PIR.
pub const DUMP_THRESHOLD_BYTES: usize = 100 * 1024;

/// Skip the gzip step entirely when the serialized zone is this big or
/// larger: at that size, even an unrealistically generous 100× compression
/// ratio still wouldn't bring it under [`DUMP_THRESHOLD_BYTES`], so the
/// check would always say "go PIR" and the only thing the compression
/// step does is burn CPU during startup.
pub const SKIP_COMPRESSION_ABOVE_BYTES: usize = 100 * DUMP_THRESHOLD_BYTES;

/// Long-lived per-server state shared across worker threads.
struct ServerState {
    zone_name: String,
    /// `Some` when PIR is set up. `None` in dump-only mode.
    pir: Option<PirState>,
    /// Gzipped zone dump. Always populated (even in PIR mode we still
    /// compute it, but only advertise it via INFO when it's small enough).
    dump: Option<Vec<u8>>,
    /// Salt that the client must feed into [`bucket::slot`] and
    /// [`bucket::discriminator`] when interpreting this server's PIR
    /// replies. Irrelevant in dump-only mode; we still ship a value
    /// (`0`) so the field is always well-defined.
    hash_salt: u64,
}

/// Hard cap on how many distinct salts we try when bucketing fails. The
/// failure modes (discriminator collision among colliding names, or a
/// bucket overflowing the wire-format budget) are rare for sensibly sized
/// shard counts, so anything up to single-digit retries should be enough.
pub const MAX_REHASH_RETRIES: u64 = 16;

/// Maps an attempt index onto a well-distributed 64-bit salt. Attempt 0
/// always uses salt 0 so the common case (success on the first try)
/// produces the same hash layout the older code did.
fn salt_for_attempt(attempt: u64) -> u64 {
    if attempt == 0 {
        0
    } else {
        attempt.wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }
}

/// Generic retry helper: invokes `builder` with successive salts until it
/// returns `Ok`, or until `max_attempts` salts have been exhausted. Returns
/// the chosen salt alongside the built value so the caller can record it
/// in `INFO`.
fn try_salts<T, F>(max_attempts: u64, mut builder: F) -> std::io::Result<(T, u64)>
where
    F: FnMut(u64) -> Result<T, String>,
{
    let mut last_err: Option<String> = None;
    for attempt in 0..max_attempts {
        let salt = salt_for_attempt(attempt);
        match builder(salt) {
            Ok(t) => {
                if attempt > 0 {
                    eprintln!(
                        "Bucket build succeeded on attempt {} (salt 0x{:016x})",
                        attempt + 1,
                        salt
                    );
                }
                return Ok((t, salt));
            }
            Err(e) => {
                eprintln!(
                    "Bucket build attempt {} (salt 0x{:016x}) failed: {}; rehashing",
                    attempt + 1,
                    salt,
                    e
                );
                last_err = Some(e);
            }
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::Other,
        format!(
            "bucket build failed after {} re-hash attempts; last error: {}",
            max_attempts,
            last_err.unwrap_or_default()
        ),
    ))
}

/// CLI-tunable flags handed to [`run`].
#[derive(Debug, Clone, Copy)]
pub struct ServerOptions {
    /// Force dump-only mode (skip PIR preprocessing) regardless of zone size.
    pub always_dump: bool,
}

impl Default for ServerOptions {
    fn default() -> Self {
        ServerOptions { always_dump: false }
    }
}

/// Server entry point used by `main`. Loads the zone, decides on dump-only
/// vs PIR, runs preprocessing as needed, then sits in an `accept` loop
/// spawning one worker thread per connection.
pub fn run(zone_path: &str, port: u16, opts: ServerOptions) -> std::io::Result<()> {
    let state = build_state(zone_path, opts)?;
    let state = Arc::new(state);
    let listener = TcpListener::bind(("0.0.0.0", port))?;
    eprintln!("Server ready. Listening on 0.0.0.0:{}", port);
    for conn in listener.incoming() {
        let conn = match conn {
            Ok(c) => c,
            Err(e) => {
                eprintln!("accept error: {}", e);
                continue;
            }
        };
        let state = state.clone();
        std::thread::spawn(move || {
            if let Err(e) = handle(conn, &state) {
                eprintln!("client error: {}", e);
            }
        });
    }
    Ok(())
}

/// Loads the zone, gzips it, and decides between dump-only and PIR mode.
#[instrument(skip_all)]
fn build_state(zone_path: &str, opts: ServerOptions) -> std::io::Result<ServerState> {
    eprintln!("Loading zone file {}...", zone_path);
    let zone = Zone::load(zone_path)?;
    let n_names = zone.records.len();
    let n_entries: usize = zone
        .records
        .values()
        .map(|r| r.a.len() + r.aaaa.len() + r.ns.len())
        .sum();
    eprintln!(
        "Loaded {} unique names ({} records); zone is .{}",
        n_names, n_entries, zone.zone_name
    );

    // Decide whether to serve this zone as a dump or via PIR. The expensive
    // bits to avoid are (a) the gzip pass and (b) constructing a multi-GB
    // serialization String we'd then throw away. So:
    //   * --always-dump: serialize fully and compress unconditionally.
    //   * Otherwise: serialize with a cap of SKIP_COMPRESSION_ABOVE_BYTES.
    //     If we overflow the cap, we already know the gzipped form can't
    //     possibly fit under DUMP_THRESHOLD_BYTES (assuming gzip doesn't
    //     compress better than 100×), so we skip both compression and the
    //     rest of the serialization and go straight to PIR mode.
    let dump = if opts.always_dump {
        let serialized = zone::serialize(&zone);
        let gz = zone::compress_bytes(serialized.as_bytes())?;
        eprintln!(
            "Compressed zone: {} B (--always-dump; threshold for dump-only mode would be {} B)",
            gz.len(),
            DUMP_THRESHOLD_BYTES
        );
        Some(gz)
    } else {
        match zone::serialize_bounded(&zone, SKIP_COMPRESSION_ABOVE_BYTES) {
            Ok(serialized) => {
                let gz = zone::compress_bytes(serialized.as_bytes())?;
                eprintln!(
                    "Compressed zone: {} B (threshold for dump-only mode: {} B)",
                    gz.len(),
                    DUMP_THRESHOLD_BYTES
                );
                Some(gz)
            }
            Err(produced) => {
                eprintln!(
                    "Serialized zone exceeds {} B (stopped after producing {} B); \
                     skipping compression and going straight to PIR mode",
                    SKIP_COMPRESSION_ABOVE_BYTES, produced,
                );
                None
            }
        }
    };

    let go_dump_only = opts.always_dump
        || dump.as_ref().is_some_and(|d| d.len() <= DUMP_THRESHOLD_BYTES);
    let zone_name = zone.zone_name.clone();

    if go_dump_only {
        if opts.always_dump {
            eprintln!("Running in dump-only mode (--always-dump).");
        } else {
            eprintln!("Running in dump-only mode (zone fits under threshold).");
        }
        Ok(ServerState {
            zone_name,
            pir: None,
            dump,
            // The hash isn't used in dump-only mode; ship 0 for determinism.
            hash_salt: 0,
        })
    } else {
        // Compute the second moment of the records-per-name distribution
        // that the entry-count estimator needs (see `pick_num_entries`
        // for the formula). One extra pass over `zone.records`,
        // negligible next to the preprocessing that follows.
        let sum_w_sq: usize = zone
            .records
            .values()
            .map(|r| {
                let w = r.a.len() + r.aaaa.len() + r.ns.len();
                w * w
            })
            .sum();
        let num_entries = pick_num_entries(n_names, n_entries, sum_w_sq);
        eprintln!(
            "Preparing {} PIR slot(s) across {} shard(s)...",
            num_entries,
            pir_wrapper::num_databases(num_entries)
        );
        let (buckets, hash_salt) = try_salts(MAX_REHASH_RETRIES, |salt| {
            build_buckets(&zone, num_entries, salt)
        })?;
        drop(zone);
        eprintln!("Using hash salt 0x{:016x} for this build.", hash_salt);
        let pir = preprocess(buckets, num_entries);
        // The dump for large zones is too big to send unconditionally, so we
        // drop it. (Storing it costs memory and gains us nothing: clients
        // wouldn't fetch it.)
        Ok(ServerState {
            zone_name,
            pir: Some(pir),
            dump: None,
            hash_salt,
        })
    }
}

/// Headroom factor in the balls-into-bins max-load bound. The asymptotic
/// expectation is `λ + sqrt((2 + ε) * λ * (E[W²]/E[W]) * ln B)`. Using
/// `2 + ε` with ε > 0 adds finite-sample slack on top of the Gaussian
/// tail constant. We keep ε small because the deviation is sub-linear
/// in it — overshoot directly costs PIR cycles — and rely on the
/// outer `try_salts` / per-bucket-salt rehash loop to mop up the rare
/// case where the realized max exceeds the prediction.
const EPSILON: f64 = 0.2;

/// Assumed share of bucket entries that carry an IPv6 payload (AAAA or NS6).
/// In real zone files today AAAA records are still uncommon and NS glue is
/// overwhelmingly IPv4, so this is a generous upper bound that leaves room
/// for future v6 growth. Average entry size becomes
/// `IPV6_FRACTION * 18 + (1 - IPV6_FRACTION) * 6`.
const IPV6_FRACTION: f64 = 0.2;

/// Picks the smallest PIR entry count (= bucket count) such that the
/// predicted whp-max bucket load (in entry bytes) still fits in one
/// bucket. PIR cost scales linearly with the entry count, so we want the
/// smallest value the bound admits — the rehashing machinery in
/// [`try_salts`] handles the rare realisation that exceeds the
/// prediction.
///
/// Candidates are fleets of fully-used shards: `k` half-size shards
/// (`k * 8192` buckets, up to `HALF_SIZE_PRIMARY_MAX_ENTRIES`) followed
/// by `k` full-size shards (`k * 16384` buckets, from the smallest fleet
/// exceeding that threshold) — exactly the entry counts that
/// `pir_wrapper::get_database_shape` maps back onto `k` fully-used
/// shards of that shape, in increasing bucket-count order. Entry counts
/// that don't fill their last shard would cost the same memory for fewer
/// usable buckets, so they're never worth picking here.
///
/// # The bound
///
/// Each PIR slot is a "bin"; each *name* (not each record) is one
/// "ball" of weight `W = #records-for-this-name`. The slot hash places
/// the name (and therefore all its records) uniformly at random in one
/// bin. Let `N = n_names`, `B = num_entries`, and define
/// `X_b = Σ_{i hits b} W_i` for bin `b`. Then
///
/// ```text
///     E[X_b]   = (N / B) * E[W] = λ        (entries per bin, expectation)
///     Var[X_b] ≈ (N / B) * E[W²] = λ * E[W²]/E[W]
/// ```
///
/// — `Var[X_b]` is the variance of *load*, not of *ball count*; the
/// `E[W²]/E[W]` factor is exactly the inflation from weighted balls
/// versus the Raab–Steger unit-weight case. Bernstein- or Chernoff-style
/// concentration combined with a union bound over the `B` bins gives,
/// with high probability,
///
/// ```text
///     max_b X_b  ≲  λ + sqrt(2 * Var[X_b] * ln B)
///                = λ + sqrt(2 * λ * (E[W²]/E[W]) * ln B)
/// ```
///
/// which collapses to the user's original `λ + sqrt(2 λ ln B)` whenever
/// `W ≡ 1`, and to `λ + sqrt(2 k λ ln B)` for `W ≡ k`. The `(2 + ε)` is
/// the standard finite-sample slack — same place the unit formula
/// would put it.
///
/// # Caveats
///
/// This is a heuristic extension of Raab–Steger (RANDOM '98) rather
/// than a citable theorem in this form; the Gaussian-regime step
/// assumes deviations are small compared to the single largest weight.
/// If one name has an outrageous record count, the rare worst-case
/// where it lands in the worst bucket is handled by re-salting (cheap)
/// rather than by padding the analytical bound (expensive in PIR
/// cycles).
fn pick_num_entries(
    n_names: usize,
    n_entries: usize,
    sum_records_per_name_sq: usize,
) -> usize {
    let N = 1 << LOG2_N;
    let max_shards = N / SIMD_COUNT;
    // The salt header + count field at the start of every bucket eat into
    // the usable payload; reserve them here so the bound matches what
    // `append_entry` will actually accept.
    let bucket_cap_bytes = (BUCKET_BYTES - FIRST_ENTRY_OFFSET) as f64;
    let avg_entry_bytes = IPV6_FRACTION * 18.0 + (1.0 - IPV6_FRACTION) * 6.0;
    let bucket_cap_entries = bucket::MAX_ENTRIES_PER_BUCKET as f64;

    // E[W²] / E[W]. For uniform records-per-name = k this is just k;
    // for variable records-per-name (the realistic case — every TLD
    // zone has names with 2-10 NS records mixed in with name-only A
    // records) it is strictly larger.
    let mean_w = if n_names == 0 {
        1.0
    } else {
        n_entries as f64 / n_names as f64
    };
    let mean_w_sq = if n_names == 0 {
        1.0
    } else {
        sum_records_per_name_sq as f64 / n_names as f64
    };
    let weighted_variance_factor = if mean_w > 0.0 { mean_w_sq / mean_w } else { 1.0 };

    // Fleets of fully-used half-size shards cover the half regime (up to
    // pir_wrapper::HALF_SIZE_PRIMARY_MAX_ENTRIES); from there, fleets of
    // fully-used full-size shards take over — the first full-shard
    // candidate is the smallest fleet whose entry count exceeds the
    // half-size regime.
    let half_shard = SIMD_COUNT * N / 2;
    let full_shard = SIMD_COUNT * N;
    let max_half_shards = pir_wrapper::HALF_SIZE_PRIMARY_MAX_ENTRIES / half_shard;
    let candidates = (1..=max_half_shards).map(|k| k * half_shard)
        .chain(((pir_wrapper::HALF_SIZE_PRIMARY_MAX_ENTRIES / full_shard + 1)..=max_shards).map(|k| k * full_shard));

    let mut num_entries = 0;
    for candidate in candidates {
        num_entries = candidate;
        let buckets = candidate as f64;
        let lambda = n_entries as f64 / buckets;
        // Use ln(max(B, e)) so the bound stays finite/meaningful for
        // tiny zones where B might be tiny too.
        let ln_b = buckets.max(std::f64::consts::E).ln();
        let expected_max = lambda
            + ((2.0 + EPSILON) * lambda * weighted_variance_factor * ln_b).sqrt();
        let expected_bytes = expected_max * avg_entry_bytes;
        if expected_max <= bucket_cap_entries && expected_bytes <= bucket_cap_bytes {
            eprintln!(
                "Sizing: {} names / {} entries / {} buckets -> \
                 lambda {:.2}, E[W²]/E[W] = {:.2}, expected max {:.1} entries \
                 (~{:.0} B, cap {:.0} B)",
                n_names,
                n_entries,
                candidate,
                lambda,
                weighted_variance_factor,
                expected_max,
                expected_bytes,
                bucket_cap_bytes
            );
            return candidate;
        }
    }
    eprintln!(
        "warn: zone needs more than {} buckets to fit the predicted max bucket; \
         capping there — build_buckets may abort with an overflow",
        num_entries
    );
    num_entries
}

/// For every slot, the encoded bucket (already serialized to `BUCKET_BYTES`).
type ShardBuckets = Vec<[u8; BUCKET_BYTES]>;

/// Hard cap on how many distinct per-bucket salts we sweep through before
/// giving up and bubbling the failure up to the global-salt retry loop.
///
/// Discriminators are 14-bit so the birthday bound is small (~128); on a
/// bucket holding ~400 distinct names the per-salt success probability is
/// roughly `e^(-400² / 2 / 16384) ≈ 0.008`, and we expect ~125 attempts on
/// average. 65 536 gives a comfortable headroom for pathological buckets
/// while still failing fast when the bucket is fundamentally over-stuffed
/// (in which case the global-salt loop in [`build_state`] is the right
/// place to retry).
pub const MAX_BUCKET_SALT_ATTEMPTS: u32 = 65_536;

/// Hashes every name in `zone` into a `(shard, slot)` position, picks a
/// per-bucket salt that gives the bucket's names a collision-free
/// 14-bit discriminator set, and writes the resulting A / AAAA / NS-glue
/// entries into the byte representation of the corresponding bucket.
/// Returns either the fully laid-out `Vec<ShardBuckets>` or an error
/// string identifying which bucket overflowed or which name set could
/// not be resolved within [`MAX_BUCKET_SALT_ATTEMPTS`] — both signals the
/// caller (via [`try_salts`]) uses to retry with a different global salt.
///
/// Because discriminators depend on a per-bucket salt that is itself part
/// of the answer, we can no longer write entries while walking the zone.
/// Instead, we first group names by bucket (a pointer-only index, so the
/// overhead is `~n_names * 8 B`) and then process the groups one bucket at
/// a time. The per-bucket pass first sweeps `bucket_salt` for a value
/// producing a collision-free discriminator set, then writes the salt
/// header and entries in one go.
#[instrument(skip_all)]
fn build_buckets(
    zone: &Zone,
    num_entries: usize,
    global_salt: u64,
) -> Result<Vec<ShardBuckets>, String> {
    // Pass 1: group every name by `(shard, slot)`. We store `&str` borrows
    // into `zone.records`'s keys; the lookup map then has one slot per PIR
    // bucket holding a (typically tiny) vector of pointers.
    //
    // Every shard gets its full complement of slots even when
    // `num_entries` doesn't fill the last one — the PIR databases store
    // that many entries regardless, and `bucket::slot` never targets the
    // padding slots.
    let entries_per_shard = bucket::entries_per_shard(num_entries);
    let shard_count = pir_wrapper::num_databases(num_entries);
    let mut names_in_bucket: Vec<Vec<Vec<&str>>> = (0..shard_count)
        .map(|_| (0..entries_per_shard).map(|_| Vec::new()).collect())
        .collect();
    for name in zone.records.keys() {
        let (sh, pri) = bucket::slot(name, num_entries, global_salt);
        names_in_bucket[sh][pri].push(name.as_ref());
    }

    // Pass 2: build each bucket. We process them shard-by-shard so the
    // per-bucket name lists for the current shard stay hot in cache.
    let mut out: Vec<ShardBuckets> = (0..shard_count)
        .map(|_| vec![[0u8; BUCKET_BYTES]; entries_per_shard])
        .collect();
    let mut total_entries = 0usize;
    // Track the realised worst-case bucket load and salt-search effort
    // so the operator can compare against the pick_num_entries
    // prediction after the build settles.
    let mut max_entries_in_any_bucket: usize = 0;
    let mut max_salt_attempts: u32 = 0;
    // Reuse a single HashSet across all per-bucket salt attempts. It's
    // sized for the worst case once and then cleared between probes,
    // which avoids the per-attempt allocator churn.
    let mut disc_seen: HashSet<u16> = HashSet::with_capacity(bucket::MAX_ENTRIES_PER_BUCKET);

    for sh in 0..shard_count {
        for pri in 0..entries_per_shard {
            let names = &names_in_bucket[sh][pri];
            if names.is_empty() {
                // Empty bucket: still need a valid salt header (zero) and
                // a zero count byte. The all-zero state we initialized
                // with already satisfies both invariants.
                continue;
            }

            let (bucket_salt, attempts) =
                match find_bucket_salt(names, global_salt, &mut disc_seen) {
                    Some(ok) => ok,
                    None => {
                        return Err(format!(
                            "shard {} bucket {}: could not find a salt avoiding \
                             14-bit discriminator collisions among {} names after \
                             {} attempts; bucket likely overfull — raise shard \
                             count (currently {}) or split the zone",
                            sh, pri, names.len(), MAX_BUCKET_SALT_ATTEMPTS, shard_count
                        ));
                    }
                };
            if attempts > max_salt_attempts {
                max_salt_attempts = attempts;
            }

            let bucket_buf = &mut out[sh][pri];
            bucket::write_bucket_salt(bucket_buf, bucket_salt);
            let mut pos: u16 = FIRST_ENTRY_OFFSET as u16;
            let on_overflow = |e: bucket::BucketError| {
                format!(
                    "shard {} bucket {}: {}; raise shard count (currently {}) \
                     or split the zone",
                    sh, pri, e, shard_count
                )
            };

            // Count entries actually written to this bucket so we can
            // surface the realised worst-case load after the build.
            let entries_before_bucket = total_entries;
            for name in names {
                let disc = bucket::discriminator(name, global_salt, bucket_salt);
                let rec = zone
                    .records
                    .get(*name)
                    .expect("name was sourced from zone.records");
                for ip in &rec.a {
                    bucket::append_entry(bucket_buf, &mut pos, KIND_A, disc, &Record::A(*ip))
                        .map_err(on_overflow)?;
                    total_entries += 1;
                }
                for ip in &rec.aaaa {
                    bucket::append_entry(bucket_buf, &mut pos, KIND_AAAA, disc, &Record::Aaaa(*ip))
                        .map_err(on_overflow)?;
                    total_entries += 1;
                }
                // For NS, resolve glue: prefer an in-zone A/AAAA, else dummy IP.
                for &ns_idx in &rec.ns {
                    let nshost = zone.ns_hostname(ns_idx);
                    let ip = zone.glue_for(nshost).unwrap_or_else(|| dummy_ip(nshost));
                    let (kind, rec_v) = match ip {
                        IpAddr::V4(_) => (KIND_NS4, Record::Ns(ip)),
                        IpAddr::V6(_) => (KIND_NS6, Record::Ns(ip)),
                    };
                    bucket::append_entry(bucket_buf, &mut pos, kind, disc, &rec_v)
                        .map_err(on_overflow)?;
                    total_entries += 1;
                }
            }
            let entries_this_bucket = total_entries - entries_before_bucket;
            if entries_this_bucket > max_entries_in_any_bucket {
                max_entries_in_any_bucket = entries_this_bucket;
            }
        }
        // Drop the per-shard pointer index once we're done with it.
        names_in_bucket[sh] = Vec::new();
    }

    eprintln!(
        "Placed {} records into buckets; max actually observed: {} entries in \
         one bucket (cap {}), worst-case per-bucket salt search took {} \
         attempt(s) (cap {})",
        total_entries,
        max_entries_in_any_bucket,
        bucket::MAX_ENTRIES_PER_BUCKET,
        max_salt_attempts,
        MAX_BUCKET_SALT_ATTEMPTS,
    );
    Ok(out)
}

/// Sweeps candidate per-bucket salts until one produces a collision-free
/// 14-bit discriminator set across `names`. Returns the chosen salt and
/// the number of distinct salts that had to be tried (so the caller can
/// surface worst-case search effort), or `None` if
/// [`MAX_BUCKET_SALT_ATTEMPTS`] is exhausted (caller bubbles up to the
/// global-salt loop in that case).
///
/// `disc_seen` is passed in (rather than allocated here) so the caller
/// can reuse a single `HashSet` across every bucket and skip the per-call
/// allocator round-trip.
fn find_bucket_salt(
    names: &[&str],
    global_salt: u64,
    disc_seen: &mut HashSet<u16>,
) -> Option<(u32, u32)> {
    for salt in 0..MAX_BUCKET_SALT_ATTEMPTS {
        disc_seen.clear();
        let mut ok = true;
        for name in names {
            let d = bucket::discriminator(name, global_salt, salt);
            if !disc_seen.insert(d) {
                ok = false;
                break;
            }
        }
        if ok {
            // `salt` is 0-based; the attempt count is one more.
            return Some((salt, salt + 1));
        }
    }
    None
}

/// Tries to allocate the PIRDatabase backing buffer on 2 MiB huge pages, and
/// falls back to a regular anonymous mmap if the kernel can't satisfy the
/// `MAP_HUGETLB` request (e.g. no huge pages reserved on the host).
fn alloc_pir_backing(bytes_needed: usize) -> MmapMut {
    match pir_wrapper::alloc_huge(bytes_needed) {
        Ok(m) => {
            eprintln!("  using huge pages for PIR backing memory");
            m
        }
        Err(e) => {
            eprintln!(
                "warn: huge-page allocation failed ({}); falling back to regular pages",
                e
            );
            MmapOptions::new()
                .len(bytes_needed)
                .map_anon()
                .expect("anonymous mmap allocation")
        }
    }
}

/// Builds the PIR engine state from the bucket-bytes produced by
/// [`build_buckets`].
///
/// Allocates one giant `mmap` (huge pages preferred — see
/// [`alloc_pir_backing`]), leaks it to obtain `&'static mut` slices, and
/// hands each shard a non-overlapping chunk. The slab is then sliced
/// shard-by-shard into a [`PIRDatabase`] (each populated via `set_db`).
/// Returns the resulting fleet wrapped in a [`PirState`].
///
/// Memory cost is `shard_count * PIRDatabase::required_memory_general(...)`
/// — for `LOG2_N = 11`, that's about 768 MB per full-group shard, or half
/// that for the half-size index group small fleets use. Preprocessing time
/// per shard scales linearly with `N²`, several seconds in release mode.
#[instrument(skip_all)]
fn preprocess(buckets: Vec<ShardBuckets>, num_entries: usize) -> PirState {
    let N = 1 << LOG2_N;
    let shard_count = buckets.len();
    assert_eq!(shard_count, pir_wrapper::num_databases(num_entries));
    let Zt = Zn::new(PRIMARY_PLAIN_MODULUS as u64);
    let R = PlainRing::new(Zt, N, [Zt.neg_one()]);
    let Zq = Zn::new(FIXED_Q as u64);
    let C = CipherRing::new(Zq, N, [Zq.neg_one()]);

    // Allocate one big slab of memory holding the backing storage for every
    // PIRDatabase, preferring 2 MiB huge pages to keep TLB pressure low
    // during the streaming inner-product loop. Leak it so we get `'static`
    // references, which sidesteps the self-referential-struct problem.
    //
    // The primary index group must match what the client derives from the
    // entry count we advertise in INFO: small databases use the half-size
    // sub-group (half the buckets per shard, and the client omits the
    // conjugated primary query ciphertext), large ones the full Galois
    // group. The secondary database (built per-query inside process_query)
    // gets its own, independently derived group.
    let (primary_index_group, _) = pir_wrapper::get_database_shape(num_entries);
    let per_db = PIRDatabase::required_memory_general(primary_index_group, N);
    let total = per_db * shard_count;
    let bytes_needed = total * std::mem::size_of::<CompressedZnx8El<26>>();
    eprintln!(
        "Allocating preprocessed state: {} shards * {} slots = {} bytes",
        shard_count, per_db, bytes_needed
    );
    let leaked_mmap: &'static mut MmapMut = Box::leak(Box::new(alloc_pir_backing(bytes_needed)));
    let base_ptr: *mut CompressedZnx8El<26> =
        leaked_mmap.as_mut_ptr() as *mut CompressedZnx8El<26>;
    // The mapping is always page-aligned (>= 4 KiB, and 2 MiB when huge
    // pages succeed), so the 32-byte alignment CompressedZnx8El<26> needs
    // is automatically satisfied.
    debug_assert!((base_ptr as usize) % std::mem::align_of::<CompressedZnx8El<26>>() == 0);

    let mut dbs: Vec<PIRDatabase<'static>> = Vec::with_capacity(shard_count);
    for (shard_idx, slot_buckets) in buckets.into_iter().enumerate() {
        // Each shard gets its own non-overlapping `&'static mut [..]` slice
        // of the leaked backing buffer.
        let chunk: &'static mut [CompressedZnx8El<26>] = unsafe {
            std::slice::from_raw_parts_mut(base_ptr.add(shard_idx * per_db), per_db)
        };
        let mut db = PIRDatabase::create(
            R.clone(),
            C.clone(),
            primary_index_group,
            UsedSeeds::FirstSet,
            chunk,
        );

        // Encode each bucket as one ring element: pack four bucket bytes
        // worth of 10-bit values into N coefficients via the canonical
        // packing in `bucket::pack_bytes_into_coeffs`. This uses the full
        // 10 bits of `t = 1025` rather than the 8-bit subset the previous
        // layout did, giving us BUCKET_BYTES = 2560 (= N/4 * 5) of payload
        // per slot instead of 2048.
        let entries_iter = slot_buckets.iter().map(|b: &[u8; BUCKET_BYTES]| {
            let coeffs = bucket::pack_bytes_into_coeffs(b);
            R.from_canonical_basis(coeffs.iter().map(|&c| {
                Zt.get_ring().from_int_promise_reduced(c as i64)
            }))
        });
        let started = std::time::Instant::now();
        db.set_db(entries_iter);
        eprintln!(
            "Preprocessed shard {}/{} in {:.1}s",
            shard_idx + 1,
            shard_count,
            started.elapsed().as_secs_f64()
        );
        dbs.push(db);
    }

    PirState { dbs, num_entries }
}

/// Per-connection worker. Reads framed commands from `conn` in a loop and
/// writes the corresponding reply on the same socket, exiting cleanly
/// when the peer half-closes. `CMD_QUERY` against a dump-only server (or
/// `CMD_DUMP` against a PIR-only server) returns the
/// [`RESP_UNAVAILABLE`] sentinel byte instead of erroring out.
#[instrument(skip_all)]
fn handle(mut conn: TcpStream, state: &ServerState) -> std::io::Result<()> {
    // Drain the socket: a single connection may carry CMD_INFO followed
    // by CMD_QUERY / CMD_DUMP. The client closes the socket to terminate.
    loop {
        let req = match read_frame(&mut conn) {
            Ok(r) => r,
            // The peer half-closed cleanly between commands.
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e),
        };
        if req.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "empty request",
            ));
        }
        match req[0] {
            CMD_INFO => {
                let mut flags = 0u8;
                if state.pir.is_some() {
                    flags |= FLAG_PIR_AVAILABLE;
                }
                if state.dump.is_some() {
                    flags |= FLAG_DUMP_AVAILABLE;
                }
                let resp = encode_info(&InfoResponse {
                    flags,
                    num_entries: state.pir.as_ref().map(|p| p.num_entries as u32).unwrap_or(0),
                    zone_name: state.zone_name.clone(),
                    dump_size: state.dump.as_ref().map(|d| d.len() as u32).unwrap_or(0),
                    hash_salt: state.hash_salt,
                });
                write_frame(&mut conn, &resp)?;
            }
            CMD_QUERY => {
                let Some(pir) = state.pir.as_ref() else {
                    write_frame(&mut conn, &RESP_UNAVAILABLE)?;
                    continue;
                };
                let query = &req[1..];
                let dbs_ref = &pir.dbs;
                let dbs_fn = (0..dbs_ref.len()).map_fn(|i| &dbs_ref[i]);
                let reply = process_query(dbs_fn, query.iter().copied(), None);
                write_frame(&mut conn, &reply)?;
            }
            CMD_DUMP => {
                let Some(dump) = state.dump.as_ref() else {
                    write_frame(&mut conn, &RESP_UNAVAILABLE)?;
                    continue;
                };
                write_frame(&mut conn, dump)?;
            }
            other => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("unknown command {}", other),
                ));
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::net::Ipv4Addr;

    use super::*;
    use crate::dns::bucket::{decode_bucket_matching, discriminator, read_bucket_salt, slot};

    /// Entry count used by most tests: exactly one half-size shard, the
    /// smallest fleet the sizing logic would ever pick.
    const TEST_NUM_ENTRIES: usize = SIMD_COUNT * (1 << LOG2_N) / 2;

    /// Writes the supplied zone-file text to a temp file and parses it.
    fn parse_zone(text: &str) -> Zone {
        let mut path = std::env::temp_dir();
        // Unique per call to allow tests to run in parallel.
        let pid = std::process::id();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        path.push(format!("pir_dns_zone_{}_{}.txt", pid, nonce));
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(text.as_bytes()).unwrap();
        }
        let zone = Zone::load(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).ok();
        zone
    }

    /// End-to-end lookup that bypasses PIR: builds the bucket layout the
    /// server would preprocess and pulls the entries that belong to `name`
    /// straight out. Exercises the same hashing / glue / discriminator
    /// logic that real queries depend on.
    fn lookup(zone: &Zone, num_entries: usize, name: &str) -> Vec<Record> {
        let salt: u64 = 0;
        let buckets = build_buckets(zone, num_entries, salt).expect("build_buckets");
        let canon = bucket::canonical(name);
        let (sh, pri) = slot(&canon, num_entries, salt);
        let bucket_salt = read_bucket_salt(&buckets[sh][pri]);
        let disc = discriminator(&canon, salt, bucket_salt);
        decode_bucket_matching(&buckets[sh][pri], disc)
    }

    /// Like [`lookup`], but also routes the bucket bytes through the
    /// 10-bit coefficient packing the server's `preprocess` applies and
    /// the matching unpack the client's `coeffs_to_bucket` applies. This
    /// catches any drift between the encoder and decoder of the PIR/DNS
    /// boundary that the plain `lookup` would miss.
    fn lookup_via_coeffs(zone: &Zone, num_entries: usize, name: &str) -> Vec<Record> {
        let salt: u64 = 0;
        let buckets = build_buckets(zone, num_entries, salt).expect("build_buckets");
        let canon = bucket::canonical(name);
        let (sh, pri) = slot(&canon, num_entries, salt);
        let coeffs = bucket::pack_bytes_into_coeffs(&buckets[sh][pri]);
        let bucket_bytes = bucket::unpack_coeffs_into_bytes(&coeffs);
        let bucket_salt = read_bucket_salt(&bucket_bytes);
        let disc = discriminator(&canon, salt, bucket_salt);
        decode_bucket_matching(&bucket_bytes, disc)
    }

    #[test]
    fn glue_resolves_to_in_zone_a_record() {
        let zone = parse_zone(
            "apex.org.\t3600\tin\tns\tns1.delegate.org.\n\
             ns1.delegate.org.\t3600\tin\ta\t10.0.0.1\n",
        );
        let recs = lookup(&zone, TEST_NUM_ENTRIES, "apex.org");
        let ns_ips: Vec<IpAddr> = recs
            .iter()
            .filter_map(|r| match r {
                Record::Ns(ip) => Some(*ip),
                _ => None,
            })
            .collect();
        assert_eq!(ns_ips, vec![IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1))]);
    }

    #[test]
    fn out_of_zone_ns_uses_placeholder() {
        let zone = parse_zone(
            "apex.org.\t3600\tin\tns\tns1.elsewhere.com.\n",
        );
        let recs = lookup(&zone, TEST_NUM_ENTRIES, "apex.org");
        let ns_ips: Vec<IpAddr> = recs
            .iter()
            .filter_map(|r| match r {
                Record::Ns(ip) => Some(*ip),
                _ => None,
            })
            .collect();
        assert_eq!(ns_ips.len(), 1);
        match ns_ips[0] {
            IpAddr::V4(ip) => {
                let o = ip.octets();
                assert_eq!([o[0], o[1], o[2]], [192, 0, 2], "expected TEST-NET-1 dummy");
            }
            IpAddr::V6(_) => panic!("expected v4 placeholder"),
        }
    }

    /// Reproduces the user's bug 1: an NS pointing to an in-zone glue name
    /// must resolve to that real glue, not to a TEST-NET-1 placeholder.
    #[test]
    fn mixed_in_and_out_of_zone_glue() {
        let zone = parse_zone(
            "apex.org.\t3600\tin\tns\tns1.outside.com.\n\
             apex.org.\t3600\tin\tns\tns2.inside.org.\n\
             apex.org.\t3600\tin\tns\tns3.outside.eu.\n\
             ns2.inside.org.\t3600\tin\ta\t178.132.200.29\n",
        );
        let recs = lookup(&zone, TEST_NUM_ENTRIES, "apex.org");
        let ns_ips: Vec<IpAddr> = recs
            .iter()
            .filter_map(|r| match r {
                Record::Ns(ip) => Some(*ip),
                _ => None,
            })
            .collect();
        assert!(
            ns_ips.contains(&IpAddr::V4(Ipv4Addr::new(178, 132, 200, 29))),
            "expected real glue IP for ns2.inside.org among {:?}",
            ns_ips
        );
    }

    /// Reproduces the user's bug 2: a name with a direct A record (here, a
    /// glue record three labels deep) must be retrievable by its full name.
    #[test]
    fn direct_a_lookup_for_deep_glue_name() {
        let zone = parse_zone(
            "apex.org.\t3600\tin\tns\tns2.inside.org.\n\
             ns2.inside.org.\t3600\tin\ta\t178.132.200.29\n",
        );
        let recs = lookup(&zone, TEST_NUM_ENTRIES, "ns2.inside.org");
        assert_eq!(recs, vec![Record::A(Ipv4Addr::new(178, 132, 200, 29))]);
    }

    #[test]
    fn nonexistent_name_returns_empty() {
        let zone = parse_zone("only.org.\t3600\tin\ta\t1.2.3.4\n");
        assert!(lookup(&zone, TEST_NUM_ENTRIES, "missing.org").is_empty());
    }

    #[test]
    fn apex_with_a_and_aaaa_returns_both() {
        let zone = parse_zone(
            "host.org.\t3600\tin\ta\t1.2.3.4\n\
             host.org.\t3600\tin\taaaa\t2001:db8::1\n",
        );
        let recs = lookup(&zone, TEST_NUM_ENTRIES, "host.org");
        assert!(recs.contains(&Record::A(Ipv4Addr::new(1, 2, 3, 4))));
        assert!(recs.contains(&Record::Aaaa("2001:db8::1".parse().unwrap())));
    }

    /// End-to-end check on the real bundled root zone: the sizing bound,
    /// the bucket build, and the retrieval of glue records all have to
    /// agree on live data, not just on the synthetic zones above.
    #[test]
    fn root_zone_real_glue() {
        // Bundled data files live at the workspace root, not in this
        // package's directory (which is what `cargo test` makes the CWD).
        let zone = Zone::load(concat!(env!("CARGO_MANIFEST_DIR"), "/../../root.txt"))
            .expect("root.txt missing");
        let n_names = zone.records.len();
        let mut n_entries = 0usize;
        let mut sum_w_sq = 0usize;
        for r in zone.records.values() {
            let w = r.a.len() + r.aaaa.len() + r.ns.len();
            n_entries += w;
            sum_w_sq += w * w;
        }
        let num_entries = pick_num_entries(n_names, n_entries, sum_w_sq);
        assert_eq!(
            num_entries, TEST_NUM_ENTRIES,
            "expected the root zone to fit in a single half-size shard"
        );

        // a.root-servers.net has both an A and an AAAA glue record; the
        // bucket for its own name must hand both back.
        let recs = lookup(&zone, num_entries, "a.root-servers.net");
        assert!(
            recs.contains(&Record::A(Ipv4Addr::new(198, 41, 0, 4))),
            "expected a.root-servers.net A 198.41.0.4, got {:?}",
            recs
        );
        assert!(
            recs.contains(&Record::Aaaa("2001:503:ba3e::2:30".parse().unwrap())),
            "expected a.root-servers.net AAAA 2001:503:ba3e::2:30, got {:?}",
            recs
        );

        // The .org delegation names a0.org.afilias-nst.info, whose A
        // record is in-zone glue; the NS list must resolve through it.
        let recs = lookup(&zone, num_entries, "org");
        let ns_ips: Vec<IpAddr> = recs
            .iter()
            .filter_map(|r| match r {
                Record::Ns(ip) => Some(*ip),
                _ => None,
            })
            .collect();
        assert!(
            ns_ips.contains(&IpAddr::V4(Ipv4Addr::new(199, 19, 56, 1))),
            "expected real glue 199.19.56.1 in NS list of org, got {:?}",
            ns_ips
        );
    }

    // ---- per-bucket salt tests -------------------------------------------

    /// Forces many names into very few buckets so that 14-bit
    /// discriminators collide on salt 0 — the build must succeed anyway
    /// because each bucket's salt is searched independently. Then every
    /// name must still be retrievable by going through the same
    /// `read_bucket_salt -> discriminator -> decode_bucket_matching` flow
    /// the real client uses.
    #[test]
    fn per_bucket_salt_resolves_disc_collisions() {
        // 600 names hashed into TEST_NUM_ENTRIES = 8192 buckets → each
        // bucket has on average ~0.04 names, but the variance is high
        // enough (and we generate enough names) that several buckets will
        // hold a few names whose salt-0 discriminators happen to clash.
        // The point isn't to force a specific bucket into the slow path;
        // it's to verify the build succeeds at all, and that lookups
        // round-trip correctly afterwards.
        let mut text = String::new();
        for i in 0..600 {
            text.push_str(&format!(
                "host{:04}.testzone.\t3600\tIN\tA\t10.0.{}.{}\n",
                i,
                (i >> 8) & 0xff,
                i & 0xff,
            ));
        }
        let zone = parse_zone(&text);
        // one half-size shard keeps the test cheap (~8k buckets).
        let buckets = build_buckets(&zone, TEST_NUM_ENTRIES, 0).expect("build_buckets");

        for i in 0..600 {
            let name = format!("host{:04}.testzone", i);
            let recs = lookup(&zone, TEST_NUM_ENTRIES, &name);
            assert_eq!(
                recs,
                vec![Record::A(Ipv4Addr::new(10, 0, ((i >> 8) & 0xff) as u8, (i & 0xff) as u8))],
                "wrong records for {} (bucket layout: {:?})",
                name,
                {
                    let canon = bucket::canonical(&name);
                    slot(&canon, TEST_NUM_ENTRIES, 0)
                }
            );
        }

        // The build must have actually written non-default salts in at
        // least one bucket if collisions ever happened. Lookups already
        // verified correctness, so this is just an integrity sniff.
        let nonzero_salts =
            buckets[0].iter().filter(|b| read_bucket_salt(b) != 0).count();
        // Either some buckets needed a non-zero salt, or every bucket
        // happened to land on salt 0 — both are valid; this assertion
        // exists only to print a useful number when the test breaks.
        assert!(
            nonzero_salts <= TEST_NUM_ENTRIES,
            "impossible: more nonzero salts than buckets ({})",
            nonzero_salts
        );
    }

    /// End-to-end via the coefficient packing path: every name's lookup
    /// must match whether we read the bucket bytes directly or detour
    /// through pack_bytes_into_coeffs + unpack_coeffs_into_bytes. Locks
    /// down the new 10-bit packing against silent drift in either
    /// direction.
    #[test]
    fn lookup_via_coeffs_matches_direct() {
        let zone = parse_zone(
            "alpha.example.\t3600\tIN\tA\t10.0.0.1\n\
             alpha.example.\t3600\tIN\tAAAA\t2001:db8::1\n\
             beta.example.\t3600\tIN\tNS\tns1.gamma.example.\n\
             beta.example.\t3600\tIN\tNS\tns2.gamma.example.\n\
             gamma.example.\t3600\tIN\tA\t10.0.0.42\n\
             ns1.gamma.example.\t3600\tIN\tA\t10.0.0.43\n\
             ns2.gamma.example.\t3600\tIN\tAAAA\t2001:db8::43\n",
        );
        for name in [
            "alpha.example",
            "beta.example",
            "gamma.example",
            "ns1.gamma.example",
            "ns2.gamma.example",
            "missing.example",
        ] {
            let direct = lookup(&zone, TEST_NUM_ENTRIES, name);
            let via_coeffs = lookup_via_coeffs(&zone, TEST_NUM_ENTRIES, name);
            assert_eq!(
                direct, via_coeffs,
                "name {} produced different records via coeffs",
                name
            );
        }
    }

    /// Stress test: pack a single bucket with ~250 names (close to the
    /// `MAX_ENTRIES_PER_BUCKET` cap, well into the discriminator birthday
    /// regime — expected collisions per salt attempt is ~2). The
    /// per-bucket salt search must still find a non-colliding layout
    /// within `MAX_BUCKET_SALT_ATTEMPTS` and every name must resolve to
    /// its own record.
    #[test]
    fn per_bucket_salt_handles_high_load_bucket() {
        // 250 names × 6 B/entry + 5 B header = 1505 B, comfortably inside
        // BUCKET_BYTES (2048). With 14-bit disc, P(no collision | salt) ≈
        // e^(-250²/2/16384) ≈ 0.150, so the search succeeds within tens of
        // attempts on average.
        let mut text = String::new();
        for i in 0..250 {
            text.push_str(&format!(
                "h{:04}.t.\t3600\tIN\tA\t10.1.{}.{}\n",
                i,
                (i >> 8) & 0xff,
                i & 0xff,
            ));
        }
        let zone = parse_zone(&text);
        // A single shard with all names sharing the same suffix doesn't
        // guarantee they all hit one bucket (slot hashing scatters them),
        // but the largest bucket will hold dozens of names — enough to
        // routinely require non-zero salts. Either way, every name must
        // round-trip.
        for i in 0..250 {
            let name = format!("h{:04}.t", i);
            let recs = lookup(&zone, TEST_NUM_ENTRIES, &name);
            assert_eq!(
                recs,
                vec![Record::A(Ipv4Addr::new(10, 1, ((i >> 8) & 0xff) as u8, (i & 0xff) as u8))],
                "wrong records for {}",
                name
            );
        }
    }

    /// `find_bucket_salt` must return `Some` for any name set whose size
    /// is comfortably below the 14-bit discriminator space, and `None`
    /// for a name set that cannot be fit (sanity-checks the failure path
    /// that bubbles up to the global salt loop).
    #[test]
    fn find_bucket_salt_succeeds_on_small_buckets() {
        let names_owned: Vec<String> = (0..50)
            .map(|i| format!("name{:03}.example", i))
            .collect();
        let names: Vec<&str> = names_owned.iter().map(|s| s.as_str()).collect();
        let mut seen = HashSet::new();
        let (salt, _attempts) = find_bucket_salt(&names, 0, &mut seen)
            .expect("50 names must fit in 14-bit disc space within MAX_BUCKET_SALT_ATTEMPTS");

        // Confirm the returned salt actually produces a collision-free
        // discriminator set — i.e. that find_bucket_salt isn't lying.
        let mut check = HashSet::new();
        for name in &names {
            assert!(
                check.insert(bucket::discriminator(name, 0, salt)),
                "salt 0x{:08x} returned by find_bucket_salt collides on {}",
                salt,
                name
            );
        }
    }

    // ---- rehash retry tests ----------------------------------------------

    #[test]
    fn try_salts_returns_first_successful_salt() {
        let (val, salt) = try_salts::<&'static str, _>(8, |s| {
            if s == salt_for_attempt(0) {
                Ok("first")
            } else {
                unreachable!("should have succeeded on attempt 0")
            }
        })
        .unwrap();
        assert_eq!(val, "first");
        assert_eq!(salt, salt_for_attempt(0));
    }

    #[test]
    fn try_salts_retries_until_success() {
        let mut seen: Vec<u64> = Vec::new();
        let (val, salt) = try_salts::<u64, _>(8, |s| {
            seen.push(s);
            if seen.len() < 3 {
                Err(format!("forced fail #{}", seen.len()))
            } else {
                Ok(s)
            }
        })
        .unwrap();
        assert_eq!(seen.len(), 3);
        assert_eq!(seen[0], salt_for_attempt(0));
        assert_eq!(seen[1], salt_for_attempt(1));
        assert_eq!(seen[2], salt_for_attempt(2));
        assert_eq!(salt, salt_for_attempt(2));
        assert_eq!(val, salt);
    }

    #[test]
    fn try_salts_gives_up_after_max_attempts() {
        let mut calls = 0usize;
        let result: std::io::Result<((), u64)> = try_salts(4, |_| {
            calls += 1;
            Err("always fails".to_owned())
        });
        assert!(result.is_err(), "expected exhaustion");
        assert_eq!(calls, 4);
    }

    #[test]
    fn salt_for_attempt_distributes_non_zero_attempts() {
        assert_eq!(salt_for_attempt(0), 0);
        // Two consecutive non-zero salts must differ.
        assert_ne!(salt_for_attempt(1), salt_for_attempt(2));
    }
}
