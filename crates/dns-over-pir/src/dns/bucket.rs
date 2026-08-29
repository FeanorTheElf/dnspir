//! Hashing, bucket layout and ring-element encoding for the DNS-over-PIR PoC.
//!
//! Each PIR slot holds one ring element of `Z_1025[X] / (X^N + 1)` with
//! `N = 2048`. The plaintext modulus `t = 1025` admits 10 usable bits per
//! coefficient; we pack four coefficients into five bytes (4 × 10 bits = 40
//! bits = 5 bytes), giving `N / 4 * 5 = 2560` payload bytes per bucket
//! rather than the 2048 the old "one byte per coefficient" layout
//! achieved. Inside that buffer we store a sequence of fixed-format
//! entries:
//!
//! ```text
//! bucket := [bucket_salt: u32 LE] [entry_count: u16 LE] [entry_0] [entry_1] ...
//! entry  := [tag: u16 LE] [rdata]
//!   tag.bits[14..16] = kind  (one of 0..=3)
//!   tag.bits[0..14]  = discriminator (14-bit fingerprint of the name)
//!   kind = A    -> rdata = 4 B  (IPv4 of the record)
//!   kind = AAAA -> rdata = 16 B (IPv6 of the record)
//!   kind = NS4  -> rdata = 4 B  (IPv4 of the delegated nameserver)
//!   kind = NS6  -> rdata = 16 B (IPv6 of the delegated nameserver)
//! ```
//!
//! An A or NS4 entry therefore takes 6 B; AAAA / NS6 take 18 B.
//!
//! With only 14 bits of fingerprint the birthday bound is `~2^7 = 128`, which
//! sits well below the maximum entries one bucket can hold — collisions are
//! likely on any non-trivial zone. We accept that and resolve it at build
//! time by making the discriminator depend on a per-bucket salt stored in
//! the first 4 bytes of the bucket. The builder searches that salt space
//! independently for each bucket (see `server::build_buckets`), so a single
//! discriminator clash only triggers a tiny per-bucket retry rather than a
//! full zone-wide rebuild.
//!
//! The client extracts the bucket salt from the decrypted PIR reply (via
//! [`read_bucket_salt`]) before computing the discriminator that
//! [`decode_bucket_matching`] should match. The 10-bit ↔ byte coefficient
//! packing is owned by [`pack_bytes_into_coeffs`] /
//! [`unpack_coeffs_into_bytes`] and is the single source of truth for the
//! PIR/DNS byte/coefficient boundary.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use tracing::instrument;

use pir::base_pir::{LOG2_N, SIMD_COUNT};
use pir::pir_wrapper::get_database_shape;

/// Ring dimension `N = 2^LOG2_N` of the PIR plaintext ring. Mirrored here
/// so the DNS layer doesn't have to import the PIR side just for one
/// constant.
pub const N: usize = 1 << LOG2_N;

/// Plaintext modulus of the PIR ring. Coefficients live in
/// `0..PLAIN_MODULUS`; we only use values that fit in [`COEFF_BITS`].
pub const PLAIN_MODULUS: u64 = 1025;

/// Number of bits per plaintext coefficient that carry bucket payload.
/// `PLAIN_MODULUS = 1025` admits 10 distinct top bits (values `0..=1023`);
/// the one unreachable value (1024) is never written.
pub const COEFF_BITS: u32 = 10;

/// Coefficients packed per byte group of the bucket payload. The boundary
/// is byte-aligned because `COEFFS_PER_GROUP * COEFF_BITS = 40` is a
/// multiple of eight — `COEFFS_PER_GROUP * COEFF_BITS / 8 =
/// COEFFS_PER_GROUP_BYTES`.
pub const COEFFS_PER_GROUP: usize = 4;
/// Byte size of one packed coefficient group.
pub const COEFFS_PER_GROUP_BYTES: usize = 5;

/// Payload bytes carried by one PIR slot. Each ring element has `N`
/// coefficients of 10 useful bits, packed `COEFFS_PER_GROUP` at a time into
/// `COEFFS_PER_GROUP_BYTES` bytes — total `N / 4 * 5 = 2560` bytes for the
/// default `N = 2048`.
pub const BUCKET_BYTES: usize = N / COEFFS_PER_GROUP * COEFFS_PER_GROUP_BYTES;

/// Number of slots in one PIR shard of a database with `num_entries`
/// total slots, each slot being its own bucket. Small databases (up to
/// `pir_wrapper::HALF_SIZE_PRIMARY_MAX_ENTRIES` slots) use half-size
/// primary databases so the client can skip the conjugated primary query
/// ciphertext — see [`get_database_shape`], which both sides derive from
/// the `num_entries` advertised in `INFO`. With the default parameters
/// this is 8 × 1024 = 8192 slots per shard for databases of up to
/// `N²/16` slots, and 8 × 2048 = 16384 beyond that.
pub fn entries_per_shard(num_entries: usize) -> usize {
    SIMD_COUNT * get_database_shape(num_entries).0.group_order()
}

/// Tag-kind code for an `A` record (IPv4 of the queried name itself).
pub const KIND_A: u8 = 0;
/// Tag-kind code for an `AAAA` record (IPv6 of the queried name itself).
pub const KIND_AAAA: u8 = 1;
/// Tag-kind code for an `NS` record whose resolved glue is an IPv4 address.
pub const KIND_NS4: u8 = 2;
/// Tag-kind code for an `NS` record whose resolved glue is an IPv6 address.
pub const KIND_NS6: u8 = 3;

/// Number of bits of the 2-byte tag reserved for the discriminator. The
/// remaining 2 bits encode the kind.
pub const DISC_BITS: u32 = 14;
/// Bitmask for the discriminator portion of a packed tag (the low
/// `DISC_BITS` bits).
pub const DISC_MASK: u16 = (1u16 << DISC_BITS) - 1;

/// Sanity ceiling on the entry-count field, which is two bytes wide. With
/// `BUCKET_BYTES = 2560` and a 6-byte minimum entry the byte budget caps
/// us at ~425 entries long before this matters; the value exists only so
/// a corrupt count field can't underflow downstream bookkeeping.
pub const MAX_ENTRIES_PER_BUCKET: usize = u16::MAX as usize;

/// Wire size of the per-bucket salt header that precedes the count field.
/// Stored as little-endian `u32` in `bucket[0..4]`. The builder is allowed
/// to pick any value here; the client reads it back to derive the
/// discriminator for the bucket it just received.
pub const BUCKET_SALT_BYTES: usize = 4;

/// Wire size of the entry-count field that follows the salt header.
/// Two bytes (little-endian `u16`) — the previous one-byte cap of 255
/// entries was the immediate cause of the `.com`-scale bucketing failures.
pub const COUNT_BYTES: usize = 2;

/// Byte offset of the entry-count field within a bucket.
pub const COUNT_OFFSET: usize = BUCKET_SALT_BYTES;

/// Byte offset of the first entry within a bucket. Equal to
/// `BUCKET_SALT_BYTES + COUNT_BYTES` — the salt header plus the count
/// field.
pub const FIRST_ENTRY_OFFSET: usize = BUCKET_SALT_BYTES + COUNT_BYTES;

/// A single DNS record carried inside a bucket. Either an A / AAAA for the
/// queried name itself, or an NS glue IP (the in-zone A/AAAA of the
/// nameserver, looked up at zone-load time — see `dns::zone::Zone::glue_for`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Record {
    /// IPv4 address record for the queried name.
    A(Ipv4Addr),
    /// IPv6 address record for the queried name.
    Aaaa(Ipv6Addr),
    /// Delegated nameserver, represented by the glue IP that should be
    /// queried next. The `IpAddr` variant tells the bucket layer whether
    /// to encode the record as `KIND_NS4` or `KIND_NS6`.
    Ns(IpAddr),
}

/// Canonical form of a DNS name: lowercase, trailing dot stripped, ASCII.
pub fn canonical(name: &str) -> String {
    name.trim_end_matches('.').to_ascii_lowercase()
}

/// Deterministic 64-bit mix used by both the slot hash and the discriminator.
/// Not cryptographic — collision-resistance is enough for the PoC.
fn mix64(input: &[u8], salt: u64) -> u64 {
    let mut h = salt ^ 0x9E3779B97F4A7C15u64;
    for &b in input {
        h = h.wrapping_add(b as u64);
        h = h.wrapping_mul(0x100000001b3); // FNV prime
        h ^= h >> 27;
    }
    // Final avalanche (Murmur-style fmix).
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h
}

/// The 14-bit fingerprint stored alongside each entry. The client matches
/// its own entries by comparing this value, so the bucket may safely contain
/// records belonging to other (colliding) names — provided the discriminators
/// happen to be distinct, which the server enforces at build time by
/// searching `bucket_salt` for a value that produces a collision-free set
/// within each bucket.
///
/// `global_salt` is the per-build randomization carried in the `INFO`
/// response — the same value that drives [`slot`]. `bucket_salt` is the
/// per-bucket value the builder picked and stored in the bucket header;
/// readers extract it via [`read_bucket_salt`].
pub fn discriminator(canon_name: &str, global_salt: u64, bucket_salt: u32) -> u16 {
    let mixed = global_salt
        ^ 0xD15CD15CD15CD15Cu64
        ^ (bucket_salt as u64).wrapping_mul(0x9E3779B97F4A7C15u64);
    (mix64(canon_name.as_bytes(), mixed) as u16) & DISC_MASK
}

/// Maps a canonical name to `(secondary_idx, primary_idx)` for a PIR
/// database with `num_entries` total slots (as advertised in `INFO`).
/// Must be the same function on both client and server — that's why it
/// lives here. `salt` is independent of (but derived from the same value
/// as) the one used for `discriminator`.
#[instrument(skip_all)]
pub fn slot(canon_name: &str, num_entries: usize, salt: u64) -> (usize, usize) {
    assert!(num_entries > 0);
    let slots_per_shard = entries_per_shard(num_entries);
    let s = (mix64(canon_name.as_bytes(), salt ^ 0x510751075107510u64) as usize) % num_entries;
    let secondary_idx = s / slots_per_shard;
    let primary_idx = s % slots_per_shard;
    (secondary_idx, primary_idx)
}

/// Packs a 14-bit discriminator and a 2-bit kind into a 2-byte little-endian
/// tag suitable for the in-bucket entry header.
fn pack_tag(disc: u16, kind: u8) -> [u8; 2] {
    debug_assert!(disc <= DISC_MASK);
    debug_assert!((kind as u16) < (1u16 << (16 - DISC_BITS)));
    let v: u16 = (disc & DISC_MASK) | ((kind as u16) << DISC_BITS);
    v.to_le_bytes()
}

/// Inverse of [`pack_tag`]: extracts `(disc, kind)` from a 2-byte tag.
fn parse_tag(bytes: [u8; 2]) -> (u16, u8) {
    let v = u16::from_le_bytes(bytes);
    let kind = (v >> DISC_BITS) as u8;
    let disc = v & DISC_MASK;
    (disc, kind)
}

/// Number of rdata bytes that follow the 2-byte tag for a given kind.
/// `None` if `kind` is not one of the four known constants — used by the
/// decoder to bail out on a corrupt entry without panicking.
fn rdata_bytes(kind: u8) -> Option<usize> {
    match kind {
        KIND_A | KIND_NS4 => Some(4),
        KIND_AAAA | KIND_NS6 => Some(16),
        _ => None,
    }
}

/// Wire size of one bucket entry (tag + rdata).
pub fn entry_size(kind: u8) -> usize {
    2 + rdata_bytes(kind).expect("unknown kind")
}

/// Reads the per-bucket salt header from a bucket. The salt is what the
/// builder picked to drive [`discriminator`] for every name in the bucket;
/// the client reads it from the PIR reply before computing the
/// discriminator for its query name.
pub fn read_bucket_salt(bucket: &[u8; BUCKET_BYTES]) -> u32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&bucket[0..BUCKET_SALT_BYTES]);
    u32::from_le_bytes(b)
}

/// Writes the per-bucket salt header into a bucket buffer. The builder
/// calls this once per bucket after it has settled on a salt that produces
/// a collision-free discriminator set for the names assigned to that bucket.
pub fn write_bucket_salt(bucket: &mut [u8; BUCKET_BYTES], bucket_salt: u32) {
    bucket[0..BUCKET_SALT_BYTES].copy_from_slice(&bucket_salt.to_le_bytes());
}

/// Reads the 2-byte entry-count field that follows the salt header.
fn read_entry_count(bucket: &[u8; BUCKET_BYTES]) -> u16 {
    let mut b = [0u8; 2];
    b.copy_from_slice(&bucket[COUNT_OFFSET..COUNT_OFFSET + COUNT_BYTES]);
    u16::from_le_bytes(b)
}

/// Writes the 2-byte entry-count field that follows the salt header.
fn write_entry_count(bucket: &mut [u8; BUCKET_BYTES], count: u16) {
    bucket[COUNT_OFFSET..COUNT_OFFSET + COUNT_BYTES]
        .copy_from_slice(&count.to_le_bytes());
}

/// Packs a slice of `N` 10-bit plaintext coefficients into the byte
/// representation of one bucket — four coefficients per five output bytes.
/// This is the "DNS bytes -> PIR coefficients" direction is the inverse;
/// see [`pack_bytes_into_coeffs`] for that one.
///
/// `coeffs` must be exactly `N` long; values must be in `0..PLAIN_MODULUS`.
/// (The packing only honours the low [`COEFF_BITS`] of each value, so the
/// single unused state `1024` would simply round-trip to `0` — we never
/// write it.)
pub fn unpack_coeffs_into_bytes(coeffs: &[u16; N]) -> [u8; BUCKET_BYTES] {
    debug_assert!(N % COEFFS_PER_GROUP == 0);
    let mut out = [0u8; BUCKET_BYTES];
    for g in 0..(N / COEFFS_PER_GROUP) {
        let c0 = (coeffs[g * 4 + 0] as u64) & ((1u64 << COEFF_BITS) - 1);
        let c1 = (coeffs[g * 4 + 1] as u64) & ((1u64 << COEFF_BITS) - 1);
        let c2 = (coeffs[g * 4 + 2] as u64) & ((1u64 << COEFF_BITS) - 1);
        let c3 = (coeffs[g * 4 + 3] as u64) & ((1u64 << COEFF_BITS) - 1);
        let v = c0 | (c1 << 10) | (c2 << 20) | (c3 << 30);
        let bytes = v.to_le_bytes();
        out[g * 5..g * 5 + 5].copy_from_slice(&bytes[0..5]);
    }
    out
}

/// Packs one bucket's bytes into the `N` plaintext coefficients the PIR
/// engine wants to see — five input bytes per four output coefficients.
/// Symmetric inverse of [`unpack_coeffs_into_bytes`].
pub fn pack_bytes_into_coeffs(bucket: &[u8; BUCKET_BYTES]) -> [u16; N] {
    debug_assert!(N % COEFFS_PER_GROUP == 0);
    let mask = (1u64 << COEFF_BITS) - 1;
    let mut out = [0u16; N];
    for g in 0..(N / COEFFS_PER_GROUP) {
        let mut buf = [0u8; 8];
        buf[0..5].copy_from_slice(&bucket[g * 5..g * 5 + 5]);
        let v = u64::from_le_bytes(buf);
        out[g * 4 + 0] = ((v >> 0) & mask) as u16;
        out[g * 4 + 1] = ((v >> 10) & mask) as u16;
        out[g * 4 + 2] = ((v >> 20) & mask) as u16;
        out[g * 4 + 3] = ((v >> 30) & mask) as u16;
    }
    out
}

/// Reasons why [`encode_bucket`] or [`append_entry`] may refuse a write.
/// Both indicate the caller needs to spread its records across more shards
/// (or use a different salt) rather than something being malformed.
#[derive(Debug)]
pub enum BucketError {
    /// More entries than the count field can address.
    TooManyEntries { got: usize, max: usize },
    /// Cumulative entry bytes exceed the bucket capacity.
    Overflow { needed: usize, available: usize },
}

impl std::fmt::Display for BucketError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BucketError::TooManyEntries { got, max } => {
                write!(f, "too many entries: {} > {}", got, max)
            }
            BucketError::Overflow { needed, available } => {
                write!(f, "bucket overflow: needed {} bytes, have {}", needed, available)
            }
        }
    }
}

impl std::error::Error for BucketError {}

/// Writes one entry into a bucket buffer in-place and advances `pos`.
///
/// Used by the server to build the bucket layout directly while iterating
/// over the zone's records, without first materializing an intermediate
/// `Vec<(kind, disc, Record)>` per slot. On the `.com` zone that intermediate
/// would be a couple of gigabytes; this helper lets the caller skip it
/// entirely. The bucket's count byte at [`COUNT_OFFSET`] is updated as part
/// of the write, so the caller can keep using the buffer with
/// [`decode_bucket_matching`] once it's done.
///
/// The caller is responsible for writing the per-bucket salt header (via
/// [`write_bucket_salt`]) before — or after — calling this function;
/// [`append_entry`] only touches the count byte and the entry region.
///
/// `bucket` is assumed to start zero-filled; `pos` should be
/// [`FIRST_ENTRY_OFFSET`] for an empty bucket.
pub fn append_entry(
    bucket: &mut [u8; BUCKET_BYTES],
    pos: &mut u16,
    kind: u8,
    disc: u16,
    rec: &Record,
) -> Result<(), BucketError> {
    let count = read_entry_count(bucket);
    if (count as usize) >= MAX_ENTRIES_PER_BUCKET {
        return Err(BucketError::TooManyEntries {
            got: count as usize + 1,
            max: MAX_ENTRIES_PER_BUCKET,
        });
    }
    let sz = entry_size(kind);
    let start = *pos as usize;
    let end = start + sz;
    if end > BUCKET_BYTES {
        return Err(BucketError::Overflow {
            needed: end,
            available: BUCKET_BYTES,
        });
    }
    bucket[start..start + 2].copy_from_slice(&pack_tag(disc, kind));
    match rec {
        Record::A(ip) => {
            debug_assert_eq!(kind, KIND_A);
            bucket[start + 2..end].copy_from_slice(&ip.octets());
        }
        Record::Aaaa(ip) => {
            debug_assert_eq!(kind, KIND_AAAA);
            bucket[start + 2..end].copy_from_slice(&ip.octets());
        }
        Record::Ns(IpAddr::V4(ip)) => {
            debug_assert_eq!(kind, KIND_NS4);
            bucket[start + 2..end].copy_from_slice(&ip.octets());
        }
        Record::Ns(IpAddr::V6(ip)) => {
            debug_assert_eq!(kind, KIND_NS6);
            bucket[start + 2..end].copy_from_slice(&ip.octets());
        }
    }
    write_entry_count(bucket, count + 1);
    *pos = end as u16;
    Ok(())
}

/// Serializes the entries into one bucket with the supplied per-bucket
/// salt header. Fails (instead of silently truncating) if the entries don't
/// fit — caller is expected to either raise the shard count or pick a
/// different salt and retry.
///
/// Implemented on top of [`append_entry`] so there is one source of truth
/// for the layout of an entry within the bucket.
#[instrument(skip_all)]
pub fn encode_bucket(
    bucket_salt: u32,
    entries: &[(u8, u16, Record)],
) -> Result<[u8; BUCKET_BYTES], BucketError> {
    let mut buf = [0u8; BUCKET_BYTES];
    write_bucket_salt(&mut buf, bucket_salt);
    let mut pos: u16 = FIRST_ENTRY_OFFSET as u16;
    for (kind, disc, rec) in entries {
        append_entry(&mut buf, &mut pos, *kind, *disc, rec)?;
    }
    Ok(buf)
}

/// Parses a bucket and returns all entries whose discriminator matches
/// `want_disc`. The caller is expected to have computed `want_disc` using
/// the per-bucket salt extracted by [`read_bucket_salt`]. Corrupt entries
/// (unknown kinds, truncated rdata) abort further iteration.
#[instrument(skip_all)]
pub fn decode_bucket_matching(bucket: &[u8; BUCKET_BYTES], want_disc: u16) -> Vec<Record> {
    let count = read_entry_count(bucket) as usize;
    let mut out = Vec::new();
    let mut pos = FIRST_ENTRY_OFFSET;
    for _ in 0..count {
        if pos + 2 > BUCKET_BYTES {
            break;
        }
        let mut t = [0u8; 2];
        t.copy_from_slice(&bucket[pos..pos + 2]);
        let (disc, kind) = parse_tag(t);
        let body = match rdata_bytes(kind) {
            Some(b) => b,
            None => break,
        };
        pos += 2;
        if pos + body > BUCKET_BYTES {
            break;
        }
        if disc == want_disc {
            let rdata = &bucket[pos..pos + body];
            let rec = match kind {
                KIND_A => Record::A(Ipv4Addr::new(rdata[0], rdata[1], rdata[2], rdata[3])),
                KIND_AAAA => {
                    let mut a = [0u8; 16];
                    a.copy_from_slice(rdata);
                    Record::Aaaa(Ipv6Addr::from(a))
                }
                KIND_NS4 => Record::Ns(IpAddr::V4(Ipv4Addr::new(
                    rdata[0], rdata[1], rdata[2], rdata[3],
                ))),
                KIND_NS6 => {
                    let mut a = [0u8; 16];
                    a.copy_from_slice(rdata);
                    Record::Ns(IpAddr::V6(Ipv6Addr::from(a)))
                }
                _ => unreachable!(),
            };
            out.push(rec);
        }
        pos += body;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_one_entry() {
        let bucket_salt = 0u32;
        let disc = discriminator("example.org", 0, bucket_salt);
        let entries = vec![(KIND_A, disc, Record::A(Ipv4Addr::new(1, 2, 3, 4)))];
        let buf = encode_bucket(bucket_salt, &entries).unwrap();
        assert_eq!(read_bucket_salt(&buf), bucket_salt);
        let recs = decode_bucket_matching(&buf, disc);
        assert_eq!(recs, vec![Record::A(Ipv4Addr::new(1, 2, 3, 4))]);
    }

    #[test]
    fn filtering_by_discriminator() {
        let bucket_salt = 7u32;
        let d1 = discriminator("a.org", 0, bucket_salt);
        let d2 = discriminator("b.org", 0, bucket_salt);
        let entries = vec![
            (KIND_A, d1, Record::A(Ipv4Addr::new(1, 1, 1, 1))),
            (KIND_A, d2, Record::A(Ipv4Addr::new(2, 2, 2, 2))),
            (KIND_AAAA, d1, Record::Aaaa(Ipv6Addr::LOCALHOST)),
        ];
        let buf = encode_bucket(bucket_salt, &entries).unwrap();
        assert_eq!(read_bucket_salt(&buf), bucket_salt);
        let recs = decode_bucket_matching(&buf, d1);
        assert_eq!(recs.len(), 2);
        let recs = decode_bucket_matching(&buf, d2);
        assert_eq!(recs, vec![Record::A(Ipv4Addr::new(2, 2, 2, 2))]);
    }

    #[test]
    fn salt_changes_hash_output() {
        // Same name, two global salts: at least one of slot/discriminator
        // must disagree, otherwise rehashing buys us nothing.
        let name = "example.org";
        let (slot0, slot1) = (slot(name, 16384, 0), slot(name, 16384, 1));
        let (disc0, disc1) = (
            discriminator(name, 0, 0),
            discriminator(name, 1, 0),
        );
        assert!(slot0 != slot1 || disc0 != disc1,
                "expected either slot or discriminator to differ between salt=0 and salt=1");
        // Identical salts must reproduce identical outputs.
        assert_eq!(slot(name, 16384, 42), slot(name, 16384, 42));
        assert_eq!(
            discriminator(name, 42, 5),
            discriminator(name, 42, 5),
        );
    }

    #[test]
    fn bucket_salt_changes_discriminator() {
        // Two distinct bucket salts on the same name must (almost) always
        // produce different discriminators — that's the property that lets
        // the builder break a per-bucket collision by rehashing only that
        // bucket. We only assert that *some* change happens across a small
        // sweep, which is overwhelmingly likely with a good hash.
        let name = "example.org";
        let base = discriminator(name, 0, 0);
        let any_different = (1u32..32).any(|s| discriminator(name, 0, s) != base);
        assert!(any_different, "discriminator did not vary with bucket salt");
    }

    #[test]
    fn canonical_form() {
        assert_eq!(canonical("ExAmPlE.OrG."), "example.org");
        assert_eq!(canonical("Foo.Bar."), "foo.bar");
    }

    #[test]
    fn entry_sizes_are_packed() {
        assert_eq!(entry_size(KIND_A), 6);
        assert_eq!(entry_size(KIND_NS4), 6);
        assert_eq!(entry_size(KIND_AAAA), 18);
        assert_eq!(entry_size(KIND_NS6), 18);
    }

    #[test]
    fn tag_roundtrip() {
        for disc in [0u16, 1, DISC_MASK, 0x1234 & DISC_MASK] {
            for kind in [KIND_A, KIND_AAAA, KIND_NS4, KIND_NS6] {
                let (d, k) = parse_tag(pack_tag(disc, kind));
                assert_eq!(d, disc);
                assert_eq!(k, kind);
            }
        }
    }

    #[test]
    fn bucket_salt_roundtrip() {
        let mut buf = [0u8; BUCKET_BYTES];
        for s in [0u32, 1, 0xDEADBEEF, u32::MAX] {
            write_bucket_salt(&mut buf, s);
            assert_eq!(read_bucket_salt(&buf), s);
        }
    }

    #[test]
    fn byte_overflow_fails_on_many_a_entries() {
        // A entries are 6 B each; 430 * 6 + 6 header = 2586 B > 2560.
        let entries: Vec<_> = (0..430u16)
            .map(|i| (KIND_A, i & DISC_MASK, Record::A(Ipv4Addr::new(0, 0, (i >> 8) as u8, (i & 0xFF) as u8))))
            .collect();
        match encode_bucket(0, &entries) {
            Err(BucketError::Overflow { .. }) => {}
            other => panic!("expected Overflow, got {:?}", other),
        }
    }

    #[test]
    fn byte_overflow_fails_on_aaaa_entries() {
        // AAAA entries are 18 B each; 150 * 18 + 6 header = 2706 B > 2560.
        let entries: Vec<_> = (0..150u16)
            .map(|i| {
                let mut a = [0u8; 16];
                a[0..2].copy_from_slice(&i.to_le_bytes());
                (KIND_AAAA, i & DISC_MASK, Record::Aaaa(Ipv6Addr::from(a)))
            })
            .collect();
        match encode_bucket(0, &entries) {
            Err(BucketError::Overflow { .. }) => {}
            other => panic!("expected Overflow, got {:?}", other),
        }
    }

    #[test]
    fn count_field_holds_more_than_255() {
        // The old layout was capped at 255 entries by the 1-byte count
        // field — that was the immediate cause of the .com bucketing
        // failure the user reported. Packing 300 small-ish entries must
        // now succeed and round-trip without truncation.
        let mut entries = Vec::new();
        // Each entry: 6 B. 300 * 6 + 6 header = 1806 B, fits in 2560.
        for i in 0..300u16 {
            let disc = i & DISC_MASK;
            entries.push((
                KIND_A,
                disc,
                Record::A(Ipv4Addr::new(10, 1, (i >> 8) as u8, (i & 0xFF) as u8)),
            ));
        }
        let buf = encode_bucket(0xC0FFEE, &entries).expect("must fit");
        assert_eq!(read_entry_count(&buf), 300);
        // Spot-check that every disc round-trips.
        for i in 0..300u16 {
            let want = i & DISC_MASK;
            let recs = decode_bucket_matching(&buf, want);
            assert_eq!(recs.len(), 1, "missing entry for disc 0x{:04x}", want);
        }
    }

    #[test]
    fn coeff_byte_roundtrip() {
        // Generate a deterministic byte pattern, pack to coeffs, unpack
        // back, and verify lossless round-trip — the boundary between
        // PIR coefficients and DNS bytes must be exact.
        let mut bucket = [0u8; BUCKET_BYTES];
        for i in 0..BUCKET_BYTES {
            bucket[i] = ((i * 31 + 7) & 0xFF) as u8;
        }
        let coeffs = pack_bytes_into_coeffs(&bucket);
        // Every coefficient must fit in COEFF_BITS bits.
        for &c in coeffs.iter() {
            assert!(
                (c as u64) < (1u64 << COEFF_BITS),
                "coeff 0x{:x} overflows COEFF_BITS",
                c
            );
        }
        let back = unpack_coeffs_into_bytes(&coeffs);
        assert_eq!(back, bucket, "byte -> coeff -> byte must round-trip");
    }

    #[test]
    fn bucket_size_matches_packing() {
        // BUCKET_BYTES must equal exactly what 4-coeffs-per-5-bytes
        // packing produces for N coefficients — anything else means a
        // partial group at the tail, which the loop-free packer doesn't
        // handle.
        assert_eq!(BUCKET_BYTES, N / COEFFS_PER_GROUP * COEFFS_PER_GROUP_BYTES);
        assert_eq!(N % COEFFS_PER_GROUP, 0);
    }
}
