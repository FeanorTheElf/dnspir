//! High-level entry points to the double-PIR engine: encoding the client's
//! query into wire bytes, processing one (or two) of those queries against
//! a preprocessed PIR fleet on the server, and decoding the reply back into
//! plaintext ring coefficients on the client.
//!
//! Compared to the lower-level [`crate::base_pir`] and
//! [`crate::double_pir`] modules, this wrapper takes care of:
//!
//! * **Modulus switching** — `mod_switch_encode` / `mod_switch_decode` from
//!   [`crate::bfv`] are applied around the ring elements so that only
//!   the necessary bits go on the wire.
//! * **Bit packing** — `condense` / `uncondense` turn the
//!   variable-bit-width coefficient sequences into a tight byte stream.
//! * **`IndexGroup` selection** — the whole database layout is derived
//!   deterministically from the total entry count via
//!   [`get_database_shape`]: small databases (up to
//!   [`HALF_SIZE_PRIMARY_MAX_ENTRIES`] plaintext-ring elements) use
//!   half-size primary databases and a small secondary sub-group, which
//!   removes the conjugated query ciphertexts from the wire; larger ones
//!   progressively fall back to the full Galois group.
//! * **Key derivation** — the client's secret key is regenerated from a
//!   compact `Seed` on every call so the server never sees raw key
//!   material.
//!
//! The protocol parameters below were chosen for ≥ 128 bits of security
//! under the lattice estimator (commit
//! `787c05a0eacc2c74bc834a6bb86262e2e90f54ab`); they should not be tuned
//! without re-running the estimator.

use std::array::from_fn;
use std::time::Instant;
use std::cmp::min;

use feanor_math::assert_el_eq;
use feanor_math::integer::IntegerRingStore;
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::*;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::rings::finite::FiniteRingStore;
use feanor_math::rings::zn::zn_64::Zn;
use feanor_math::seq::{VectorFn, VectorView};
use memmap2::{MmapMut, MmapOptions};
use rand::{rngs::StdRng, CryptoRng, Rng, RngCore, SeedableRng};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSliceMut;
use tracing::instrument;

use crate::double_pir::*;
use crate::base_pir::*;
use crate::bfv::*;
use crate::simd_zn::CompressedZnx8El;

/// Mod-switching target for the `a`-part of the secondary-PIR response
/// ciphertext. Fewer bits on the wire than the working modulus would
/// require, while still leaving enough headroom that decryption succeeds.
pub const SECONDARY_A_LOG2_RESPONSE_Q: usize = 17;
/// Same as [`SECONDARY_A_LOG2_RESPONSE_Q`] but for the `b`-part.
pub const SECONDARY_B_LOG2_RESPONSE_Q: usize = 12;
/// Mod-switching target for the Galois-key ciphertexts the client sends.
pub const GALOIS_KEY_LOG2_MODULUS: usize = 50;
/// Mod-switching target for the primary query ciphertext (encrypts `X^i`
/// with `i` the within-shard position).
pub const PRIMARY_QUERY_LOG2_MODULUS: usize = 34;
/// Mod-switching target for the secondary query ciphertext (encrypts the
/// `IndexGroup`-aware shard selector).
pub const SECONDARY_QUERY_LOG2_MODULUS: usize = 34;
/// Standard deviation of the discrete Gaussian noise added to the
/// ciphertexts the client sends.
pub const SIGMA: f64 = 3.2;
/// Hamming weight of the client's secret key (sparse ternary BFV).
pub const SK_HWT: usize = 128;

/// Convenience alias for a `Box<[u8]>` payload, used as the on-the-wire
/// form of queries and replies.
type Bytes = Box<[u8]>;

/// Packs a sequence of `u64` values, each known to use only the low
/// `BITS` bits, into a tight little-endian byte stream. Used to compress
/// mod-switched ciphertext coefficients before they're written to the
/// wire — for non-multiple-of-8 `BITS` the leftover sub-byte fragments
/// are folded into bit-packed chunks.
#[instrument(skip_all)]
fn condense<const BITS: usize>(data: impl Clone + ExactSizeIterator<Item = u64>) -> Vec<u8> {
    let data_it = data.clone().chain((0..((data.len() / 8) * 8 - data.len())).map(|_| 0));
    let mut result = Vec::with_capacity((data.len() / 8) * BITS);
    for i in 0..(BITS / u8::BITS as usize) {
        result.extend(data_it.clone().map(|x| (x >> (u8::BITS as usize * i)) as u8));
    }
    if (BITS % 8) >= 4 {
        let extract = |x: u64| ((x >> ((BITS / 8) * 8)) & ((1 << 4) - 1)) as u8;
        for data in data_it.clone().array_chunks::<2>() {
            result.push((extract(data[0]) << 4) | extract(data[1]));
        }
    }
    if (BITS % 4) >= 2 {
        let extract = |x: u64| ((x >> ((BITS / 4) * 4)) & ((1 << 2) - 1)) as u8;
        for data in data_it.clone().array_chunks::<4>() {
            result.push((extract(data[0]) << 6) | (extract(data[1]) << 4) | (extract(data[2]) << 2) | extract(data[3]));
        }
    }
    if (BITS % 2) >= 1 {
        let extract = |x: u64| ((x >> ((BITS / 2) * 2)) & 1) as u8;
        for data in data_it.clone().array_chunks::<8>() {
            result.push(
                (extract(data[0]) << 7) | (extract(data[1]) << 6) | (extract(data[2]) << 5) | (extract(data[3]) << 4) |
                (extract(data[4]) << 3) | (extract(data[5]) << 2) | (extract(data[6]) << 1) | extract(data[7])
            );
        }
    }
    debug_assert_eq!((data.len() / 8) * BITS, result.len());
    return result;
}

/// Inverse of [`condense`]: reads a tight bit-packed stream back into
/// `count` `u64` coefficients with `BITS` significant bits each. Panics
/// (`"not enough data"`) if the byte stream runs out before `count`
/// values have been reconstructed.
#[instrument(skip_all)]
fn uncondense<const BITS: usize>(count: usize, mut data: impl Iterator<Item = u8>) -> Vec<u64> {
    let mut result: Vec<u64> = (0..count).map(|_| 0).collect();
    for i in 0..(BITS / u8::BITS as usize) {
        for target in result.iter_mut() {
            let source = data.next().expect("not enough data");
            *target |= (source as u64) << (u8::BITS as usize * i);
        }
    }
    if (BITS % 8) >= 4 {
        for target in result.iter_mut().array_chunks::<2>() {
            let source = data.next().expect("not enough data");
            *target[0] |= ((source >> 4) as u64) << ((BITS / 8) * 8);
            *target[1] |= ((source & 0xF) as u64) << ((BITS / 8) * 8);
        }
    }
    if (BITS % 4) >= 2 {
        for target in result.iter_mut().array_chunks::<4>() {
            let source = data.next().expect("not enough data");
            *target[0] |= ((source >> 6) as u64) << ((BITS / 4) * 4);
            *target[1] |= (((source >> 4) & 0x3) as u64) << ((BITS / 4) * 4);
            *target[2] |= (((source >> 2) & 0x3) as u64) << ((BITS / 4) * 4);
            *target[3] |= ((source & 0x3) as u64) << ((BITS / 4) * 4);
        }
    }
    if (BITS % 2) >= 1 {
        for target in result.iter_mut().array_chunks::<8>() {
            let source = data.next().expect("not enough data");
            *target[0] |= ((source >> 7) as u64) << ((BITS / 2) * 2);
            *target[1] |= (((source >> 6) & 1) as u64) << ((BITS / 2) * 2);
            *target[2] |= (((source >> 5) & 1) as u64) << ((BITS / 2) * 2);
            *target[3] |= (((source >> 4) & 1) as u64) << ((BITS / 2) * 2);
            *target[4] |= (((source >> 3) & 1) as u64) << ((BITS / 2) * 2);
            *target[5] |= (((source >> 2) & 1) as u64) << ((BITS / 2) * 2);
            *target[6] |= (((source >> 1) & 1) as u64) << ((BITS / 2) * 2);
            *target[7] |= ((source & 1) as u64) << ((BITS / 2) * 2);
        }
    }
    return result;
}

/// Picks the smallest [`IndexGroup`] that is large enough to hold one
/// entry per shard. The secondary PIR's preprocessing cost scales with
/// the index-group order, so for small fleets we want a sub-group rather
/// than the full ring group. Callers outside this module go through
/// [`get_database_shape`], which derives `num_shards` from the database's
/// entry count.
fn choose_index_group_secondary(num_shards: usize) -> IndexGroup {
    let N = 1 << LOG2_N;
    assert!(num_shards <= N / SIMD_COUNT);
    let log2_l = StaticRing::<i64>::RING.abs_log2_ceil(&(num_shards as i64)).unwrap_or(0);
    if SIMD_COUNT << log2_l == N {
        IndexGroup::full_group(N)
    } else {
        IndexGroup::subgroup_of_size(N, SIMD_COUNT << log2_l)
    }
}

/// Entry-count threshold up to which the primary databases use the
/// half-size index sub-group (and the client omits the conjugated primary
/// query ciphertext). Half-size shards double the fleet's shard count and
/// with it the per-query secondary database, whose cost grows linearly
/// with the entry count — while the wire saving stays a constant ~8.7 KB
/// — so the trade is only worth it for small databases. At `N²/16` the
/// measured latency penalty (+36% on a 4-core dev box, tens of ms on
/// server-grade hardware) is about what the saved upload is worth on a
/// constrained ~1 Mbit/s client uplink; letting the rule run up to its
/// structural limit of `N²/4` instead was measured at +89% total query
/// latency, with the doubled secondary phase dominating.
pub const HALF_SIZE_PRIMARY_MAX_ENTRIES: usize = (1 << LOG2_N) * (1 << LOG2_N) / 16;

/// Computes the shape `(index_group_primary, index_group_secondary)` of a
/// PIR database holding `num_entries` elements of the plaintext ring
/// (2560 B of payload each with the default parameters). This is **the**
/// deterministic rule both client and server derive the database layout
/// from: the server advertises `num_entries` via `INFO`, and everything
/// else — the shape returned here, the number of base-level databases
/// ([`num_databases`]), and the number of entries each of them stores
/// (`SIMD_COUNT * index_group_primary.group_order()`) — follows from it.
///
/// The rule trades conjugated query ciphertexts on the wire (each
/// ~`N * PRIMARY_QUERY_LOG2_MODULUS / 8` B) against server-side work:
///
/// * `num_entries ≤ HALF_SIZE_PRIMARY_MAX_ENTRIES`: the primary
///   databases use the sub-group of order `N/2` and the secondary group
///   order stays well below `N/2` — *no* conjugated ciphertext is sent
///   at all. See [`HALF_SIZE_PRIMARY_MAX_ENTRIES`] for why the half
///   regime stops there rather than at its structural limit of `N²/4`.
/// * `HALF_SIZE_PRIMARY_MAX_ENTRIES < num_entries ≤ N²/2`: full-size
///   primary shards, but the fleet still fits within `N/(2*SIMD_COUNT)`
///   shards, so the secondary group is a proper sub-group — only the
///   conjugated *primary* ciphertext is sent.
/// * `num_entries > N²/2`: full-size shards number more than
///   `N/(2*SIMD_COUNT)`, so both groups are the full Galois group and
///   both conjugated ciphertexts are sent.
///
/// # Panics
///
/// Panics if `num_entries` is zero or exceeds the `N²` entries a single
/// fleet can address.
pub fn get_database_shape(num_entries: usize) -> (IndexGroup, IndexGroup) {
    let N = 1 << LOG2_N;
    assert!(num_entries >= 1);
    assert!(num_entries <= N * N);
    let index_group_primary = if num_entries <= HALF_SIZE_PRIMARY_MAX_ENTRIES {
        IndexGroup::subgroup_of_size(N, N / 2)
    } else {
        IndexGroup::full_group(N)
    };
    let num_shards = num_entries.div_ceil(SIMD_COUNT * index_group_primary.group_order());
    let index_group_secondary = choose_index_group_secondary(num_shards);
    return (index_group_primary, index_group_secondary);
}

/// Number of base-level [`PIRDatabase`]s a fleet serving `num_entries`
/// plaintext-ring elements consists of. Each of them stores
/// `SIMD_COUNT * get_database_shape(num_entries).0.group_order()` entries;
/// the last one is padded with zero entries when `num_entries` is not a
/// multiple of that.
pub fn num_databases(num_entries: usize) -> usize {
    let (index_group_primary, _) = get_database_shape(num_entries);
    num_entries.div_ceil(SIMD_COUNT * index_group_primary.group_order())
}

/// Client-side: produces the wire bytes of one PIR query plus the seed
/// the client must keep around to decrypt the eventual reply.
///
/// `num_entries` is the total size of the queried database in
/// plaintext-ring elements, as advertised by the server; the database
/// shape (and with it the number of ciphertexts in the query) is derived
/// from it via [`get_database_shape`]. `primary_idx` is the position of
/// the desired entry within its base-level database
/// (`0..SIMD_COUNT * index_group_primary.group_order()`) and
/// `secondary_idx` is the index of that database
/// (`0..num_databases(num_entries)`). The returned `Seed` is the entropy
/// used to regenerate the secret key on `process_reply`; the bytes
/// contain the ciphertexts the server will operate on (Galois keys,
/// primary query, secondary query, plus a conjugate of either query only
/// when the respective `IndexGroup` requires one).
#[instrument(skip_all)]
pub fn prepare_query<R: Rng + RngCore + CryptoRng>(
    rng: &mut R,
    primary_idx: usize,
    secondary_idx: usize,
    num_entries: usize
) -> (Bytes, Seed) {
    let (index_group_primary, index_group_secondary) = get_database_shape(num_entries);
    prepare_query_general(rng, primary_idx, secondary_idx, index_group_primary, index_group_secondary)
}

/// Like [`prepare_query`], but with explicitly chosen [`IndexGroup`]s
/// instead of the ones [`get_database_shape`] derives from the entry
/// count. Module-private on purpose: the deterministic derivation is a
/// protocol invariant, and the only legitimate use of an override is
/// [`bench_wrapped_pir`]'s full-size-shard mode — against databases that
/// were built with the same groups.
#[instrument(skip_all)]
fn prepare_query_general<R: Rng + RngCore + CryptoRng>(
    rng: &mut R,
    primary_idx: usize,
    secondary_idx: usize,
    index_group_primary: IndexGroup,
    index_group_secondary: IndexGroup
) -> (Bytes, Seed) {
    let Zq = Zn::new(FIXED_Q as u64);
    let C = CipherRing::new(Zq, 1 << LOG2_N, [Zq.neg_one()]);
    let Zt_primary = Zn::new(PRIMARY_PLAIN_MODULUS as u64);
    let R_primary = PlainRing::new(Zt_primary, 1 << LOG2_N, [Zt_primary.neg_one()]);
    let Zt_secondary = Zn::new(SECONDARY_PLAIN_MODULUS as u64);
    let R_secondary = PlainRing::new(Zt_secondary, 1 << LOG2_N, [Zt_secondary.neg_one()]);
    let sk_seed = std::array::from_fn(|_| (rng.next_u32() & 0xFFFF) as u8);
    let sk_rng = StdRng::from_seed(sk_seed);
    let sk = gen_sk(&C, sk_rng, SK_HWT);
    let (qry1, qry2, gk) = enc_double_pir(
        rng,
        &R_primary,
        &R_secondary,
        index_group_primary,
        index_group_secondary,
        &C,
        &sk,
        primary_idx,
        secondary_idx,
        SIGMA
    );
    let mut result = Vec::new();
    for gk_part in &gk {
        result.extend(condense::<GALOIS_KEY_LOG2_MODULUS>(mod_switch_encode::<GALOIS_KEY_LOG2_MODULUS>(&C, gk_part).iter().copied()));
    }
    result.extend(condense::<PRIMARY_QUERY_LOG2_MODULUS>(mod_switch_encode::<PRIMARY_QUERY_LOG2_MODULUS>(&C, &qry1.0).iter().copied()));
    if index_group_primary.requires_conj() {
        result.extend(condense::<PRIMARY_QUERY_LOG2_MODULUS>(mod_switch_encode::<PRIMARY_QUERY_LOG2_MODULUS>(&C, qry1.1.as_ref().unwrap()).iter().copied()));
    }
    result.extend(condense::<SECONDARY_QUERY_LOG2_MODULUS>(mod_switch_encode::<SECONDARY_QUERY_LOG2_MODULUS>(&C, &qry2.0).iter().copied()));
    if index_group_secondary.requires_conj() {
        result.extend(condense::<SECONDARY_QUERY_LOG2_MODULUS>(mod_switch_encode::<SECONDARY_QUERY_LOG2_MODULUS>(&C, qry2.1.as_ref().unwrap()).iter().copied()));
    }
    return (result.into_boxed_slice(), sk_seed);
}

/// Server-side: answers a single PIR query against `databases` (which
/// must have length `num_databases(num_entries)` and contain the
/// preprocessed primary fleet). Convenience wrapper around [`process_queries`] for the common
/// single-query case; the second batch slot is filled with `None`.
///
/// `debug_sk`, when `Some`, instruments the engine with the client's
/// secret key so it can sanity-check intermediate decryptions; production
/// code passes `None`.
#[allow(unused)]
pub fn process_query<'a, 'b, D>(
    databases: D,
    query: impl Iterator<Item = u8>,
    debug_sk: Option<&El<CipherRing>>
) -> Bytes
    where D: Sync + VectorFn<&'a PIRDatabase<'b>>,
        'b: 'a
{
    process_queries([&databases; BATCH_COUNT], [Some(query), None], debug_sk).into_iter().next().unwrap().unwrap()
}

/// Server-side: answers up to [`BATCH_COUNT`] PIR queries in one pass over
/// the preprocessed databases. Each entry in `queries` is either
/// `Some(query_bytes)` to be answered or `None` to leave that batch slot
/// empty; the corresponding output position is `None` either way when no
/// query was supplied.
///
/// All slots share the same secondary `IndexGroup` (derived from the
/// fleet length), but each gets its own first-level primary results
/// before they're combined in the second stage.
#[instrument(skip_all)]
pub fn process_queries<'a, 'b, D>(
    databases: [D; BATCH_COUNT],
    queries: [Option<impl Iterator<Item = u8>>; BATCH_COUNT],
    debug_sk: Option<&El<CipherRing>>
) -> [Option<Bytes>; BATCH_COUNT]
    where D: Sync + VectorFn<&'a PIRDatabase<'b>>,
        'b: 'a
{
    let N = 1 << LOG2_N;
    let Zq = Zn::new(FIXED_Q as u64);
    let C = CipherRing::new(Zq, N, [Zq.neg_one()]);
    let Zt_secondary = Zn::new(SECONDARY_PLAIN_MODULUS as u64);
    let R_secondary = PlainRing::new(Zt_secondary, N, [Zt_secondary.neg_one()]);
    assert!(databases[0].len() >= databases[1].len());
    // the wire format of the query (in particular whether it carries the
    // conjugated primary ciphertext) is determined by the shape of the
    // preprocessed fleet, which the caller built via `get_database_shape`
    // from the same entry count the client derived its query from
    let index_group_primary = *databases[0].at(0).index_group();
    let index_group_secondary = choose_index_group_secondary(databases[0].len());

    let unpacked_queries: [Option<([El<CipherRing>; GK_DIGITS], El<CipherRing>, Option<El<CipherRing>>, El<CipherRing>, Option<El<CipherRing>>)>; BATCH_COUNT] = queries.map(|query| query.map(|mut query| (
        from_fn(|_| mod_switch_decode::<GALOIS_KEY_LOG2_MODULUS>(&C, &uncondense::<GALOIS_KEY_LOG2_MODULUS>(N, query.by_ref()))),
        mod_switch_decode::<PRIMARY_QUERY_LOG2_MODULUS>(&C, &uncondense::<PRIMARY_QUERY_LOG2_MODULUS>(N, query.by_ref())),
        if index_group_primary.requires_conj() {
            Some(mod_switch_decode::<PRIMARY_QUERY_LOG2_MODULUS>(&C, &uncondense::<PRIMARY_QUERY_LOG2_MODULUS>(N, query.by_ref())))
        } else {
            None
        },
        mod_switch_decode::<SECONDARY_QUERY_LOG2_MODULUS>(&C, &uncondense::<SECONDARY_QUERY_LOG2_MODULUS>(N, query.by_ref())),
        if index_group_secondary.requires_conj() {
            Some(mod_switch_decode::<SECONDARY_QUERY_LOG2_MODULUS>(&C, &uncondense::<SECONDARY_QUERY_LOG2_MODULUS>(N, query.by_ref())))
        } else {
            None
        }
    )));
    let unpacked_queries = from_fn(|i| unpacked_queries[i].as_ref().map(|query| DoubleQueryRef {
        gk_b: from_fn(|i| &query.0[i]),
        primary_qry_b: &query.1,
        primary_qry_conj_b: query.2.as_ref(),
        secondary_qry_b: &query.3,
        secondary_qry_conj_b: query.4.as_ref()
    }));
    
    let replies = perform_batched_double_pir(
        databases, 
        &R_secondary, 
        unpacked_queries,
        index_group_secondary,
        debug_sk
    );

    return from_fn(|i| replies[i].as_ref().map(|reply| {
        let mut result = Vec::new();
        for (a, b) in reply {
            result.extend(condense::<SECONDARY_A_LOG2_RESPONSE_Q>(mod_switch_encode::<SECONDARY_A_LOG2_RESPONSE_Q>(&C, &a).into_iter()));
            result.extend(condense::<SECONDARY_B_LOG2_RESPONSE_Q>(mod_switch_encode::<SECONDARY_B_LOG2_RESPONSE_Q>(&C, &b).into_iter()));
        }
        return result.into_boxed_slice();
    }));
}

/// Client-side: decodes the server's reply bytes into the plaintext-ring
/// coefficients of the requested entry. `sk_seed` must be the seed
/// returned alongside the query by [`prepare_query`] — the secret key
/// is regenerated from it deterministically.
///
/// Returns the `N` coefficients of the entry's ring element, in
/// canonical-basis order. For the DNS use-case the caller maps those
/// coefficients back into bytes via `Zn::smallest_positive_lift`.
#[instrument(skip_all)]
pub fn process_reply(
    mut reply: impl Iterator<Item = u8>,
    sk_seed: Seed
) -> Vec<El<Zn>> {
    let N = 1 << LOG2_N;
    let Zq = Zn::new(FIXED_Q as u64);
    let C = CipherRing::new(Zq, N, [Zq.neg_one()]);
    let Zt_primary = Zn::new(PRIMARY_PLAIN_MODULUS as u64);
    let R_primary = PlainRing::new(Zt_primary, N, [Zt_primary.neg_one()]);
    let Zt_secondary = Zn::new(SECONDARY_PLAIN_MODULUS as u64);
    let R_secondary = PlainRing::new(Zt_secondary, N, [Zt_secondary.neg_one()]);
    let sk_rng = StdRng::from_seed(sk_seed);
    let sk = gen_sk(&C, sk_rng, SK_HWT);
    
    let reply = std::array::from_fn(|_| (
        mod_switch_decode::<SECONDARY_A_LOG2_RESPONSE_Q>(&C, &uncondense::<SECONDARY_A_LOG2_RESPONSE_Q>(N, reply.by_ref())),
        mod_switch_decode::<SECONDARY_B_LOG2_RESPONSE_Q>(&C, &uncondense::<SECONDARY_B_LOG2_RESPONSE_Q>(N, reply.by_ref()))
    ));
    let result = dec_double_pir(&R_primary, &R_secondary, &C, &sk, &reply);
    return R_primary.wrt_canonical_basis(&result).iter().collect();
}

#[test]
fn test_condense_uncondense() {
    let data = (0..1000).collect::<Vec<_>>();
    assert_eq!(&data, &uncondense::<10>(1000, condense::<10>(data.iter().copied()).into_iter()));
    assert_eq!(&data, &uncondense::<11>(1000, condense::<11>(data.iter().copied()).into_iter()));
    assert_eq!(&data, &uncondense::<12>(1000, condense::<12>(data.iter().copied()).into_iter()));
    assert_eq!(&data, &uncondense::<13>(1000, condense::<13>(data.iter().copied()).into_iter()));
    assert_eq!(&data, &uncondense::<14>(1000, condense::<14>(data.iter().copied()).into_iter()));
    assert_eq!(&data, &uncondense::<15>(1000, condense::<15>(data.iter().copied()).into_iter()));
    assert_eq!(&data, &uncondense::<16>(1000, condense::<16>(data.iter().copied()).into_iter()));
}

/// Allocates a 2 MiB-aligned anonymous mapping. The requested size is
/// rounded up to the next 2 MiB boundary. Tries huge pages first
/// (`MAP_HUGETLB`) for TLB efficiency during the streaming inner-product
/// loops, and silently falls back to regular pages if the kernel can't
/// satisfy the request — typically: no huge pages reserved on this host.
pub fn alloc_huge(size_bytes: usize) -> std::io::Result<MmapMut> {
    const PAGE_2MB: usize = 1 << 21;
    let aligned = size_bytes.div_ceil(PAGE_2MB) * PAGE_2MB;
    MmapOptions::new()
        .len(aligned)
        .huge(Some(21))
        .map_anon()
        .or_else(|_| MmapOptions::new()
            .len(aligned)
            .map_anon()
        )
}

///
/// Benchmarks the protocol by running `16 * try_seeds` queries against a database of
/// `num_entries` plaintext-ring elements, laid out exactly as [`get_database_shape`]
/// prescribes (both shapes are reachable by picking `num_entries` on either side of
/// [`HALF_SIZE_PRIMARY_MAX_ENTRIES`]). Entry counts beyond one fleet's `N²` capacity
/// spill into the second batch slot, answered by the same query.
///
/// If `preprocess_db_count` is given and smaller than the number of base-level DBs,
/// then only that many DBs will actually be created and pre-processed, but the
/// protocol will still query all of them, using the same preprocessed data for
/// multiple base-level DBs. Thus, the online performance of the protocol should be
/// roughly the same as for the fully preprocessed fleet, but the preprocessing cost
/// is much lower.
///
/// `force_full_dbs` overrides the small-database rule and uses full-size primary
/// databases (with the conjugated primary query) even when `num_entries` is at most
/// [`HALF_SIZE_PRIMARY_MAX_ENTRIES`] — the layout the protocol itself can no longer
/// express, kept around so the cost of the half-size trade (doubled shard count,
/// doubled secondary group) can be measured against its wire savings. A no-op for
/// larger `num_entries`.
///
pub fn bench_wrapped_pir(preprocess_db_count: Option<usize>, num_entries: usize, try_seeds: u8, force_full_dbs: bool) {
    let Zt = Zn::new(PRIMARY_PLAIN_MODULUS as u64);
    let N = 1 << LOG2_N;
    let R = PlainRing::new(Zt, N, [Zt.neg_one()]);
    let Zq = Zn::new(FIXED_Q as u64);
    let C = CipherRing::new(Zq, N, [Zq.neg_one()]);

    assert!(num_entries >= 1);
    assert!(num_entries <= BATCH_COUNT * N * N, "at most {} entries are supported", BATCH_COUNT * N * N);
    // the query's shape is that of the first (fully loaded) fleet; the DNS
    // server never exceeds one fleet, but the bench also exercises the
    // second batch slot
    let fleet_entries = min(num_entries, N * N);
    let (index_group_primary, index_group_secondary) = if force_full_dbs {
        let index_group_primary = IndexGroup::full_group(N);
        let fleet_shards = fleet_entries.div_ceil(SIMD_COUNT * N);
        (index_group_primary, choose_index_group_secondary(fleet_shards))
    } else {
        get_database_shape(fleet_entries)
    };
    let entries_per_db = SIMD_COUNT * index_group_primary.group_order();
    let db_count = num_entries.div_ceil(entries_per_db);
    let preprocess_db_count = preprocess_db_count.unwrap_or(db_count);
    assert!(
        preprocess_db_count <= db_count,
        "--preprocessed-db-count ({}) must not exceed the number of base-level databases ({})",
        preprocess_db_count, db_count
    );
    println!(
        "Primary databases: {} x {} entries{}, conjugated primary query {}, conjugated secondary query {}",
        db_count,
        entries_per_db,
        if force_full_dbs { " (full size forced)" } else { "" },
        if index_group_primary.requires_conj() { "sent" } else { "omitted" },
        if index_group_secondary.requires_conj() { "sent" } else { "omitted" }
    );
    let per_db_memory = PIRDatabase::required_memory_general(index_group_primary, N);
    let data = (0..(preprocess_db_count * entries_per_db)).map_fn(|i| {
        let mut seed = [0; 32];
        seed[0] = (i % 0xFF) as u8;
        seed[1] = ((i >> 8) & 0xFF) as u8;
        seed[2] = ((i >> 16) & 0xFF) as u8;
        let mut data_rng = StdRng::from_seed(seed);
        return R.random_element(|| data_rng.next_u64());
    });
    println!("Allocating {}B for preprocessed database", preprocess_db_count * per_db_memory * size_of::<CompressedZnx8El<25>>());
    let mut mmapped_memory = alloc_huge(preprocess_db_count * per_db_memory * size_of::<CompressedZnx8El<26>>()).unwrap();
    let memory = bytemuck::cast_slice_mut(&mut mmapped_memory);

    println!("Preparing database of {} x {} x {} = {} elements of Z/{}Z", db_count, entries_per_db, N, db_count * entries_per_db * N, PRIMARY_PLAIN_MODULUS);
    let start = Instant::now();
    let dbs = memory.par_chunks_mut(per_db_memory).enumerate().map(|(i, memory)| {
        let mut db = PIRDatabase::create(R, C, index_group_primary, UsedSeeds::FirstSet, memory);
        db.set_db(((i * entries_per_db)..((i + 1) * entries_per_db)).map(|i| data.at(i)));
        println!("Preprocessed database {}/{}", i + 1, preprocess_db_count);
        return db
    }).collect::<Vec<_>>();
    let end = Instant::now();
    println!("Preprocessing done in {} s", (end - start).as_secs());
    
    let mut time_sum = 0;
    let mut time_sqr_sum = 0;
    // Phase breakdown, read back from the metric statics the engine
    // publishes (see `PRIMARY_PHASE_MICROS` / `SECONDARY_PHASE_MICROS`).
    let mut primary_time_sum: u128 = 0;
    let mut secondary_time_sum: u128 = 0;
    let mut query_size: Option<usize> = None;
    let mut response_size: Option<usize> = None;
    let mut count = 0;
    let indices = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 2), (1, 2), (2, 0), (2, 1), (2, 2), (2742, 128), (923, 255), (1389, 129), (200, 200), (4000, 255), (3000, 0), (3000, 1)];
    println!("Performing PIR queries");
    for seed in 0..try_seeds {
        let mut rng = StdRng::from_seed(std::array::from_fn(|i| if i == 0 { seed } else { 0 }));

        for (i, j) in &indices {
            let (query, sk) = prepare_query_general(&mut rng, *i, j % db_count, index_group_primary, index_group_secondary);
            if let Some(query_size) = query_size {
                assert_eq!(query_size, query.len());
            } else {
                query_size = Some(query.len());
            }
            let dbs_ref = &dbs;

            let start = Instant::now();
            let reply = process_queries(
                from_fn(|l| (0..min(db_count.saturating_sub(l * N / SIMD_COUNT), N / SIMD_COUNT)).map_fn(move |k| &dbs_ref[(k + l * N / SIMD_COUNT) % preprocess_db_count])),
                from_fn(|l| if l * N / SIMD_COUNT < db_count { Some(query.iter().copied()) } else { None }),
                None
            ).into_iter().map(|rep| rep.unwrap_or(Box::new([]))).collect::<Vec<_>>();
            let end = Instant::now();
            time_sum += (end - start).as_micros();
            time_sqr_sum += (end - start).as_micros() * (end - start).as_micros();
            primary_time_sum += PRIMARY_PHASE_MICROS.load(std::sync::atomic::Ordering::Relaxed) as u128;
            secondary_time_sum += SECONDARY_PHASE_MICROS.load(std::sync::atomic::Ordering::Relaxed) as u128;

            if let Some(response_size) = response_size {
                assert_eq!(response_size, reply.iter().map(|rep| rep.len()).sum::<usize>());
            } else {
                response_size = Some(reply.iter().map(|rep| rep.len()).sum());
            }
            let result = process_reply(reply[0].iter().copied(), sk);
            let expected = data.at(i + ((j % db_count) % preprocess_db_count) * entries_per_db);
            assert_el_eq!(&R, &expected, &R.from_canonical_basis(result));
            count += 1;
            println!("Performed query {}/{}", count, indices.len() * try_seeds as usize);
        }
    }
    println!("done");
    // Break the upload down by wire-format section: the Galois keys are a
    // fixed prefix of GK_DIGITS mod-switched ciphertexts, everything after
    // them is the (primary + secondary, conjugated where the respective
    // index group demands it) query ciphertexts.
    let galois_key_bytes = GK_DIGITS * (N / 8) * GALOIS_KEY_LOG2_MODULUS;
    let queries_run = indices.len() as u128 * try_seeds as u128;
    println!("Communication:");
    println!("  Galois keys:          {} B", galois_key_bytes);
    println!("  Query (excl. keys):   {} B", query_size.unwrap() - galois_key_bytes);
    println!("  Response:             {} B", response_size.unwrap());
    println!("Response time (avg):    {} us", time_sum / queries_run);
    println!("  Primary DBs (avg):    {} us", primary_time_sum / queries_run);
    println!("  Secondary DB (avg):   {} us", secondary_time_sum / queries_run);
    println!("Response time stddev:   {}", (time_sqr_sum as f64 / (indices.len() as f64 * try_seeds as f64) - (time_sum as f64 / (indices.len() as f64 * try_seeds as f64)).powi(2)).sqrt());
}