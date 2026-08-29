//! Two-level PIR over a fleet of [`crate::base_pir::PIRDatabase`] shards.
//!
//! The client's primary index selects one slot inside every shard, and the
//! secondary index selects which shard's answer it actually wants. The
//! first level runs the per-shard inner product; its (encrypted) results
//! are then re-encoded as plaintexts of a second, per-query PIR database,
//! over which the secondary index is evaluated. Only the final ciphertext
//! goes back to the client, so the reply size is independent of the fleet
//! size.
//!
//! [`enc_double_pir`] builds the query, [`perform_batched_double_pir`] is
//! the server-side evaluation (batched over `BATCH_COUNT` independent
//! queries), and [`dec_double_pir`] recovers the plaintext. Callers
//! normally reach these through [`crate::pir_wrapper`] rather than
//! directly.

use feanor_math::integer::IntegerRingStore;
use feanor_math::primitive_int::StaticRing;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::rings::zn::zn_64::Zn;
use feanor_math::rings::zn::{ZnRing, ZnRingStore};
use feanor_math::seq::{VectorFn, VectorView};
use feanor_math::ring::*;
use std::array::from_fn;
use std::cell::RefCell;
use std::time::Instant;
use rayon::prelude::*;
use rand::*;
use tracing::instrument;

use crate::align::AligningAlloc;
use crate::bfv::*;
use crate::base_pir::*;
use crate::simd_zn::CompressedZnx8El;

pub const CTXT_PARTS: usize = 3;
pub const BATCH_COUNT: usize = 2;
pub const PRIMARY_PLAIN_MODULUS: i64 = 1025;
pub const SECONDARY_PLAIN_MODULUS: i64 = 1025;
pub const PRIMARY_A_LOG2_RESPONSE_Q: usize = 17;
pub const PRIMARY_B_LOG2_RESPONSE_Q: usize = 12;

thread_local! {
    /// Reusable backing buffer for the per-query secondary [`PIRDatabase`]
    /// built inside [`perform_batched_double_pir`]. Allocating its
    /// multi-hundred-MB backing store freshly on every query goes through
    /// `mmap`/`munmap`, so each query would re-pay the page-fault and
    /// kernel-zeroing round-trip — whose latency additionally varies wildly
    /// with transparent-huge-page compaction. Keeping the buffer per-thread
    /// (server workers handle one connection each) avoids that without any
    /// synchronization. It is only ever grown, and `set_db` overwrites
    /// every slot of the region in use, so stale content from a previous
    /// query is harmless.
    static SECONDARY_DB_MEMORY: RefCell<Vec<CompressedZnx8El<26>, AligningAlloc>> =
        RefCell::new(Vec::new_in(AligningAlloc::default()));
}

#[derive(Clone, Copy)]
pub struct DoubleQueryRef<'a> {
    pub primary_qry_b: &'a El<CipherRing>,
    /// `None` when the primary databases were built on an [`IndexGroup`]
    /// that does not require the conjugated query (small databases — see
    /// `pir_wrapper::get_database_shape`).
    pub primary_qry_conj_b: Option<&'a El<CipherRing>>,
    pub secondary_qry_b: &'a El<CipherRing>,
    pub secondary_qry_conj_b: Option<&'a El<CipherRing>>,
    pub gk_b: [&'a El<CipherRing>; GK_DIGITS]
}

fn encode_ciphertext_plaintext(a_coeff: u64, b_coeff: u64, Zt: &Zn) -> [El<Zn>; CTXT_PARTS] {
    let integer = (a_coeff * (1 << PRIMARY_B_LOG2_RESPONSE_Q)) + b_coeff;
    debug_assert!(integer < StaticRing::<i64>::RING.pow(*Zt.modulus(), CTXT_PARTS) as u64);
    let mut current = integer;
    let result = std::array::from_fn(|_| {
        let result = current % SECONDARY_PLAIN_MODULUS as u64;
        current = current / SECONDARY_PLAIN_MODULUS as u64;
        return Zt.get_ring().from_int_promise_reduced(result as i64);
    });
    assert_eq!(0, current);
    return result;
}

fn decode_ciphertext_plaintext(data: [El<Zn>; CTXT_PARTS], Zt: &Zn) -> (u64, u64) {
    let mut integer = 0;
    for x in data.into_iter().rev() {
        integer *= SECONDARY_PLAIN_MODULUS as u64;
        integer += Zt.smallest_positive_lift(x) as u64;
    }
    assert!(integer / (1 << PRIMARY_B_LOG2_RESPONSE_Q) < (1 << PRIMARY_A_LOG2_RESPONSE_Q));
    let b = integer % (1 << PRIMARY_B_LOG2_RESPONSE_Q);
    let a = integer / (1 << PRIMARY_B_LOG2_RESPONSE_Q);
    return (a, b);
}

///
/// Assume we want to query the `primary_idx`-th entry of the `secondary_idx`-th database. Then
///  - `primary_qry` should encrypt `(X^(i*k), X^(-i*k))` where `i = floor(primary_idx / SIMD_COUNT)`
///    and `k` is the `query_base_power()` of the primary databases' [`IndexGroup`]; the conjugated
///    part is only required (and only accepted) when that index group `requires_conj()`
///  - `secondary_qry` should encrypt `(X^j, X^-j)` where `j = secondary_idx * SIMD_COUNT + primary_idx % SIMD_COUNT`
///
#[instrument(skip_all)]
#[allow(unused)]
pub fn perform_double_pir<'a, 'b, D>(
    databases: D,
    R_secondary: &PlainRing,
    primary_qry_b: (&El<CipherRing>, Option<&El<CipherRing>>),
    secondary_qry_b: (&El<CipherRing>, Option<&El<CipherRing>>),
    gk_b: [&El<CipherRing>; GK_DIGITS],
    index_group_secondary: IndexGroup,
    debug_sk: Option<&El<CipherRing>>
) -> [(El<CipherRing>, El<CipherRing>); CTXT_PARTS]
    where D: Sync + VectorFn<&'a PIRDatabase<'b>>,
        'b: 'a
{
    perform_batched_double_pir([&databases; BATCH_COUNT], R_secondary, [Some(DoubleQueryRef {
        primary_qry_b: primary_qry_b.0,
        primary_qry_conj_b: primary_qry_b.1,
        secondary_qry_b: secondary_qry_b.0,
        secondary_qry_conj_b: secondary_qry_b.1,
        gk_b: gk_b
    }), None], index_group_secondary, debug_sk).into_iter().next().unwrap().unwrap()
}

#[instrument(skip_all)]
pub fn perform_batched_double_pir<'a, 'b, D>(
    databases: [D; BATCH_COUNT],
    R_secondary: &PlainRing,
    queries: [Option<DoubleQueryRef>; BATCH_COUNT],
    index_group_secondary: IndexGroup,
    _debug_sk: Option<&El<CipherRing>>
) -> [Option<[(El<CipherRing>, El<CipherRing>); CTXT_PARTS]>; BATCH_COUNT]
    where D: Sync + VectorFn<&'a PIRDatabase<'b>>,
        'b: 'a
{
    assert_eq!(SECONDARY_PLAIN_MODULUS, *R_secondary.base_ring().modulus());
    let C = databases[0].at(0).ciphertext_ring();
    let N = R_secondary.rank();
    for k in 0..BATCH_COUNT {
        for i in 0..databases[k].len() {
            assert!(*databases[k].at(i).used_seeds() == UsedSeeds::FirstSet);
            assert!(databases[k].at(i).ciphertext_ring().get_ring() == C.get_ring());
        }
    }
    let log2_t_floor = StaticRing::<i64>::RING.abs_log2_floor(R_secondary.base_ring().modulus()).unwrap();
    let ciphertext_parts = (PRIMARY_A_LOG2_RESPONSE_Q + PRIMARY_B_LOG2_RESPONSE_Q - 1) / log2_t_floor + 1;
    assert_eq!(CTXT_PARTS, ciphertext_parts);
    assert!(CTXT_PARTS * BATCH_COUNT <= SIMD_COUNT);
    // each database is actually SIMD_COUNT databases, thus the final 1-out-of-N selection
    // can select at most from `index_group_secondary.group_order()/SIMD_COUNT` databases
    let databases_len: [_; BATCH_COUNT] = from_fn(|i| databases[i].len());
    assert!(databases_len.iter().all(|len| len * SIMD_COUNT <= index_group_secondary.group_order()));
    let Zt_ref = R_secondary.base_ring();

    MEM_LOAD_DATA.store(0, std::sync::atomic::Ordering::Relaxed);
    let start = Instant::now();
    let batched_part_results: Vec<Option<Vec<[El<PlainRing>; CTXT_PARTS]>>> = queries.into_par_iter().zip(databases.into_par_iter()).map(|(query, databases)| query.map(|query| (0..databases.len()).into_par_iter().flat_map_iter(|i| {
        // the client only ships the conjugated primary ciphertext when the
        // (deterministically agreed-upon) primary index group needs it
        assert_eq!(databases.at(i).index_group().requires_conj(), query.primary_qry_conj_b.is_some());
        let mut a = databases.at(i).get_a().into_iter();
        let mut b = databases.at(i).perform_pir(query.primary_qry_b, query.primary_qry_conj_b, query.gk_b).into_iter();

        let a_modswitched: [_; SIMD_COUNT] = std::array::from_fn(|_| mod_switch_encode::<PRIMARY_A_LOG2_RESPONSE_Q>(C, &a.next().unwrap()));
        let b_modswitched: [_; SIMD_COUNT] = std::array::from_fn(|_| mod_switch_encode::<PRIMARY_B_LOG2_RESPONSE_Q>(C, &b.next().unwrap()));
        
        let result = (0..SIMD_COUNT).map(move |k|
            std::array::from_fn::<_, CTXT_PARTS, _>(|j| R_secondary.from_canonical_basis((0..N).map(|i|
                encode_ciphertext_plaintext(
                    *a_modswitched[k].at(i), 
                    *b_modswitched[k].at(i), 
                    Zt_ref
                )[j]
            )))
        ).collect::<Vec<_>>();
        return result.into_iter();
    }).collect::<Vec<_>>())).collect::<Vec<_>>();
    let primary_end = Instant::now();
    let micros = (primary_end - start).as_micros();
    let bytes = MEM_LOAD_DATA.load(std::sync::atomic::Ordering::Relaxed);
    PRIMARY_PHASE_MICROS.store(micros as u64, std::sync::atomic::Ordering::Relaxed);
    println!("RAM bandwidth (MB/s): {}", bytes as f64 / micros as f64);

    assert_eq!(BATCH_COUNT, batched_part_results.len());
    assert!(batched_part_results.iter().filter_map(Option::as_ref).zip(databases_len).all(|(res, len)| len * SIMD_COUNT == res.len()));

    let zero = C.zero();
    let zero_query: QueryRef<'_> = QueryRef {
        gk_b: [&zero; GK_DIGITS],
        qry_b: &zero,
        qry_conj_b: Some(&zero)
    };

    let batched_part_result_ref = &batched_part_results;
    let mut final_result = SECONDARY_DB_MEMORY.with_borrow_mut(|memory| {
        let len = PIRDatabase::required_memory_general(index_group_secondary, N);
        if memory.len() < len {
            memory.resize_with(len, CompressedZnx8El::default);
        }
        let mut db = PIRDatabase::create(R_secondary.clone(), C.clone(), index_group_secondary, UsedSeeds::SecondSet, &mut memory[..len]);
        db.set_db((0..index_group_secondary.group_order()).flat_map(|i| (0..SIMD_COUNT).map(move |j| if j < CTXT_PARTS * BATCH_COUNT && i < SIMD_COUNT * databases_len[j / CTXT_PARTS]  {
            if let Some(part_results) = &batched_part_result_ref[j / CTXT_PARTS] {
                R_secondary.clone_el(&part_results[i][j % CTXT_PARTS])
            } else {
                R_secondary.zero()
            }
        } else {
            R_secondary.zero()
        })));
        let mut a = db.get_a().into_iter();
        let mut b = db.perform_pir_split(from_fn(|i| queries.get(i / CTXT_PARTS).copied().flatten().map(|q| QueryRef {
            gk_b: q.gk_b,
            qry_b: q.secondary_qry_b,
            qry_conj_b: q.secondary_qry_conj_b
        }).unwrap_or(zero_query))).into_iter();
        let ct: [_; SIMD_COUNT] = std::array::from_fn(|_| (a.next().unwrap(), b.next().unwrap()));
        ct
    }).into_iter();

    let end = Instant::now();
    SECONDARY_PHASE_MICROS.store((end - primary_end).as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    println!("PIR time: {} ms", (end - start).as_millis());

    return from_fn(|i| {
        let result = from_fn(|_| final_result.next().unwrap());
        if queries[i].is_some() { Some(result) } else { None }
    });
}

/// Wall-clock time the most recent [`perform_batched_double_pir`] call spent
/// in its primary phase (querying the base-level fleet), in microseconds.
/// Written on every call, in the same spirit as [`MEM_LOAD_DATA`], so that
/// diagnostics like `bench_wrapped_pir` can report a phase breakdown without
/// threading timing data through the return types. Only meaningful when
/// calls do not overlap — concurrent queries (e.g. multiple server worker
/// threads) simply overwrite each other here.
pub static PRIMARY_PHASE_MICROS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Same as [`PRIMARY_PHASE_MICROS`], but for the secondary phase (building
/// the per-query secondary database from the first-level results and
/// querying it).
pub static SECONDARY_PHASE_MICROS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[instrument(skip_all)]
pub fn enc_double_pir<R: Rng + RngCore + CryptoRng>(
    rng: &mut R,
    R_primary: &PlainRing,
    R_secondary: &PlainRing,
    index_group_primary: IndexGroup,
    index_group_secondary: IndexGroup,
    C: &CipherRing,
    sk: &El<CipherRing>,
    primary_idx: usize,
    secondary_idx: usize,
    sigma: f64
) -> (
    (El<CipherRing>, Option<El<CipherRing>>),
    (El<CipherRing>, Option<El<CipherRing>>),
    [El<CipherRing>; GK_DIGITS]
) {
    assert_eq!(SECONDARY_PLAIN_MODULUS, *R_secondary.base_ring().modulus());
    assert_eq!(PRIMARY_PLAIN_MODULUS, *R_primary.base_ring().modulus());
    let N = R_primary.rank();
    assert_eq!(N, index_group_primary.N());
    assert!(primary_idx < index_group_primary.group_order() * SIMD_COUNT);
    assert!(secondary_idx < index_group_secondary.group_order() / SIMD_COUNT);

    let i = primary_idx / SIMD_COUNT;
    let j = secondary_idx * SIMD_COUNT + primary_idx % SIMD_COUNT;
    let mut gk = gen_gk_b(C, &UsedSeeds::FirstSet.get_gk_seeds(), &mut*rng, sk, 5, GK_DIGITS, LOG2_B, sigma).into_iter();
    return (
        (
            enc_sym_b(R_primary, C, UsedSeeds::FirstSet.get_qry_seeds()[0], &mut*rng, sk, &R_primary.pow(R_primary.canonical_gen(), i * index_group_primary.query_base_power()), sigma),
            if index_group_primary.requires_conj() {
                Some(enc_sym_b(R_primary, C, UsedSeeds::FirstSet.get_qry_seeds()[1], &mut*rng, sk, &R_primary.pow(R_primary.canonical_gen(), 2 * N - i * index_group_primary.query_base_power()), sigma))
            } else {
                None
            }
        ), (
            enc_sym_b(R_secondary, C, UsedSeeds::SecondSet.get_qry_seeds()[0], &mut*rng, sk, &R_secondary.pow(R_secondary.canonical_gen(), j * index_group_secondary.query_base_power()), sigma),
            if index_group_secondary.requires_conj() {
                Some(enc_sym_b(R_secondary, C, UsedSeeds::SecondSet.get_qry_seeds()[1], &mut*rng, sk, &R_secondary.pow(R_secondary.canonical_gen(), 2 * N - j * index_group_secondary.query_base_power()), sigma))
            } else {
                None
            }
        ),
        std::array::from_fn(|_| gk.next().unwrap())
    );
}

#[instrument(skip_all)]
pub fn dec_double_pir(
    R_primary: &PlainRing,
    R_secondary: &PlainRing,
    C: &CipherRing,
    sk: &El<CipherRing>,
    reply: &[(El<CipherRing>, El<CipherRing>); CTXT_PARTS]
) -> El<PlainRing> {
    assert_eq!(SECONDARY_PLAIN_MODULUS, *R_secondary.base_ring().modulus());
    assert_eq!(PRIMARY_PLAIN_MODULUS, *R_primary.base_ring().modulus());
    let N = R_primary.rank();

    let ciphertext_parts: [_; CTXT_PARTS] = std::array::from_fn(|i|
        dec(R_secondary, C, sk, &reply[i])
    );
    let ciphertext_parts: [_; CTXT_PARTS] = std::array::from_fn(|i| R_secondary.wrt_canonical_basis(&ciphertext_parts[i]));
    let a = mod_switch_decode::<PRIMARY_A_LOG2_RESPONSE_Q>(C, &(0..N).map(|i| decode_ciphertext_plaintext(
        std::array::from_fn(|k| ciphertext_parts[k].at(i)),
        R_secondary.base_ring()
    ).0).collect::<Vec<_>>());
    let b = mod_switch_decode::<PRIMARY_B_LOG2_RESPONSE_Q>(C, &(0..N).map(|i| decode_ciphertext_plaintext(
        std::array::from_fn(|k| ciphertext_parts[k].at(i)),
        R_secondary.base_ring()
    ).1).collect::<Vec<_>>());

    return dec(R_primary, C, sk, &(a, b));
}

#[cfg(test)]
use feanor_math::rings::finite::FiniteRingStore;
#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use crate::pir_wrapper::SK_HWT;
#[cfg(test)]
use rand::rngs::StdRng;

#[test]
fn test_double_pir() {
    let mut rng = StdRng::from_seed([0; 32]);

    let log2_N = 8;
    let N = 1 << log2_N;
    let sk_hwt = SK_HWT;
    let primary_t = PRIMARY_PLAIN_MODULUS as u64;
    let secondary_t = SECONDARY_PLAIN_MODULUS as u64;
    let q = FIXED_Q as u64;
    let sigma = 3.2;
    let index_group_secondary = IndexGroup::full_group(N);
    let C = CipherRing::new(Zn::new(q), N, [Zn::new(q).neg_one()]);
    let R_primary = PlainRing::new(Zn::new(primary_t), N, [Zn::new(primary_t).neg_one()]);
    let R_secondary = PlainRing::new(Zn::new(secondary_t), N, [Zn::new(secondary_t).neg_one()]);
    
    let db_count = 1;
    let data = (0..(db_count * R_primary.rank() * SIMD_COUNT)).map_fn(|i| {
        let mut seed = [0; 32];
        seed[0] = (i % 0xFF) as u8;
        seed[1] = ((i >> 8) & 0xFF) as u8;
        seed[2] = ((i >> 16) & 0xFF) as u8;
        let mut data_rng = StdRng::from_seed(seed);
        return R_primary.random_element(|| data_rng.next_u64());
    });
    println!("Querying {} x {} x {} = {} elements of Z/{}Z", db_count, N * SIMD_COUNT, N, db_count * N * SIMD_COUNT * N, primary_t);
    let start = Instant::now();
    let mut memory = aligned_test_memory(db_count * PIRDatabase::required_memory_general(IndexGroup::full_group(N), N));
    let dbs = memory.chunks_mut(PIRDatabase::required_memory_general(IndexGroup::full_group(N), N)).enumerate().map(|(i, memory)| {
        let mut db = PIRDatabase::create(R_primary, C, IndexGroup::full_group(N), UsedSeeds::FirstSet, memory);
        db.set_db(((i * R_primary.rank() * SIMD_COUNT)..((i + 1) * R_primary.rank() * SIMD_COUNT)).map(|i| data.at(i)));
        println!("Preprocessed database {}/{}", i + 1, db_count);
        return db
    }).collect::<Vec<_>>();
    let end = Instant::now();
    println!("Preprocessing done in {} s", (end - start).as_secs());

    let sk = gen_sk(&C, &mut rng, sk_hwt);
    
    for (i, j) in [(0, 0), (1, 1), (2, 0), (2, 1)] {
        let query = enc_double_pir(&mut rng, &R_primary, &R_secondary, IndexGroup::full_group(N), index_group_secondary, &C, &sk, i, j, sigma);
        let start = Instant::now();
        let reply = perform_double_pir(
            dbs.as_fn(),
            &R_secondary,
            (&query.0.0, query.0.1.as_ref()),
            (&query.1.0, query.1.1.as_ref()),
            std::array::from_fn(|i| &query.2[i]),
            index_group_secondary,
            None
        );
        let end = Instant::now();
        let result = dec_double_pir(&R_primary, &R_secondary, &C, &sk, &reply);
        println!("Query on {} primary databases done in {} us", db_count, (end - start).as_micros());
        if j < db_count {
            assert_el_eq!(&R_primary, &data.at(i + j * R_primary.rank()), result);
        } else {
            assert_el_eq!(&R_primary, &R_primary.zero(), result);
        }
    }
}

///
/// Same round-trip as [`test_double_pir`], but with the primary databases
/// built on the half-size index subgroup — the small-fleet configuration
/// where the conjugated primary query ciphertext is omitted entirely.
///
#[test]
fn test_double_pir_primary_subgroup() {
    let mut rng = StdRng::from_seed([1; 32]);

    let log2_N = 8;
    let N = 1 << log2_N;
    let sk_hwt = SK_HWT;
    let primary_t = PRIMARY_PLAIN_MODULUS as u64;
    let secondary_t = SECONDARY_PLAIN_MODULUS as u64;
    let q = FIXED_Q as u64;
    let sigma = 3.2;
    let db_count = 2;
    let index_group_primary = IndexGroup::subgroup_of_size(N, N / 2);
    let index_group_secondary = IndexGroup::subgroup_of_size(N, SIMD_COUNT * db_count);
    assert!(!index_group_primary.requires_conj());
    assert!(!index_group_secondary.requires_conj());
    let C = CipherRing::new(Zn::new(q), N, [Zn::new(q).neg_one()]);
    let R_primary = PlainRing::new(Zn::new(primary_t), N, [Zn::new(primary_t).neg_one()]);
    let R_secondary = PlainRing::new(Zn::new(secondary_t), N, [Zn::new(secondary_t).neg_one()]);

    let entries_per_db = index_group_primary.group_order() * SIMD_COUNT;
    let data = (0..(db_count * entries_per_db)).map_fn(|i| {
        let mut seed = [0; 32];
        seed[0] = (i % 0xFF) as u8;
        seed[1] = ((i >> 8) & 0xFF) as u8;
        seed[2] = ((i >> 16) & 0xFF) as u8;
        let mut data_rng = StdRng::from_seed(seed);
        return R_primary.random_element(|| data_rng.next_u64());
    });
    println!("Querying {} x {} x {} = {} elements of Z/{}Z", db_count, entries_per_db, N, db_count * entries_per_db * N, primary_t);
    let per_db_memory = PIRDatabase::required_memory_general(index_group_primary, N);
    let mut memory = aligned_test_memory(db_count * per_db_memory);
    let dbs = memory.chunks_mut(per_db_memory).enumerate().map(|(i, memory)| {
        let mut db = PIRDatabase::create(R_primary, C, index_group_primary, UsedSeeds::FirstSet, memory);
        db.set_db(((i * entries_per_db)..((i + 1) * entries_per_db)).map(|i| data.at(i)));
        return db
    }).collect::<Vec<_>>();

    let sk = gen_sk(&C, &mut rng, sk_hwt);

    for (i, j) in [(0, 0), (1, 1), (2, 0), (9, 1), (entries_per_db - 1, 0), (entries_per_db - 1, 1)] {
        let query = enc_double_pir(&mut rng, &R_primary, &R_secondary, index_group_primary, index_group_secondary, &C, &sk, i, j, sigma);
        assert!(query.0.1.is_none(), "half-group primary must not produce a conjugated query");
        assert!(query.1.1.is_none(), "subgroup secondary must not produce a conjugated query");
        let reply = perform_double_pir(
            dbs.as_fn(),
            &R_secondary,
            (&query.0.0, query.0.1.as_ref()),
            (&query.1.0, query.1.1.as_ref()),
            std::array::from_fn(|i| &query.2[i]),
            index_group_secondary,
            None
        );
        let result = dec_double_pir(&R_primary, &R_secondary, &C, &sk, &reply);
        assert_el_eq!(&R_primary, &data.at(i + j * entries_per_db), result);
    }
}
