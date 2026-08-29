//! One preprocessed PIR shard and the query evaluation over it.
//!
//! A [`PIRDatabase`] holds `SIMD_COUNT * N` (or half that) plaintext ring
//! elements in an NTT- and SIMD-friendly layout, together with the
//! precomputed data the response ciphertext is assembled from. Building one
//! is the expensive step — seconds per shard, and hundreds of megabytes of
//! backing store — but it is done once at server startup, after which each
//! query is an inner product over the whole slab.
//!
//! [`IndexGroup`] describes which sub-group of the ring's Galois group the
//! database is laid out over. Choosing the smallest sufficient sub-group
//! lets small databases skip the conjugated query ciphertext entirely; the
//! choice is made by `pir_wrapper::get_database_shape` and must match on
//! both sides.
//!
//! The ring parameters (`LOG2_N`, the RNS factors of [`FIXED_Q`], the
//! Galois-key digit count) and the nothing-up-my-sleeve [`SEEDS`] used to
//! expand ciphertext `a`-parts are fixed here for the whole engine.
//!
//! To retrieve from more entries than a single shard holds, compose several
//! shards with [`crate::double_pir::perform_batched_double_pir`].

use std::array::from_fn;
use std::mem::{swap, transmute};
use std::cmp::min;
use std::mem::Alignment;
use std::sync::atomic::{AtomicU64, Ordering};

use feanor_math::algorithms::fft::cooley_tuckey::CooleyTuckeyFFT;
use feanor_math::algorithms::fft::FFTAlgorithm;
use feanor_math::algorithms::unity_root::get_prim_root_of_unity_pow2;
use feanor_math::delegate::{DelegateRing, UnwrapHom};
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::integer::IntegerRingStore;
use feanor_math::matrix::OwnedMatrix;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::rings::zn::{ZnRing, ZnRingStore};
use feanor_math::seq::{SwappableVectorViewMut, VectorFn, VectorView};
use feanor_math::{primitive_int::StaticRing, ring::*};
use feanor_math::rings::zn::zn_64::Zn;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::join;
use rayon::slice::ParallelSliceMut;
use tracing::instrument;

use crate::align::{AligningAlloc, zero_vec};
use crate::permute::permute;
use crate::avx::*;
use crate::simd_zn::{CompletelyReducedSpecialZnx8, CompressedZnx8El, SpecialZnx8, SpecialZnx8El};
use crate::bfv::*;

///
/// Fixed seeds used to generate the a-part of ciphertexts;
/// To prove that these are not backdoored, they have been chosen
/// as the SHA-3 hashes of the following UTF-8 encoded strings
///  - "Star Wars: Episode I - The Phantom Menace"
///  - "Star Wars: Episode II - Attack of the Clones"
///  - "Star Wars: Episode III – Revenge of the Sith"
///  - "Star Wars: Episode IV - A New Hope"
///  - "Star Wars: Episode V - The Empire Strikes Back"
///  - "Star Wars: Episode VI - Return of the Jedi"
/// 
pub const SEEDS: [Seed; 6] = [
    [0xda, 0xd2, 0x0c, 0x14, 0x2d, 0xeb, 0x8b, 0xb4, 0xd6, 0x63, 0x20, 0xee, 0x07, 0xb1, 0xc1, 0x90, 0xec, 0x25, 0xc0, 0xa1, 0x54, 0x1c, 0xca, 0xf5, 0x98, 0xfa, 0x22, 0xff, 0xdf, 0xdd, 0x68, 0xba],
    [0xd3, 0xc5, 0xf6, 0xee, 0xb3, 0x3a, 0x4c, 0xf3, 0x33, 0x51, 0x6c, 0xe2, 0x73, 0xd5, 0x74, 0x76, 0xb5, 0x79, 0x6c, 0xce, 0x9c, 0xf3, 0x37, 0xa1, 0x9c, 0x84, 0xbc, 0xbe, 0xa4, 0x3d, 0x2c, 0xb5],
    [0x95, 0x27, 0xb8, 0x6c, 0x30, 0xdc, 0x30, 0xcf, 0x1b, 0xb1, 0x74, 0xc3, 0xcd, 0x6c, 0xd7, 0x50, 0x72, 0x09, 0xda, 0xa5, 0x4b, 0xbd, 0xb9, 0x99, 0x04, 0x14, 0x00, 0x08, 0x3d, 0x68, 0x24, 0x8a],
    [0x4c, 0xa2, 0xf6, 0x47, 0x8a, 0x58, 0x11, 0x91, 0x67, 0x67, 0x41, 0x00, 0x2a, 0x00, 0xed, 0xd1, 0xac, 0x97, 0xf9, 0x30, 0x98, 0x0b, 0x7f, 0xbe, 0xf0, 0x33, 0xb4, 0x48, 0x23, 0x4b, 0x21, 0x6b],
    [0x98, 0xfc, 0x62, 0xc5, 0x17, 0x52, 0x0e, 0x5d, 0x29, 0xfc, 0x30, 0x7e, 0x73, 0xef, 0xb1, 0x0e, 0x14, 0x18, 0xaa, 0x07, 0x0e, 0x32, 0x36, 0x4f, 0x89, 0x3f, 0xa8, 0xbd, 0x92, 0x13, 0x03, 0x27],
    [0xb3, 0x61, 0xdc, 0x94, 0x34, 0xdb, 0xba, 0xe9, 0x28, 0xe5, 0xc9, 0x64, 0xdc, 0x18, 0x02, 0x54, 0x02, 0xfe, 0x81, 0xaa, 0xf7, 0x49, 0x17, 0x20, 0x8a, 0x6e, 0x6f, 0xd9, 0x55, 0x70, 0x2a, 0x21]
];
pub const GK_DIGITS: usize = 2;
pub const SIMD_COUNT: usize = 8;
pub const LOG2_B: usize = 26;
pub const Q_FACTOR1: i64 = (1 << 26) - (1 << 12) + 1;
pub const Q_FACTOR2: i64 = (1 << 25) - (1 << 12) + 1;
pub const FIXED_Q: i64 = Q_FACTOR1 * Q_FACTOR2;
pub const LOG2_N: usize = 11;

type NTT<const K: u32> = CooleyTuckeyFFT<SpecialZnx8<K>, CompletelyReducedSpecialZnx8<K>, UnwrapHom<RingValue<CompletelyReducedSpecialZnx8<K>>, RingValue<SpecialZnx8<K>>>>;
type CompressedRNSEl = (Vec<CompressedZnx8El<26>, AligningAlloc>, Vec<CompressedZnx8El<25>, AligningAlloc>);
type RNSEl = (Vec<SpecialZnx8El<26>, AligningAlloc>, Vec<SpecialZnx8El<25>, AligningAlloc>);

#[derive(Clone, Copy)]
pub struct QueryRef<'a> {
    pub qry_b: &'a El<CipherRing>,
    pub qry_conj_b: Option<&'a El<CipherRing>>,
    pub gk_b: [&'a El<CipherRing>; GK_DIGITS]
}

///
/// Selects which set of [`SEEDS`] is used to derive the public `a`-parts of
/// the client's ciphertexts.
///
/// In the double-PIR protocol the *primary* base-level databases (the ones
/// the client's query actually selects an entry from) are queried under
/// [`UsedSeeds::FirstSet`]. The *secondary* database, created internally by
/// [`crate::double_pir::perform_batched_double_pir`] to aggregate the
/// partial replies, is queried under [`UsedSeeds::SecondSet`].
///
/// Application code building a fleet of primary databases should always
/// pass [`UsedSeeds::FirstSet`] to [`PIRDatabase::create`].
///
#[derive(Clone, PartialEq, Eq)]
pub enum UsedSeeds {
    FirstSet, SecondSet
}

///
/// Specifies the region of a [`PIRDatabase`] that is used for PIR
/// lookups.
/// 
/// In many cases, it will be the whole region, which can be described by
/// [`IndexGroup::full_group()`]. However, if only a smaller part is required,
/// you can instead use [`IndexGroup::subgroup_of_size()`].
/// 
/// # Internals
/// 
/// The algorithm behind [`PIRDatabase`] proceeds as follows:
/// Let `H = { 1, 5, 9, 13, ..., 4l - 3 }`. We encode the data as
/// ```text
///   \hat{a}_j = \sum_{i < l} a_i ζ_(4l)^(-ij)
/// ```
/// Then a query is performed as
/// ```text
///   \sum_(j in H) \hat{a}_j σ_j(qry)    where qry = ζ_(4l)^(qry-idx)
/// ```
/// 
/// Alternatively, if the whole space should be used, we set
/// `H = (Z/mZ)*`.
/// 
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct IndexGroup {
    N: usize,
    l: usize
}

impl IndexGroup {

    pub const fn full_group(N: usize) -> Self {
        Self { N, l: N }
    }

    ///
    /// Creates a new [`IndexGroup`] that describes a degree-`size`
    /// subspace of the full space. In other words, a [`PIRDatabase`]
    /// instantiated with this [`IndexGroup`] can store exactly 
    /// `degree * SIMD_COUNT` elements of the plaintext ring.
    /// 
    pub const fn subgroup_of_size(N: usize, size: usize) -> Self {
        assert!(N % size == 0);
        assert!(size * 2 <= N);
        Self { N, l: size }
    }

    pub const fn query_base_power(self) -> usize {
        self.N / 2 / self.loop_len()
    }

    pub const fn N(&self) -> usize {
        self.N
    }

    pub const fn group_order(&self) -> usize {
        self.l
    }

    const fn loop_len(&self) -> usize {
        if self.l < self.N {
            self.l
        } else {
            self.N / 2
        }
    }

    pub const fn requires_conj(&self) -> bool {
        self.l == self.N
    }

    fn gen_H(&self, Zm: &Zn) -> El<Zn> {
        Zm.int_hom().map(5)
    }
}

fn index_to_galois(group_order: usize, Zm: &Zn, idx: usize) -> El<Zn> {
    Zm.coerce(&StaticRing::<i64>::RING, Zm.modulus() / group_order as i64 * idx as i64 + 1)
}

fn galois_to_index(group_order: usize, Zm: &Zn, g: El<Zn>) -> usize {
    StaticRing::<i64>::RING.checked_div(&(Zm.smallest_positive_lift(g) - 1), &(Zm.modulus() / group_order as i64)).unwrap().try_into().unwrap()
}

impl UsedSeeds {

    pub fn get_gk_seeds(&self) -> [Seed; GK_DIGITS] {
        [SEEDS[0], SEEDS[1]]
    }

    pub fn get_qry_seeds(&self) -> [Seed; 2] {
        match self {
            Self::FirstSet => [SEEDS[2], SEEDS[3]],
            Self::SecondSet => [SEEDS[4], SEEDS[5]],
        }
    }
}

///
/// A preprocessed base-level database that can answer encrypted PIR queries
/// under the BFV-based scheme of this crate.
///
/// A single [`PIRDatabase`] holds up to `SIMD_COUNT * N` *entries*, where each
/// entry is an element of the plaintext ring `R = Z_t[X] / (X^N + 1)`. With
/// the default parameters (`N = 2048`, `SIMD_COUNT = 8`, `t = PRIMARY_PLAIN_MODULUS = 1025`)
/// this gives `16 384` entries of roughly `2048 * 10 ≈ 20 kbit` each per database,
/// for ~41 MiB of raw payload per base-level database.
/// 
/// Note that it is possible to work in a subspace of the whole space by instantiating
/// [`PIRDatabase`] with a [`PIRDatabase::create()`], which results in
/// a [`PIRDatabase`] holding `SIMD_COUNT * index_group.group_order()` elements of the
/// plaintext ring `R = Z_t[X] / (X^N + 1)`; We emphasize that the standard parameters
/// and the plaintext ring remain unchanged, so security is not affected. The only difference
/// is that with this configuration, only a subspace of the plaintext space is actually
/// filled with information.
///
/// Once preprocessing is done the database is read-only as far as queries
/// are concerned, so a single instance can be shared concurrently across
/// arbitrarily many query-serving threads via shared references.
///
pub struct PIRDatabase<'a> {
    ring: PlainRing,
    ciphertext_ring: CipherRing,
    index_group: IndexGroup,
    /// Galois automorphisms and NTT applied to the database elements; 
    /// Permuted so that the online access will be sequential.
    /// 
    /// index_group.loop_len() x N
    transformed_entries: (&'a mut [CompressedZnx8El<26>], &'a mut [CompressedZnx8El<25>]),
    /// index_group.loop_len() x N
    transformed_entries_conj: (&'a mut [CompressedZnx8El<26>], &'a mut [CompressedZnx8El<25>]),
    reply_a: RNSEl,
    /// The precomputed gadget decompositions of the `a`-part of the inputs to the homomorphic
    /// Galois automorphism applications. In the online phase, we will multiply the `b`-parts
    /// of the Galois key with those. Store in "transposed" format to enable sequential access
    /// during the online phase.
    /// 
    /// index_group.loop_len() x N
    inner_prod_gk: [(&'a mut [CompressedZnx8El<26>], &'a mut [CompressedZnx8El<25>]); GK_DIGITS],
    /// index_group.loop_len() x N
    inner_prod_conj_gk: [(&'a mut [CompressedZnx8El<26>], &'a mut [CompressedZnx8El<25>]); GK_DIGITS],
    rns_base: (SpecialZnx8<26>, SpecialZnx8<25>),
    len_N_ntts: (NTT<26>, NTT<25>),
    len_subgroup_order_ntts: (NTT<26>, NTT<25>),
    zeta: (SpecialZnx8El<26>, SpecialZnx8El<25>),
    use_seeds: UsedSeeds
}

impl<'a> PIRDatabase<'a> {

    ///
    /// Creates a new [`PIRDatabase`] that can store `SIMD_COUNT * N` entries in `R/tR`.
    /// It is zero-initialized, and can later populated with data using [`PIRDatabase::set_db()`]
    /// 
    /// The parameter `memory` should be a slice of length [`PIRDatabase::required_memory()`] elements
    /// of type [`CompressedZnx8El`], and will be used to store the encoding of the database. It is
    /// recommended to pass an allocation that supports large pages, since it will be read in streaming
    /// fashion during the critical loop when running [`PIRDatabase::perform_pir()`].
    /// 
    #[allow(unused)]
    pub fn new(ring: PlainRing, ciphertext_ring: CipherRing, use_seeds: UsedSeeds, memory: &'a mut [CompressedZnx8El<26>]) -> Self {
        let N = ring.rank();
        assert_eq!(1 << LOG2_N, N, "N has to be 2048; smaller N are insecure, and larger N are not supported");
        Self::create(ring, ciphertext_ring, IndexGroup::full_group(N), use_seeds, memory)
    }

    ///
    /// Like [`PIRDatabase::new()`], but instantiations with `N < 2048` or an `index_group` that is
    /// not the full Galois group. Note that choosing `N < 2048` will be insecure.
    /// 
    #[instrument(skip_all)]
    pub fn create(ring: PlainRing, ciphertext_ring: CipherRing, index_group: IndexGroup, use_seeds: UsedSeeds, mut memory: &'a mut [CompressedZnx8El<26>]) -> Self {
        let N = ring.rank();
        let log2_N = StaticRing::<i64>::RING.abs_log2_ceil(&N.try_into().unwrap()).unwrap();
        assert_eq!(N, 1 << log2_N);
        assert_eq!(Self::required_memory_general(index_group, N), memory.len());
        let rns_base = (
            SpecialZnx8::<26>::new((1 << 12) - 1),
            SpecialZnx8::<25>::new((1 << 12) - 1),
        );
        assert_eq!(Q_FACTOR1, *rns_base.0.modulus());
        assert_eq!(Q_FACTOR2, *rns_base.1.modulus());
        assert_eq!(*rns_base.0.modulus() * *rns_base.1.modulus(), *ciphertext_ring.base_ring().modulus());
        assert_eq!(N, ciphertext_ring.rank());
        assert_eq!(N, index_group.N());

        fn get_CompressedZnx8El26<'a, 'b>(memory: &'b mut &'a mut [CompressedZnx8El<26>], size: usize) -> &'a mut [CompressedZnx8El<26>] {
            let mut result = None;
            take_mut::take(memory, |data| {
                let (res, rest) = data.split_at_mut(size);
                assert!(res.as_ptr().is_aligned_to(Alignment::of::<m512i>().as_usize()));
                assert!(rest.as_ptr().is_aligned_to(Alignment::of::<m512i>().as_usize()));
                result = Some(res);
                return rest;
            });
            return result.unwrap();
        }
        fn get_CompressedZnx8El25<'a, 'b>(memory: &'b mut &'a mut [CompressedZnx8El<26>], size: usize) -> &'a mut [CompressedZnx8El<25>] {
            assert_eq!(align_of::<CompressedZnx8El<25>>(), align_of::<CompressedZnx8El<26>>());
            assert_eq!(size_of::<CompressedZnx8El<25>>(), size_of::<CompressedZnx8El<26>>());
            unsafe { transmute::<&'a mut [CompressedZnx8El<26>], &'a mut [CompressedZnx8El<25>]>(get_CompressedZnx8El26(memory, size)) }
        }

        let transformed_entries = (get_CompressedZnx8El26(&mut memory, N * index_group.loop_len()), get_CompressedZnx8El25(&mut memory, N * index_group.loop_len()));
        let transformed_entries_conj = if index_group.requires_conj() {
            (get_CompressedZnx8El26(&mut memory, N * index_group.loop_len()), get_CompressedZnx8El25(&mut memory, N * index_group.loop_len()))
        } else {
            (&mut [][..], &mut [][..])
        };
        let reply_a = (Vec::new_in(AligningAlloc::default()), Vec::new_in(AligningAlloc::default()));
        let inner_prod_gk = from_fn(|_| (get_CompressedZnx8El26(&mut memory, N * index_group.loop_len()), get_CompressedZnx8El25(&mut memory, N * index_group.loop_len())));
        let inner_prod_conj_gk = if index_group.requires_conj() {
            from_fn(|_| (get_CompressedZnx8El26(&mut memory, N * index_group.loop_len()), get_CompressedZnx8El25(&mut memory, N * index_group.loop_len())))
        } else {
            from_fn(|_| (&mut [][..], &mut [][..]))
        };
        let Fp0 = rns_base.0.component_ring().as_field().ok().unwrap();
        let zeta0 = get_prim_root_of_unity_pow2(&Fp0, log2_N + 1).unwrap();
        let Fp1 = rns_base.1.component_ring().as_field().ok().unwrap();
        let zeta1 = get_prim_root_of_unity_pow2(&Fp1, log2_N + 1).unwrap();
        let zeta = (
            rns_base.0.from_components([rns_base.0.component_ring().coerce(&Fp0, zeta0); 8]),
            rns_base.1.from_components([rns_base.1.component_ring().coerce(&Fp1, zeta1); 8])
        );
        
        let red_rns_base = (RingValue::from(CompletelyReducedSpecialZnx8::from(rns_base.0.clone())), RingValue::from(CompletelyReducedSpecialZnx8::from(rns_base.1.clone())));
        let red_zeta = (
            red_rns_base.0.get_ring().rev_delegate(zeta.0),
            red_rns_base.1.get_ring().rev_delegate(zeta.1),
        );
        let len_N_ntts = (
            CooleyTuckeyFFT::new_with_hom(UnwrapHom::new(red_rns_base.0.clone(), RingValue::from(rns_base.0.clone())), red_rns_base.0.pow(red_zeta.0, 2), log2_N),
            CooleyTuckeyFFT::new_with_hom(UnwrapHom::new(red_rns_base.1.clone(), RingValue::from(rns_base.1.clone())), red_rns_base.1.pow(red_zeta.1, 2), log2_N)
        );
        let log_subgroup_order = StaticRing::<i64>::RING.abs_log2_ceil(&(index_group.group_order() as i64)).unwrap();
        let len_subgroup_order_ntts = (
            CooleyTuckeyFFT::new_with_hom(UnwrapHom::new(red_rns_base.0.clone(), RingValue::from(rns_base.0.clone())), red_rns_base.0.pow(red_zeta.0, 2 << (log2_N - log_subgroup_order)), log_subgroup_order),
            CooleyTuckeyFFT::new_with_hom(UnwrapHom::new(red_rns_base.1.clone(), RingValue::from(rns_base.1.clone())), red_rns_base.1.pow(red_zeta.1, 2 << (log2_N - log_subgroup_order)), log_subgroup_order)
        );
        return Self { ring, transformed_entries, transformed_entries_conj, index_group, reply_a, inner_prod_gk, inner_prod_conj_gk, rns_base, len_N_ntts, len_subgroup_order_ntts, zeta, ciphertext_ring, use_seeds };
    }

    ///
    /// Number of [`CompressedZnx8El`] slots that must be passed as the
    /// `memory` argument of [`PIRDatabase::create`] for a database with
    /// rank `N` (i.e. `SIMD_COUNT * N` entries).
    ///
    /// To translate to bytes, multiply by `size_of::<CompressedZnx8El<26>>()`
    /// (which equals the size of `CompressedZnx8El<25>` — the two layouts are
    /// asserted compatible inside `create`). 
    ///
    #[allow(unused)]
    pub const fn required_memory() -> usize {
        Self::required_memory_general(IndexGroup::full_group(1 << LOG2_N), 1 << LOG2_N)
    }
    
    ///
    /// Like [`PIRDatabase::required_memory()`], but supports also non-standard `index_group`
    /// and `N`.
    /// 
    pub const fn required_memory_general(index_group: IndexGroup, N: usize) -> usize {
        (GK_DIGITS + 1) * 2 * N * index_group.group_order()
    }

    #[allow(unused)]
    pub fn entry_ring(&self) -> &PlainRing {
        &self.ring
    }

    fn Zq(&self) -> &Zn {
        self.ciphertext_ring.base_ring()
    }

    pub fn used_seeds(&self) -> &UsedSeeds {
        &self.use_seeds
    }

    ///
    /// Computes the evaluation of `f = a0 + a1 X + ... + a(l - 1) X^(l - 1)`
    /// at the points `ζ_(4l)^-1, ζ_(4l)^-g, ζ_(4l)^(-g^2), ...` in this order,
    /// where `l = index_group.group_order()` and `g = 5`. If the index group is the
    /// full Galois group, this will additional compute the evaluations at the
    /// points `ζ_(4l)^1, ζ_(4l)^g, ζ_(4l)^g^2, ...`.
    ///  
    /// Note that `ζ_(4l) = base_zeta^zeta_power`, and the roof of unity underlying
    /// the fft table should be `base_zeta^(m/l)`. This means that usually, `zeta_power`
    /// will be divisible by `m/(4l)`.
    /// 
    fn coset_ntt<const K: u32>(
        index_group: IndexGroup,
        Zqx8: RingRef<SpecialZnx8<K>>, 
        base_zeta: SpecialZnx8El<K>, 
        zeta_power: El<Zn>,
        fft_table: &NTT<K>, 
        mut values: impl SwappableVectorViewMut<SpecialZnx8El<K>>
    ) {
        let N = index_group.N();
        let m = 2 * N;
        let l = values.len();
        assert_eq!(l, index_group.group_order());
        let Zm = Zn::new(m as u64);
        assert!(Zm.divides(&zeta_power, &Zm.int_hom().map(index_group.query_base_power().try_into().unwrap())));

        // we operate in two steps: in the first step, we compute the evaluation
        // at `base_zeta^-k` for k in `{ i m/l + offset | i }`. We do this by taking
        // the standard len-`l` ntt over the points `a_i base_zeta^(-i * coset)`,
        // which results in the values `\sum_i a_i base_zeta^(-i (j * m/l + coset))`
        let coset = Zm.smallest_positive_lift(zeta_power) % (m  / l) as i64;
        let mut current_zeta = Zqx8.one();
        let zeta = Zqx8.pow(base_zeta, usize::try_from(coset).unwrap());
        let zeta_inv = Zqx8.invert(&zeta).unwrap();
        for i in 0..values.len() {
            Zqx8.mul_assign(values.at_mut(i), current_zeta);
            Zqx8.mul_assign(&mut current_zeta, zeta_inv);
        }
        fft_table.fft(&mut values, Zqx8);
        
        // the second step is the permute step: after the ntt, we find that the `i`-th entry
        // stores the evaluation at `base_zeta^-(i * m/l + coset)`; we need to write this to
        // the correct index
        let g = index_group.gen_H(&Zm);
        let galois_to_index = |galois_element| usize::try_from(StaticRing::<i64>::RING.checked_div(
            &(Zm.smallest_positive_lift(Zm.negate(galois_element)) - coset), 
            &((m / l) as i64)
        ).unwrap()).unwrap();
        let zeta_power_neg = Zm.negate(zeta_power);
        if index_group.requires_conj() {
            permute(&mut values, |i| if i < N / 2 {
                galois_to_index(Zm.mul(zeta_power_neg, Zm.pow(g, i)))
            } else {
                galois_to_index(Zm.mul(zeta_power, Zm.pow(g, i - N / 2)))
            })
        } else {
            permute(&mut values, |i| galois_to_index(Zm.mul(zeta_power_neg, Zm.pow(g, i))))
        }
    }

    ///
    /// Standard pow-2-length weighted NTT which, given `a0, ..., a(l - 1)`, 
    /// computes `f(ζ^(-1)), f(ζ^(-k - 1)), f(ζ^(-2k - 1)), etc` for 
    /// `f = a0 + a1 X + ... + a(n - 1) * X^(n - 1)`. Here `l` is the length of
    /// the given fft table, and `k = m/l`, where `ζ` is a primitive `m`-th
    /// root of unity.
    /// 
    fn weighted_ntt<const K: u32>(
        Zqx8: RingRef<SpecialZnx8<K>>, 
        zeta: SpecialZnx8El<K>, 
        fft_table: &NTT<K>, 
        mut values: impl SwappableVectorViewMut<SpecialZnx8El<K>>
    ) {
        let mut current_zeta = Zqx8.one();
        let zeta_inv = Zqx8.invert(&zeta).unwrap();
        for i in 0..values.len() {
            Zqx8.mul_assign(values.at_mut(i), current_zeta);
            Zqx8.mul_assign(&mut current_zeta, zeta_inv);
        }
        fft_table.fft(values, Zqx8);
    }

    ///
    /// Inverse of [`PIRDatabase::weighted_ntt()`].
    /// 
    fn inv_weighted_ntt<const K: u32>(
        Zqx8: RingRef<SpecialZnx8<K>>, 
        zeta: SpecialZnx8El<K>, 
        fft_table: &NTT<K>, 
        mut values: impl SwappableVectorViewMut<SpecialZnx8El<K>>
    ) {
        fft_table.inv_fft(&mut values, Zqx8);
        let mut current_zeta = Zqx8.one();
        for i in 0..values.len() {
            Zqx8.mul_assign(values.at_mut(i), current_zeta);
            Zqx8.mul_assign(&mut current_zeta, zeta);
        }
    }

    ///
    /// Computes a homomorphic Galois automorphism on the `a`-part of an
    /// RLWE-based ciphertext.
    /// 
    /// Concretely, a homomorphic Galois automorphism has the form
    /// `(<Decomp(gal(a)), gk_a>, <Decomp(gal(a)), gk_b> + gal(b))`.
    /// This function will thus return `<Decomp(gal(a)), gk_a>` as main
    /// result, and additionally `Decomp(gal(a))` which is necessary 
    /// to later compute the transform on `b`.
    /// 
    #[instrument(skip_all)]
    pub(crate) fn partial_gal(
        rns_base: &(SpecialZnx8<26>, SpecialZnx8<25>),
        zeta: (SpecialZnx8El<26>, SpecialZnx8El<25>),
        ntt: (&NTT<26>, &NTT<25>),
        Zq: Zn,
        current: &RNSEl, 
        Gal: Zn,
        g: El<Zn>, 
        gk_a: [&RNSEl; GK_DIGITS]
    ) -> (RNSEl, [RNSEl; GK_DIGITS]) {
        let N = (*Gal.modulus() / 2) as usize;
        let mut gal_result: RNSEl = (zero_vec(&rns_base.0, N), zero_vec(&rns_base.1, N));
        let inv_g = Gal.invert(&g).unwrap();
        for i in 0..N {
            let in_g = index_to_galois(N, &Gal, i);
            let out_g = Gal.mul(inv_g, in_g);
            let out_i = galois_to_index(N, &Gal, out_g);
            gal_result.0[out_i] = current.0[i];
            gal_result.1[out_i] = current.1[i];
        }
        join(
            || Self::inv_weighted_ntt(RingRef::new(&rns_base.0), zeta.0.into(), &ntt.0, &mut gal_result.0),
            || Self::inv_weighted_ntt(RingRef::new(&rns_base.1), zeta.1.into(), &ntt.1, &mut gal_result.1)
        );

        let avx = Context::check_target_features();

        assert_eq!(2, GK_DIGITS);
        let mut decomposed: (RNSEl, RNSEl) = (
            (zero_vec(&rns_base.0, N), zero_vec(&rns_base.1, N)),
            (zero_vec(&rns_base.0, N), zero_vec(&rns_base.1, N))
        );

        let Zq1 = rns_base.0.component_ring();
        let Zq2 = rns_base.1.component_ring();
        let avx_q: m512i = avx.mm512_set1_epi64(*Zq.modulus());
        let avx_q1_multiple: m512i = avx.mm512_set1_epi64(((*Zq.modulus() >> LOG2_B) / *Zq1.modulus() + 2) * *Zq1.modulus());
        let avx_q2_multiple: m512i = avx.mm512_set1_epi64(((*Zq.modulus() >> LOG2_B) / *Zq2.modulus() + 2) * *Zq2.modulus());
        let avx_q_half: m512i = avx.mm512_set1_epi64(*Zq.modulus() / 2);
        let avx_B_half: m512i = avx.mm512_set1_epi64((1 << LOG2_B) / 2);
        let q2_inv_mod_q1 = Zq1.invert(&Zq1.coerce(Zq2.integer_ring(), *Zq2.modulus())).unwrap();
        let avx_inv_q2_mod_q1 = rns_base.0.from_components([q2_inv_mod_q1; 8]);
        let q1_inv_mod_q2 = Zq2.invert(&Zq2.coerce(Zq2.integer_ring(), *Zq1.modulus())).unwrap();
        let avx_inv_q1_mod_q2 = rns_base.1.from_components([q1_inv_mod_q2; 8]);

        (&mut decomposed.0.0[..]).into_par_iter().zip((&mut decomposed.0.1[..]).into_par_iter()).zip(
            (&mut decomposed.1.0[..]).into_par_iter().zip((&mut decomposed.1.1[..]).into_par_iter())
        ).enumerate().panic_fuse().for_each(|(i, decomposed_entries)| {
            let entry: (m512i, m512i) = (
                rns_base.0.simd_smallest_positive_lift(rns_base.0.mul(gal_result.0[i], avx_inv_q2_mod_q1)),
                rns_base.1.simd_smallest_positive_lift(rns_base.1.mul(gal_result.1[i], avx_inv_q1_mod_q2)),
            );
            let shortest_lift = avx.mm512_add_epi64(
                avx.mm512_mul_epu32(entry.0, rns_base.1.modulus_avx()), 
                avx.mm512_mul_epu32(entry.1, rns_base.0.modulus_avx())
            );
            let greater_than_q = avx.mm512_cmpgt_epi64_mask(shortest_lift, avx_q);
            let shortest_lift = avx.mm512_sub_epi64(shortest_lift, avx.mm512_mask(avx_q, greater_than_q));
            let greater_than_q_half = avx.mm512_cmpgt_epi64_mask(shortest_lift, avx_q_half);
            let shortest_lift = avx.mm512_sub_epi64(shortest_lift, avx.mm512_mask(avx_q, greater_than_q_half));

            let mut current = shortest_lift;
            {
                let high = avx.mm512_srai_epi64::<{LOG2_B as u32}>(avx.mm512_add_epi64(current, avx_B_half));
                let low = avx.mm512_sub_epi64(current, avx.mm512_slli_epi64::<{LOG2_B as u32}>(high));
                *decomposed_entries.0.0 = rns_base.0.simd_from_p_sqr(avx.mm512_add_epi64(low, avx_q1_multiple));
                *decomposed_entries.0.1 = rns_base.1.simd_from_p_sqr(avx.mm512_add_epi64(low, avx_q2_multiple));
                current = high;
            }
            {
                let high = avx.mm512_srai_epi64::<{LOG2_B as u32}>(avx.mm512_add_epi64(current, avx_B_half));
                let low = avx.mm512_sub_epi64(current, avx.mm512_slli_epi64::<{LOG2_B as u32}>(high));
                *decomposed_entries.1.0 = rns_base.0.simd_from_p_sqr(avx.mm512_add_epi64(low, avx_q1_multiple));
                *decomposed_entries.1.1 = rns_base.1.simd_from_p_sqr(avx.mm512_add_epi64(low, avx_q2_multiple));
            }
        });
        let mut decomposed: [RNSEl; GK_DIGITS] = decomposed.into();

        (&mut decomposed).into_par_iter().panic_fuse().for_each(|d| {
            join(
                || Self::weighted_ntt(RingRef::new(&rns_base.0), zeta.0.into(), &ntt.0, &mut d.0),
                || Self::weighted_ntt(RingRef::new(&rns_base.1), zeta.1.into(), &ntt.1, &mut d.1)
            );
        });

        let mut reply_a: RNSEl = (Vec::with_capacity_in(N, AligningAlloc::default()), Vec::with_capacity_in(N, AligningAlloc::default()));
        reply_a.0.extend(
            (0..N).map(|i| rns_base.0.sum((0..GK_DIGITS).map(|k|
                rns_base.0.mul(decomposed[k].0[i], gk_a[k].0[i]),
            )))
        );
        reply_a.1.extend(
            (0..N).map(|i| rns_base.1.sum((0..GK_DIGITS).map(|k|
                rns_base.1.mul(decomposed[k].1[i], gk_a[k].1[i]),
            )))
        );

        return (reply_a, decomposed);
    }

    #[allow(unused)]
    pub fn ciphertext_ring(&self) -> &CipherRing {
        &self.ciphertext_ring
    }

    #[allow(unused)]
    pub fn plaintext_ring(&self) -> &PlainRing {
        &self.ring
    }

    #[allow(unused)]
    pub fn index_group(&self) -> &IndexGroup {
        &self.index_group
    }

    ///
    /// Populates this database with exactly `index_group.group_order()`
    /// plaintext entries, overwriting any previous content.
    ///
    /// The iterator yields ring elements of the [`PlainRing`] that was passed to
    /// [`PIRDatabase::create`]; with the default parameters that is
    /// `Z_1025[X] / (X^N + 1)`. Entries are consumed in their natural
    /// addressing order, so the `k`-th element yielded by `entries` will be the
    /// one returned for `primary_idx = k` (within this base-level database).
    ///
    /// This call is **the** expensive preprocessing step: it runs the FFTs,
    /// Galois rotations and gadget decompositions that subsequent
    /// [`PIRDatabase::perform_pir`] / `process_query` calls just stream over.
    /// Expect a multi-second cost per database with the default parameters; the
    /// resulting state can be cached and reused for as many queries as needed.
    ///
    /// # Panics
    ///
    /// Panics if `entries` does not yield exactly `index_group.group_order()` elements.
    ///
    #[instrument(skip_all)]
    pub fn set_db<I>(&mut self, entries: I)
        where I: IntoIterator<Item = El<PlainRing>>
    {
        self.set_transformed_entries(entries);
        self.preprocess();
    }

    ///
    /// Computes the values
    /// ```text
    ///   \hat{a}_j = \sum_{i < l} a_i ζ_(4l)^{-ij}    in Z_t[X]/(X^N + 1)
    /// ```
    /// for `j in H = { 1, 5, 9, 13, ..., 4l - 3 }`. In case the index space is
    /// the full group, then instead we will have `H = (Z/mZ)*`.
    /// 
    /// These are ordered as `\hat{a}_{5^{l - 1}}, ..., \hat{a}_5, \hat{a}_1`,
    /// then the automorphisms `X -> X^{5^{-l + 1}}, ..., X -> X^{5^-1}, X -> X`
    /// are applied, and then of the resulting plaintext ring elements is embedded
    /// into the ciphertext ring. Finally, each resulting ciphertext ring element is
    /// transformed into CRT form - the results are stored in
    /// [`PIRDatabase::transformed_entries`]. 
    /// 
    /// Similarly, if required for the current index group, the same is done for the
    /// values `\hat{a}_{-5^{N/2 - 1}}, ..., \hat{a}_-5, \hat{a}_-1`
    /// 
    /// Note that this is exactly the format that they are read during
    /// [`PIRDatabase::perform_pir()`], and this specific ordering allows for
    /// sequential access.
    /// 
    #[instrument(skip_all)]
    fn set_transformed_entries<I>(&mut self, entries: I)
        where I: IntoIterator<Item = El<PlainRing>>
    {
        let R = self.ring;
        let N = R.rank();
        let Zm = Zn::new((2 * N).try_into().unwrap());
        let t = *R.base_ring().modulus();
        let max_coeff = 2 * t * N as i64;
        assert!(max_coeff < *self.rns_base.0.modulus());
        let Zqx8 = RingRef::new(&self.rns_base.0);
        let zeta = self.zeta.0;
        let fft_table = &self.len_N_ntts.0;
        
        // STEP 1: Columns of `result` = coefficient vectors of input elements
        let mut result = OwnedMatrix::zero(N, self.index_group.group_order(), Zqx8);
        let mut actual_len = 0;
        let input = entries.into_iter().inspect(|_| actual_len += 1);
        for (j, x) in input.array_chunks::<SIMD_COUNT>().enumerate() {
            let db_j_wrt_basis: [_; SIMD_COUNT] = from_fn(|k| R.wrt_canonical_basis(&x[k]));
            for i in 0..N {
                *result.at_mut(i, j) = Zqx8.get_ring().from_components(from_fn(|k| db_j_wrt_basis[k].at(i))).into();
            }
        }
        // println!("embedded coefficients");
        // println!("{}", format_matrix(N, self.index_group.group_order(), |i, j| result.at(i, j), RingRef::new(&self.rns_base.0)));

        // STEP 2: Columns of `result` = input elements in CRT form
        let cols = result.data_mut().col_iter().collect::<Vec<_>>();
        cols.into_par_iter().panic_fuse().for_each(|c| {
            Self::weighted_ntt(Zqx8, zeta.into(), fft_table, c);
        });
        assert_eq!(self.index_group.group_order() * SIMD_COUNT, actual_len);

        // println!("in crt form");
        // println!("{}", format_matrix(N, self.index_group.group_order(), |i, j| result.at(i, j), RingRef::new(&self.rns_base.0)));

        // STEP 3: Columns of `result` = FFT'ed elements in CRT form (these are the \hat{a}_j)
        let rows = result.data_mut().row_iter().collect::<Vec<_>>();
        let base_power = Zm.negate(Zm.coerce(&StaticRing::<i64>::RING, self.index_group.query_base_power() as i64));
        rows.into_par_iter().enumerate().panic_fuse()
            .for_each(|(i, r)| Self::coset_ntt(self.index_group, Zqx8, zeta.into(), Zm.mul(index_to_galois(N, &Zm, i), base_power), &self.len_subgroup_order_ntts.0, r));
        
        // println!("row-wise ntt");
        // println!("{}", format_matrix(N, self.index_group.group_order(), |i, j| result.at(i, j), RingRef::new(&self.rns_base.0)));
        // result.data_mut().col_iter().collect::<Vec<_>>().into_par_iter().for_each(|col| Self::inv_weighted_ntt(Zqx8, zeta, fft_table, col));
        // println!("temporarily un-ntt'ed");
        // println!("{}", format_matrix(N, self.index_group.group_order(), |i, j| result.at(i, j), RingRef::new(&self.rns_base.0)));
        // result.data_mut().col_iter().collect::<Vec<_>>().into_par_iter().for_each(|col| Self::weighted_ntt(Zqx8, zeta, fft_table, col));

        // STEP 4: Undo Galois automorphisms that will be applied during the main online loop
        let cols = result.data_mut().col_iter().collect::<Vec<_>>();
        let g = self.index_group.gen_H(&Zm);
        let g_inv = Zm.invert(&g).unwrap();
        cols.into_par_iter().enumerate().panic_fuse().for_each(|(j, col)| {
            let automorphism = Zm.pow(g_inv, j);
            permute(col, |i| galois_to_index(N, &Zm, Zm.mul(automorphism, index_to_galois(N, &Zm, i))));
        });
        
        // println!("column-wise automorphisms");
        // println!("{}", format_matrix(N, self.index_group.group_order(), |i, j| result.at(i, j), RingRef::new(&self.rns_base.0)));
        // result.data_mut().col_iter().collect::<Vec<_>>().into_par_iter().for_each(|col| Self::inv_weighted_ntt(Zqx8, zeta, fft_table, col));
        // println!("temporarily un-ntt'ed");
        // println!("{}", format_matrix(N, self.index_group.group_order(), |i, j| result.at(i, j), RingRef::new(&self.rns_base.0)));
        // result.data_mut().col_iter().collect::<Vec<_>>().into_par_iter().for_each(|col| Self::weighted_ntt(Zqx8, zeta, fft_table, col));

        // STEP 5: Reverse columns of `result` within `<g>` and `-<g>`
        let rows = result.data_mut().row_iter().collect::<Vec<_>>();
        if self.index_group.requires_conj() {
            rows.into_par_iter().panic_fuse().for_each(|row| {
                (&mut row[..(N / 2)]).reverse();
                (&mut row[(N / 2)..]).reverse();
            });
        } else {
            rows.into_par_iter().panic_fuse().for_each(|row| {
                row.reverse();
            });
        }

        // STEP 6: Embed into full ciphertext space
        let cols = result.data_mut().col_iter().collect::<Vec<_>>();
        let ZZ_to_Rbase = R.base_ring().can_hom(&StaticRing::<i64>::RING).unwrap();
        let scale = R.base_ring().invert(&R.base_ring().coerce(&StaticRing::<i64>::RING, self.index_group.group_order() as i64)).unwrap();
        let result_0 = self.transformed_entries.0.par_chunks_mut(N).chain(self.transformed_entries_conj.0.par_chunks_mut(N));
        let result_1 = self.transformed_entries.1.par_chunks_mut(N).chain(self.transformed_entries_conj.1.par_chunks_mut(N));
        cols.into_par_iter().panic_fuse().zip(result_0.zip(result_1)).for_each(|(mut c, (res_0, res_1))| {
            Self::inv_weighted_ntt(Zqx8, zeta.into(), &fft_table, c.reborrow());
            let entries = (0..N).map(|i| {
                let entry = Zqx8.get_ring().get_components(*c.at(i));
                from_fn::<_, SIMD_COUNT, _>(|k| R.base_ring().smallest_lift(R.base_ring().mul(ZZ_to_Rbase.map(Zqx8.get_ring().component_ring().smallest_lift(entry[k])), scale)) as i32)
            }).collect::<Vec<_>>();
            let mut result: RNSEl = (Vec::with_capacity_in(N, AligningAlloc::default()), Vec::with_capacity_in(N, AligningAlloc::default()));
            result.0.extend(entries.iter().map(|x| self.rns_base.0.from_components(from_fn(|k| self.rns_base.0.component_ring().int_hom().map(x[k])))));
            result.1.extend(entries.iter().map(|x| self.rns_base.1.from_components(from_fn(|k| self.rns_base.1.component_ring().int_hom().map(x[k])))));
            Self::weighted_ntt(RingRef::new(&self.rns_base.0), self.zeta.0.into(), &self.len_N_ntts.0, &mut result.0[..]);
            Self::weighted_ntt(RingRef::new(&self.rns_base.1), self.zeta.1.into(), &self.len_N_ntts.1, &mut result.1[..]);
            let (val_0, val_1) = Self::compress(N, &self.rns_base, &result);
            for (j, x) in val_0.into_iter().enumerate() {
                res_0[j] = x;
            }
            for (j, x) in val_1.into_iter().enumerate() {
                res_1[j] = x;
            }
        });

        // println!("result");
        // let result_matrix = OwnedMatrix::from_fn(N, self.index_group.loop_len(), |i, j| self.rns_base.0.uncompress(self.transformed_entries.0[i + j * N]));
        // println!("{}", format_matrix(N, self.index_group.loop_len(), |i, j| result_matrix.at(i, j), RingRef::new(&self.rns_base.0)));
        
        self.reply_a.0 = Vec::new_in(AligningAlloc::default());
        self.reply_a.1 = Vec::new_in(AligningAlloc::default());
    }

    fn compress(N: usize, rns_base: &(SpecialZnx8<26>, SpecialZnx8<25>), el: &RNSEl) -> CompressedRNSEl {
        let mut result = (Vec::with_capacity_in(N, AligningAlloc::default()), Vec::with_capacity_in(N, AligningAlloc::default()));
        result.0.extend(el.0.iter().map(|x| rns_base.0.compress(*x)));
        result.1.extend(el.1.iter().map(|x| rns_base.1.compress(*x)));
        return result;
    }
    
    #[instrument(skip_all)]
    fn preprocess(&mut self) {
        let gk_a: [_; GK_DIGITS] = from_fn(|i| self.to_simd_element([&expand(&self.ciphertext_ring, self.use_seeds.get_gk_seeds()[i]); SIMD_COUNT]));
        let qry_a = self.to_simd_element([&expand(&self.ciphertext_ring, self.use_seeds.get_qry_seeds()[0]); SIMD_COUNT]);
        let qry_conj_a = if self.index_group.requires_conj() {
            self.to_simd_element([&expand(&self.ciphertext_ring, self.use_seeds.get_qry_seeds()[1]); SIMD_COUNT])
        } else {
            (Vec::new_in(AligningAlloc::default()), Vec::new_in(AligningAlloc::default()))
        };
        let N = self.ring.rank();
        let Zm = Zn::new((2 * N).try_into().unwrap());
        let g = self.index_group.gen_H(&Zm);
        let mut sum: RNSEl = (zero_vec(&self.rns_base.0, N), zero_vec(&self.rns_base.1, N));
        let mut sum_conj: RNSEl = if self.index_group.requires_conj() {
            (zero_vec(&self.rns_base.0, N), zero_vec(&self.rns_base.1, N))
        } else {
            (Vec::new_in(AligningAlloc::default()), Vec::new_in(AligningAlloc::default()))
        };

        let Zq = self.ciphertext_ring.base_ring();
        let inner_prod_gk: [_; 2] = self.inner_prod_gk.iter_mut().map(|x| (&mut x.0[..], &mut x.1[..])).collect::<Vec<_>>().try_into().ok().unwrap();
        let inner_prod_gk_conj: [_; 2] = self.inner_prod_conj_gk.iter_mut().map(|x| (&mut x.0[..], &mut x.1[..])).collect::<Vec<_>>().try_into().ok().unwrap();

        join(
            || preprocess_loop(self.zeta, &self.len_N_ntts, Zq, N, Zm, g, self.index_group, &self.rns_base, (&qry_a.0[..], &qry_a.1[..]), &gk_a, &mut sum, inner_prod_gk, (&self.transformed_entries.0, &self.transformed_entries.1)), 
            || if self.index_group.requires_conj() {
                preprocess_loop(self.zeta, &self.len_N_ntts, Zq, N, Zm, g, self.index_group, &self.rns_base, (&qry_conj_a.0[..], &qry_conj_a.1[..]), &gk_a, &mut sum_conj, inner_prod_gk_conj, (&self.transformed_entries_conj.0, &self.transformed_entries_conj.1));
            }
        );
        
        if self.index_group.requires_conj() {
            for j in 0..N {
                self.rns_base.0.add_assign(&mut sum.0[j], sum_conj.0[j]);
                self.rns_base.1.add_assign(&mut sum.1[j], sum_conj.1[j]);
            }
        }
        self.reply_a = sum;

        fn preprocess_loop(
            zeta: (SpecialZnx8El<26>, SpecialZnx8El<25>),
            ntts: &(NTT<26>, NTT<25>),
            Zq: &Zn,
            N: usize,
            Gal: Zn,
            g: El<Zn>,
            index_group: IndexGroup,
            rns_base: &(SpecialZnx8<26>, SpecialZnx8<25>),
            qry_a: (&[SpecialZnx8El<26>], &[SpecialZnx8El<25>]),
            gk_a: &[(Vec<SpecialZnx8El<26>, AligningAlloc>, Vec<SpecialZnx8El<25>, AligningAlloc>); GK_DIGITS],
            sum: &mut (Vec<SpecialZnx8El<26>, AligningAlloc>, Vec<SpecialZnx8El<25>, AligningAlloc>),
            inner_prod_gk: [(&mut [CompressedZnx8El<26>], &mut [CompressedZnx8El<25>]); GK_DIGITS],
            transformed_entries: (&[CompressedZnx8El<26>], &[CompressedZnx8El<25>])
        ) {
            assert!(inner_prod_gk.iter().all(|l| l.0.len() == N * index_group.loop_len()));
            assert!(inner_prod_gk.iter().all(|l| l.1.len() == N * index_group.loop_len()));
            assert_eq!(N * index_group.loop_len(), transformed_entries.0.len());
            assert_eq!(N * index_group.loop_len(), transformed_entries.1.len());
            for i in 0..index_group.loop_len() {
                let (gal_sum, gal_sum_b) = PIRDatabase::partial_gal(
                    rns_base,
                    zeta,
                    (&ntts.0, &ntts.1),
                    *Zq,
                    sum,
                    Gal,
                    g,
                    from_fn(|k| &gk_a[k])
                );
                for k in 0..GK_DIGITS {
                    for (j, x) in gal_sum_b[k].0.iter().enumerate() {
                        inner_prod_gk[k].0[i * N + j] = rns_base.0.compress(*x);
                    }
                    for (j, x) in gal_sum_b[k].1.iter().enumerate() {
                        inner_prod_gk[k].1[i * N + j] = rns_base.1.compress(*x);
                    }
                }
                for j in 0..N {
                    sum.0[j] = rns_base.0.add(gal_sum.0[j], rns_base.0.mul(qry_a.0[j], rns_base.0.uncompress(transformed_entries.0[i * N + j])));
                    sum.1[j] = rns_base.1.add(gal_sum.1[j], rns_base.1.mul(qry_a.1[j], rns_base.1.uncompress(transformed_entries.1[i * N + j])));
                }
            }
        }
    }

    fn to_simd_element(&self, els: [&El<CipherRing>; SIMD_COUNT]) -> RNSEl {
        let N = self.ring.rank();
        let C = &self.ciphertext_ring;
        let coeffs = els.map(|el| C.wrt_canonical_basis(el));
        let mut result: RNSEl = (Vec::with_capacity_in(N, AligningAlloc::default()), Vec::with_capacity_in(N, AligningAlloc::default()));
        result.0.extend((0..N).map(|i| self.rns_base.0.from_components(coeffs.map(|cs| self.rns_base.0.component_ring().coerce(&StaticRing::<i64>::RING, C.base_ring().smallest_lift(cs.at(i)))))));
        result.1.extend((0..N).map(|i| self.rns_base.1.from_components(coeffs.map(|cs| self.rns_base.1.component_ring().coerce(&StaticRing::<i64>::RING, C.base_ring().smallest_lift(cs.at(i)))))));
        Self::weighted_ntt(RingRef::new(&self.rns_base.0), self.zeta.0.into(), &self.len_N_ntts.0, &mut result.0);
        Self::weighted_ntt(RingRef::new(&self.rns_base.1), self.zeta.1.into(), &self.len_N_ntts.1, &mut result.1);
        let mut res = (Vec::with_capacity_in(N, AligningAlloc::default()), Vec::with_capacity_in(N, AligningAlloc::default()));
        res.0.extend(result.0.into_iter());
        res.1.extend(result.1.into_iter());
        return res;
    }

    fn to_compressed_simd_element(&self, el: [&El<CipherRing>; SIMD_COUNT]) -> CompressedRNSEl {
        let as_simd = self.to_simd_element(el);
        let N = self.ring.rank();
        let mut res = (Vec::with_capacity_in(N, AligningAlloc::default()), Vec::with_capacity_in(N, AligningAlloc::default()));
        res.0.extend(as_simd.0.into_iter().map(|x| self.rns_base.0.compress(x)));
        res.1.extend(as_simd.1.into_iter().map(|x| self.rns_base.1.compress(x)));
        return res;
    }

    ///
    /// Returns the `a`-components of the four result ciphertext.
    /// This does not depend on any client-provided data, but does depend on the
    /// contents of the database, i.e. will change when `set_db()` is called.
    /// 
    /// The `b`-components of these ciphertexts can be retrieved via `perform_pir()`.
    /// 
    #[instrument(skip_all)]
    pub fn get_a(&self) -> [El<CipherRing>; SIMD_COUNT] {
        let N = self.ring.rank();
        let mut res = (Vec::with_capacity_in(N, AligningAlloc::default()), Vec::with_capacity_in(N, AligningAlloc::default()));
        res.0.extend((0..N).map(|i| self.reply_a.0[i]));
        res.1.extend((0..N).map(|i| self.reply_a.1[i]));
        return self.from_simd_element(res);
    }

    fn from_simd_element(&self, mut el: RNSEl) -> [El<CipherRing>; SIMD_COUNT] {
        let N = self.ring.rank();
        let C = &self.ciphertext_ring;
        Self::inv_weighted_ntt(RingRef::new(&self.rns_base.0), self.zeta.0.into(), &self.len_N_ntts.0, &mut el.0);
        Self::inv_weighted_ntt(RingRef::new(&self.rns_base.1), self.zeta.1.into(), &self.len_N_ntts.1, &mut el.1);

        let mod_q = self.Zq().can_hom(&StaticRing::<i64>::RING).unwrap();
        let Zq1 = self.rns_base.0.component_ring();
        let Zq2 = self.rns_base.1.component_ring();
        let q2_inv_mod_q1 = Zq1.invert(&Zq1.coerce(Zq2.integer_ring(), *Zq2.modulus())).unwrap();
        let q1_inv_mod_q2 = Zq2.invert(&Zq2.coerce(Zq2.integer_ring(), *Zq1.modulus())).unwrap();
        let mut result: [Vec<_>; SIMD_COUNT] = from_fn(|_| Vec::with_capacity(N));
        for i in 0..N {
            let entry = (
                self.rns_base.0.get_components(el.0[i]),
                self.rns_base.1.get_components(el.1[i]),
            );
            let in_zq: [_; SIMD_COUNT] = from_fn(|k| self.Zq().add(
                self.Zq().mul(self.Zq().coerce(Zq2.integer_ring(), *Zq2.modulus()), mod_q.map(
                    Zq1.smallest_lift(Zq1.mul(entry.0[k], q2_inv_mod_q1))
                )),
                self.Zq().mul(self.Zq().coerce(Zq1.integer_ring(), *Zq1.modulus()), mod_q.map(
                    Zq2.smallest_lift(Zq2.mul(entry.1[k], q1_inv_mod_q2))
                )),
            ));
            for k in 0..SIMD_COUNT {
                result[k].push(in_zq[k]);
            }
        }
        return from_fn(|i| C.from_canonical_basis(result[i].iter().copied()));
    }

    ///
    /// Returns the `b`-components of four ciphertexts encrypting the ring elements at
    /// positions `4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3` in this database, assuming
    /// that `qry_b` and `qry_b_con` encrypt `X^i` resp. `X^-i`.
    /// 
    /// The `a`-components of these ciphertexts can be retrieved via `get_a()`.
    /// 
    #[allow(unused)]
    pub fn perform_pir(
        &self, 
        qry_b: &El<CipherRing>, 
        qry_conj_b: Option<&El<CipherRing>>, 
        gk_b: [&El<CipherRing>; GK_DIGITS]
    ) -> [El<CipherRing>; SIMD_COUNT] {
        self.perform_pir_split([QueryRef { qry_b, qry_conj_b, gk_b }; SIMD_COUNT])
    }

    /// 
    /// Runs the `i`-th query on the `i`-th SIMD-entry of this database.
    ///  
    #[instrument(skip_all)]
    pub fn perform_pir_split(
        &self, 
        queries: [QueryRef; SIMD_COUNT]
    ) -> [El<CipherRing>; SIMD_COUNT] {
        let N = self.ring.rank();
        let qry_b = self.to_compressed_simd_element(from_fn(|i| queries[i].qry_b));
        let qry_b_conj = if self.index_group.requires_conj() {
            self.to_compressed_simd_element(from_fn(|i| queries[i].qry_conj_b.expect("conjugate query must be given when index subgroup requires conjugation")))
        } else {
            (Vec::new_in(AligningAlloc::default()), Vec::new_in(AligningAlloc::default()))
        };
        let gk_b: [_; GK_DIGITS] = from_fn(|i| self.to_compressed_simd_element(from_fn(|j| queries[j].gk_b[i])));

        let Zm = Zn::new((2 * N).try_into().unwrap());
        let g = self.index_group.gen_H(&Zm);
        let inv_g = Zm.invert(&g).unwrap();
        let galois_lookup_table = (0..N).map(|i| {
            let in_g = Zm.get_ring().from_int_promise_reduced(2 * i as i64 + 1);
            let out_g = Zm.mul(inv_g, in_g);
            return (Zm.smallest_positive_lift(out_g) as usize - 1) / 2;
        }).collect::<Vec<_>>();

        let zero_vec = || {
            let mut res = Vec::with_capacity_in(N, AligningAlloc::default());
            res.extend((0..N).map(|_| self.rns_base.0.zero().to_raw()));
            return res;
        };

        let mut sum: (Vec<m512i, AligningAlloc>, Vec<m512i, AligningAlloc>) = (zero_vec(), zero_vec());
        let mut sum_new: (Vec<m512i, AligningAlloc>, Vec<m512i, AligningAlloc>) = (zero_vec(), zero_vec());
        let mut sum_conj: (Vec<m512i, AligningAlloc>, Vec<m512i, AligningAlloc>) = if self.index_group.requires_conj() {
            (zero_vec(), zero_vec())
        } else {
            (Vec::new_in(AligningAlloc::default()), Vec::new_in(AligningAlloc::default()))
        };
        let mut sum_conj_new: (Vec<m512i, AligningAlloc>, Vec<m512i, AligningAlloc>) = if self.index_group.requires_conj() {
            (zero_vec(), zero_vec())
        } else {
            (Vec::new_in(AligningAlloc::default()), Vec::new_in(AligningAlloc::default()))
        };

        let avx = Context::check_target_features();

        // here, we manually manage precision; note that we sum up `N * (GK_DIGITS + 1)` products of integers `<= repr_bound`
        let reduce_every_iterations = min(
            u64::MAX as i128 / self.rns_base.0.repr_bound() as i128 / self.rns_base.0.repr_bound() as i128 / (GK_DIGITS + 1) as i128,
            u64::MAX as i128 / self.rns_base.1.repr_bound() as i128 / self.rns_base.1.repr_bound() as i128 / (GK_DIGITS + 1) as i128,
        ) as usize;

        join(|| join(
        || main_pir_loop(
            avx,
            N,
            self.index_group.loop_len(),
            reduce_every_iterations,
            &self.rns_base.0,
            &mut sum.0,
            &mut sum_new.0,
            &galois_lookup_table,
            &qry_b.0,
            from_fn(|i| &gk_b[i].0[..]),
            from_fn(|i| &*self.inner_prod_gk[i].0),
            &self.transformed_entries.0
        ),
        || main_pir_loop(
            avx,
            N,
            self.index_group.loop_len(),
            reduce_every_iterations,
            &self.rns_base.1,
            &mut sum.1,
            &mut sum_new.1,
            &galois_lookup_table,
            &qry_b.1,
            from_fn(|i| &gk_b[i].1[..]),
            from_fn(|i| &*self.inner_prod_gk[i].1),
            &self.transformed_entries.1
        )), || if self.index_group.requires_conj() { 
            join(|| main_pir_loop(
                avx,
                N,
                self.index_group.loop_len(),
                reduce_every_iterations,
                &self.rns_base.0,
                &mut sum_conj.0,
                &mut sum_conj_new.0,
                &galois_lookup_table,
                &qry_b_conj.0,
                from_fn(|i| &gk_b[i].0[..]),
                from_fn(|i| &*self.inner_prod_conj_gk[i].0),
                &self.transformed_entries_conj.0
            ),
            || main_pir_loop(
                avx,
                N,
                self.index_group.loop_len(),
                reduce_every_iterations,
                &self.rns_base.1,
                &mut sum_conj.1,
                &mut sum_conj_new.1,
                &galois_lookup_table,
                &qry_b_conj.1,
                from_fn(|i| &gk_b[i].1[..]),
                from_fn(|i| &*self.inner_prod_conj_gk[i].1),
                &self.transformed_entries_conj.1
            )); 
        });

        if self.index_group.requires_conj() {
            for j in 0..N {
                sum.0[j] = avx.mm512_add_epi64(sum.0[j], sum_conj.0[j]);
                sum.1[j] = avx.mm512_add_epi64(sum.1[j], sum_conj.1[j]);
            }
        }
        permute(&mut sum.0, |i| galois_lookup_table[i]);
        permute(&mut sum.1, |i| galois_lookup_table[i]);
        let mut res = (Vec::with_capacity_in(N, AligningAlloc::default()), Vec::with_capacity_in(N, AligningAlloc::default()));
        res.0.extend(sum.0.into_iter().map(|x| self.rns_base.0.reduce_from_u64(x)));
        res.1.extend(sum.1.into_iter().map(|x| self.rns_base.1.reduce_from_u64(x)));
        return self.from_simd_element(res);

        ///
        /// This is a non-inlined, stand-alone function to facilitate optimizations, in particular
        /// make it easier to read the generated assembly
        /// 
        #[inline(never)]
        #[instrument(skip_all)]
        fn main_pir_loop<'a, const K: u32>(
            avx: Context,
            N: usize,
            len: usize,
            reduce_every_iterations: usize,
            Zqx8: &SpecialZnx8<K>, 
            mut sum: &'a mut [m512i],
            mut sum_new: &'a mut [m512i],
            galois_lookup_table: &[usize],
            qry_b: &[CompressedZnx8El<K>],
            gk_b: [&[CompressedZnx8El<K>]; GK_DIGITS],
            inner_prod_gk: [&[CompressedZnx8El<K>]; GK_DIGITS],
            transformed_entries: &[CompressedZnx8El<K>]
        ) {
            assert_eq!(N * len, transformed_entries.len());
            assert_eq!(N * len, inner_prod_gk[0].len());
            assert_eq!(N * len, inner_prod_gk[1].len());
            assert_eq!(N, qry_b.len());
            assert_eq!(N, galois_lookup_table.len());
            assert_eq!(N, gk_b[0].len());
            assert_eq!(N, gk_b[1].len());
            assert!(galois_lookup_table.iter().all(|i| *i < N));
            assert!(transformed_entries.as_ptr().is_aligned_to(Alignment::of::<m512i>().as_usize()));
            assert!(inner_prod_gk[0].as_ptr().is_aligned_to(Alignment::of::<m512i>().as_usize()));
            assert!(inner_prod_gk[1].as_ptr().is_aligned_to(Alignment::of::<m512i>().as_usize()));

            let mut index = 0;

            for i_outer in (0..len).step_by(reduce_every_iterations) {

                for i_inner in 0..min(len - i_outer, reduce_every_iterations) {

                    let mut preload_inner_prod_gk_0_0 = unsafe { *(inner_prod_gk[0].as_ptr().offset(index) as *const m512i) };
                    let mut preload_inner_prod_gk_0_1 = unsafe { *(inner_prod_gk[0].as_ptr().offset(index + 2) as *const m512i) };
                    let mut preload_inner_prod_gk_0_2 = unsafe { *(inner_prod_gk[0].as_ptr().offset(index + 4) as *const m512i) };
                    let mut preload_inner_prod_gk_0_3 = unsafe { *(inner_prod_gk[0].as_ptr().offset(index + 6) as *const m512i) };
                    let mut preload_inner_prod_gk_0_4 = unsafe { *(inner_prod_gk[0].as_ptr().offset(index + 8) as *const m512i) };
                    let mut preload_inner_prod_gk_0_5 = unsafe { *(inner_prod_gk[0].as_ptr().offset(index + 10) as *const m512i) };
                    let mut preload_inner_prod_gk_0_6 = unsafe { *(inner_prod_gk[0].as_ptr().offset(index + 12) as *const m512i) };
                    let mut preload_inner_prod_gk_0_7 = unsafe { *(inner_prod_gk[0].as_ptr().offset(index + 14) as *const m512i) };
                    
                    let mut preload_inner_prod_gk_1_0 = unsafe { *(inner_prod_gk[1].as_ptr().offset(index) as *const m512i) };
                    let mut preload_inner_prod_gk_1_1 = unsafe { *(inner_prod_gk[1].as_ptr().offset(index + 2) as *const m512i) };
                    let mut preload_inner_prod_gk_1_2 = unsafe { *(inner_prod_gk[1].as_ptr().offset(index + 4) as *const m512i) };
                    let mut preload_inner_prod_gk_1_3 = unsafe { *(inner_prod_gk[1].as_ptr().offset(index + 6) as *const m512i) };
                    let mut preload_inner_prod_gk_1_4 = unsafe { *(inner_prod_gk[1].as_ptr().offset(index + 8) as *const m512i) };
                    let mut preload_inner_prod_gk_1_5 = unsafe { *(inner_prod_gk[1].as_ptr().offset(index + 10) as *const m512i) };
                    let mut preload_inner_prod_gk_1_6 = unsafe { *(inner_prod_gk[1].as_ptr().offset(index + 12) as *const m512i) };
                    let mut preload_inner_prod_gk_1_7 = unsafe { *(inner_prod_gk[1].as_ptr().offset(index + 14) as *const m512i) };
                    
                    let mut preload_transformed_entries_0 = unsafe { *(transformed_entries.as_ptr().offset(index) as *const m512i) };
                    let mut preload_transformed_entries_1 = unsafe { *(transformed_entries.as_ptr().offset(index + 2) as *const m512i) };
                    let mut preload_transformed_entries_2 = unsafe { *(transformed_entries.as_ptr().offset(index + 4) as *const m512i) };
                    let mut preload_transformed_entries_3 = unsafe { *(transformed_entries.as_ptr().offset(index + 6) as *const m512i) };
                    let mut preload_transformed_entries_4 = unsafe { *(transformed_entries.as_ptr().offset(index + 8) as *const m512i) };
                    let mut preload_transformed_entries_5 = unsafe { *(transformed_entries.as_ptr().offset(index + 10) as *const m512i) };
                    let mut preload_transformed_entries_6 = unsafe { *(transformed_entries.as_ptr().offset(index + 12) as *const m512i) };
                    let mut preload_transformed_entries_7 = unsafe { *(transformed_entries.as_ptr().offset(index + 14) as *const m512i) };

                    let mut current: m512i;
                    let mut current_gk_b_0: m512i;
                    let mut current_gk_b_1: m512i;
                    let mut current_qry_b: m512i;
                    let mut prod: m512i;
                    let mut uncompressed: m512i;

                    let mut j = 0;

                    macro_rules! unrolled_loop_body {
                        ($should_preload:literal; $($idx:literal),*) => {

                            $(
                            
                            current_gk_b_0 = unsafe { *(gk_b[0].as_ptr().offset(j as isize + $idx * 2) as *const m512i) };
                            current_gk_b_1 = unsafe { *(gk_b[1].as_ptr().offset(j as isize + $idx * 2) as *const m512i) };
                            current_qry_b = unsafe { *(qry_b.as_ptr().offset(j as isize + $idx * 2) as *const m512i) };
                            current = unsafe { *sum.get_unchecked(j + $idx * 2) };

                            prod = avx.mm512_cvtepi32_epi64(avx.mm512_extract_si256(${concat(preload_inner_prod_gk_0_, $idx)}).0);
                            uncompressed = avx.mm512_cvtepi32_epi64(avx.mm512_extract_si256(current_gk_b_0).0);
                            prod = avx.mm512_mul_epu32(prod, uncompressed);
                            current = avx.mm512_add_epi64(current, prod);
                            
                            prod = avx.mm512_cvtepi32_epi64(avx.mm512_extract_si256(${concat(preload_inner_prod_gk_1_, $idx)}).0);
                            uncompressed = avx.mm512_cvtepi32_epi64(avx.mm512_extract_si256(current_gk_b_1).0);
                            prod = avx.mm512_mul_epu32(prod, uncompressed);
                            current = avx.mm512_add_epi64(current, prod);

                            prod = avx.mm512_cvtepi32_epi64(avx.mm512_extract_si256(${concat(preload_transformed_entries_, $idx)}).0);
                            uncompressed = avx.mm512_cvtepi32_epi64(avx.mm512_extract_si256(current_qry_b).0);
                            prod = avx.mm512_mul_epu32(prod, uncompressed);
                            current = avx.mm512_add_epi64(current, prod);

                            *unsafe { sum_new.get_unchecked_mut(galois_lookup_table[j + $idx * 2]) } = current;

                            current = unsafe { *sum.get_unchecked(j + $idx * 2 + 1) };

                            prod = avx.mm512_cvtepi32_epi64(avx.mm512_extract_si256(${concat(preload_inner_prod_gk_0_, $idx)}).1);
                            uncompressed = avx.mm512_cvtepi32_epi64(avx.mm512_extract_si256(current_gk_b_0).1);
                            prod = avx.mm512_mul_epu32(prod, uncompressed);
                            current = avx.mm512_add_epi64(current, prod);
                            
                            prod = avx.mm512_cvtepi32_epi64(avx.mm512_extract_si256(${concat(preload_inner_prod_gk_1_, $idx)}).1);
                            uncompressed = avx.mm512_cvtepi32_epi64(avx.mm512_extract_si256(current_gk_b_1).1);
                            prod = avx.mm512_mul_epu32(prod, uncompressed);
                            current = avx.mm512_add_epi64(current, prod);

                            prod = avx.mm512_cvtepi32_epi64(avx.mm512_extract_si256(${concat(preload_transformed_entries_, $idx)}).1);
                            uncompressed = avx.mm512_cvtepi32_epi64(avx.mm512_extract_si256(current_qry_b).1);
                            prod = avx.mm512_mul_epu32(prod, uncompressed);
                            current = avx.mm512_add_epi64(current, prod);

                            *unsafe { sum_new.get_unchecked_mut(galois_lookup_table[j + $idx * 2 + 1]) } = current;

                            #[allow(unused)]
                            if $should_preload {
                                ${concat(preload_inner_prod_gk_0_, $idx)} = unsafe { *(inner_prod_gk[0].as_ptr().offset(index + $idx * 2 + 16) as *const m512i) };
                                ${concat(preload_inner_prod_gk_1_, $idx)} = unsafe { *(inner_prod_gk[1].as_ptr().offset(index + $idx * 2 + 16) as *const m512i) };
                                ${concat(preload_transformed_entries_, $idx)} = unsafe { *(transformed_entries.as_ptr().offset(index + $idx * 2 + 16) as *const m512i) };
                            }

                            )*
                        };
                    }

                    const UNROLL_COUNT: usize = 16;
                    assert!(N % UNROLL_COUNT == 0);
                    for _ in (0..(N - UNROLL_COUNT)).step_by(UNROLL_COUNT) {
                        unrolled_loop_body!(true; 0, 1, 2, 3, 4, 5, 6, 7);
                        j += 16;
                        index += 16;
                    }
                    unrolled_loop_body!(false; 0, 1, 2, 3, 4, 5, 6, 7);
                    j += 16;
                    index += 16;
                    assert_eq!(N, j);
                    assert_eq!((i_outer + i_inner + 1) * N, index as usize);

                    // j = 0;
                    // index -= N as isize;
                    // for _ in 0..N {
                    //     println!("{}", j);
                    //     assert_eq!(avx.mm512_extract_epi64::<0>(*unsafe { sum_new.get_unchecked_mut(galois_lookup_table[j]) }), avx.mm512_extract_epi64::<0>(avx.mm512_add_epi64(
                    //         avx.mm512_add_epi64(
                    //             unsafe { *sum.get_unchecked(j) },
                    //             avx.mm512_mul_epu32(Zqx8.uncompress(unsafe { *inner_prod_gk[0].get_unchecked(index as usize) }).to_raw(), Zqx8.uncompress(unsafe { *gk_b[0].get_unchecked(j) }).to_raw())
                    //         ), avx.mm512_add_epi64(
                    //             avx.mm512_mul_epu32(Zqx8.uncompress(unsafe { *inner_prod_gk[1].get_unchecked(index as usize) }).to_raw(), Zqx8.uncompress(unsafe { *gk_b[1].get_unchecked(j) }).to_raw()),
                    //             avx.mm512_mul_epu32(Zqx8.uncompress(unsafe { *transformed_entries.get_unchecked(index as usize) }).to_raw(), Zqx8.uncompress(unsafe { *qry_b.get_unchecked(j) }).to_raw())
                    //         )
                    //     )));
                    //     j += 1;
                    //     index += 1;
                    // }

                    swap(&mut sum, &mut sum_new);
                }
                MEM_LOAD_DATA.fetch_add((min(len - i_outer, reduce_every_iterations) * N * 3 * 32) as u64, Ordering::Relaxed);
                for j in 0..N {
                    sum[j] = Zqx8.reduce_from_u64(sum[j]).to_raw();
                }
            }
        }
    }
}

pub static MEM_LOAD_DATA: AtomicU64 = AtomicU64::new(0);

///
/// Allocates a zero-initialized backing buffer for [`PIRDatabase::create`]
/// in tests. The global allocator only guarantees the alignment of
/// `CompressedZnx8El` itself (32 B), but `create` requires the 64-B
/// alignment of `m512i` — so the buffer must come from [`AligningAlloc`].
///
#[cfg(test)]
pub(crate) fn aligned_test_memory(len: usize) -> Vec<CompressedZnx8El<26>, AligningAlloc> {
    let mut result = Vec::with_capacity_in(len, AligningAlloc::default());
    result.resize_with(len, CompressedZnx8El::default);
    return result;
}

#[cfg(test)]
use feanor_math::rings::finite::FiniteRingStore;
#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use rand::rngs::StdRng;
#[cfg(test)]
use rand::{RngCore, SeedableRng};
#[cfg(test)]
use crate::pir_wrapper;
#[cfg(test)]
use crate::double_pir;
#[cfg(test)]
use std::time::Instant;

#[test]
fn test_set_transformed_entries_full_group() {
    let log2_N = 3;
    let N = 1 << log2_N;
    let q = FIXED_Q as u64;
    let t = 1025;
    let C = CipherRing::new(Zn::new(q), N, [Zn::new(q).neg_one()]);
    let R = PlainRing::new(Zn::new(t), N, [Zn::new(t).neg_one()]);
    let Zt = R.base_ring();
    let factor = Zt.invert(&Zt.int_hom().map(8)).unwrap();

    let assert_correctly_preprocessed = |expected: &[[El<Zn>; 8]], db: &PIRDatabase| {
        assert_eq!(expected.len() * N, db.transformed_entries.0.len());
        let Zq1 = db.rns_base.0.component_ring();
        let zeta1 = Zq1.invert(&db.rns_base.0.get_components(db.zeta.0)[0]).unwrap();
        for (k, (expected, actual0)) in expected.iter().zip(db.transformed_entries.0.chunks(N)).enumerate() {
            let expected = expected.iter().map(|x| Zq1.coerce(&StaticRing::<i64>::RING, Zt.smallest_lift(Zt.mul(factor, *x)))).collect::<Vec<_>>();
            let expected_ntt = (0..N).map(|i| Zq1.sum((0..N).map(|j| Zq1.mul(expected[j], Zq1.pow(zeta1, (2 * i + 1) * j))))).collect::<Vec<_>>();
            for i in 0..N {
                assert!(Zq1.eq_el(&expected_ntt[i], &db.rns_base.0.get_components(db.rns_base.0.uncompress(actual0[i]))[0]),
                    "value mismatch in {}-th of transformed_entries; expected [{}] = {} but got {}", k, i, Zq1.format(&expected_ntt[i]), Zq1.format(&db.rns_base.0.get_components(db.rns_base.0.uncompress(actual0[i]))[0]));
            }
        }
    };

    let mut memory = aligned_test_memory(PIRDatabase::required_memory_general(IndexGroup::full_group(N), 1 << log2_N));
    let mut db = PIRDatabase::create(R, C, IndexGroup::full_group(N), UsedSeeds::FirstSet, &mut memory);

    db.set_transformed_entries((0..64).map(|i| if i == 0 { R.one() } else { R.zero() }));
    assert_correctly_preprocessed(&[[Zt.one(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero()]; 4], &db);

    db.set_transformed_entries((0..64).map(|i| if i == 8 { R.one() } else { R.zero() }));
    // because in the last step, we undo the automorphism g^j, and this will map \zeta^(-g^j) to \zeta^-1
    assert_correctly_preprocessed(&[[Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.neg_one()]; 4], &db);

    db.set_transformed_entries((0..64).map(|i| if i == 0 { R.canonical_gen() } else { R.zero() }));
    assert_correctly_preprocessed(&[
        [Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.one(), Zt.zero(), Zt.zero()],
        [Zt.zero(), Zt.neg_one(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero()],
        [Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.neg_one(), Zt.zero(), Zt.zero()],
        [Zt.zero(), Zt.one(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero()]
    ], &db);
}

#[test]
fn test_set_transformed_entries_full_g_subgroup() {
    let log2_N = 3;
    let N = 1 << log2_N;
    let q = FIXED_Q as u64;
    let t = 1025;
    let C = CipherRing::new(Zn::new(q), N, [Zn::new(q).neg_one()]);
    let R = PlainRing::new(Zn::new(t), N, [Zn::new(t).neg_one()]);
    let Zt = R.base_ring();
    let factor = Zt.invert(&Zt.int_hom().map(4)).unwrap();

    let assert_correctly_preprocessed = |expected: &[[El<Zn>; 8]], db: &PIRDatabase| {
        assert_eq!(expected.len() * N, db.transformed_entries.0.len());
        let Zq1 = db.rns_base.0.component_ring();
        let zeta1 = Zq1.invert(&db.rns_base.0.get_components(db.zeta.0)[0]).unwrap();
        for (k, (expected, actual0)) in expected.iter().zip(db.transformed_entries.0.chunks(N)).enumerate() {
            let expected = expected.iter().map(|x| Zq1.coerce(&StaticRing::<i64>::RING, Zt.smallest_lift(Zt.mul(factor, *x)))).collect::<Vec<_>>();
            let expected_ntt = (0..N).map(|i| Zq1.sum((0..N).map(|j| Zq1.mul(expected[j], Zq1.pow(zeta1, (2 * i + 1) * j))))).collect::<Vec<_>>();
            for i in 0..N {
                assert!(Zq1.eq_el(&expected_ntt[i], &db.rns_base.0.get_components(db.rns_base.0.uncompress(actual0[i]))[0]),
                    "value mismatch in {}-th of transformed_entries; expected [{}] = {} but got {}", k, i, Zq1.format(&expected_ntt[i]), Zq1.format(&db.rns_base.0.get_components(db.rns_base.0.uncompress(actual0[i]))[0]));
            }
        }
    };

    let index_group = IndexGroup::subgroup_of_size(N, 4);
    let mut memory = aligned_test_memory(PIRDatabase::required_memory_general(index_group, 1 << log2_N));
    let mut db = PIRDatabase::create(R, C, index_group, UsedSeeds::FirstSet, &mut memory);

    db.set_transformed_entries((0..32).map(|i| if i == 0 { R.one() } else { R.zero() }));
    assert_correctly_preprocessed(&[[Zt.one(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero()]; 4], &db);

    db.set_transformed_entries((0..32).map(|i| if i == 8 { R.one() } else { R.zero() }));
    assert_correctly_preprocessed(&[[Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.neg_one()]; 4], &db);

    db.set_transformed_entries((0..32).map(|i| if i == 0 { R.canonical_gen() } else { R.zero() }));
    assert_correctly_preprocessed(&[
        [Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.one(), Zt.zero(), Zt.zero()],
        [Zt.zero(), Zt.neg_one(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero()],
        [Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.neg_one(), Zt.zero(), Zt.zero()],
        [Zt.zero(), Zt.one(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero()]
    ], &db);
}

#[test]
fn test_set_transformed_entries_small_subgroup() {
    let log2_N = 3;
    let N = 1 << log2_N;
    let q = FIXED_Q as u64;
    let t = 1025;
    let C = CipherRing::new(Zn::new(q), N, [Zn::new(q).neg_one()]);
    let R = PlainRing::new(Zn::new(t), N, [Zn::new(t).neg_one()]);
    let Zt = R.base_ring();
    let factor = Zt.invert(&Zt.int_hom().map(2)).unwrap();

    let assert_correctly_preprocessed = |expected: &[[El<Zn>; 8]], db: &PIRDatabase| {
        assert_eq!(expected.len() * N, db.transformed_entries.0.len());
        let Zq1 = db.rns_base.0.component_ring();
        let zeta1 = Zq1.invert(&db.rns_base.0.get_components(db.zeta.0)[0]).unwrap();
        for (k, (expected, actual0)) in expected.iter().zip(db.transformed_entries.0.chunks(N)).enumerate() {
            let expected = expected.iter().map(|x| Zq1.coerce(&StaticRing::<i64>::RING, Zt.smallest_lift(Zt.mul(factor, *x)))).collect::<Vec<_>>();
            let expected_ntt = (0..N).map(|i| Zq1.sum((0..N).map(|j| Zq1.mul(expected[j], Zq1.pow(zeta1, (2 * i + 1) * j))))).collect::<Vec<_>>();
            for i in 0..N {
                assert!(Zq1.eq_el(&expected_ntt[i], &db.rns_base.0.get_components(db.rns_base.0.uncompress(actual0[i]))[0]),
                    "value mismatch in {}-th of transformed_entries; expected [{}] = {} but got {}", k, i, Zq1.format(&expected_ntt[i]), Zq1.format(&db.rns_base.0.get_components(db.rns_base.0.uncompress(actual0[i]))[0]));
            }
        }
    };

    let index_group = IndexGroup::subgroup_of_size(N, 2);
    let mut memory = aligned_test_memory(PIRDatabase::required_memory_general(index_group, 1 << log2_N));
    let mut db = PIRDatabase::create(R, C, index_group, UsedSeeds::FirstSet, &mut memory);

    db.set_transformed_entries((0..16).map(|i| if i == 0 { R.one() } else { R.zero() }));
    assert_correctly_preprocessed(&[[Zt.one(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero()]; 2], &db);

    println!("zeta mod {} = {}", db.rns_base.0.component_ring().modulus(), db.rns_base.0.component_ring().format(&db.rns_base.0.get_components(db.zeta.0)[0]));

    db.set_transformed_entries((0..16).map(|i| if i == 8 { R.one() } else { R.zero() }));
    assert_correctly_preprocessed(&[[Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.neg_one(), Zt.zero()]; 2], &db);

    db.set_transformed_entries((0..16).map(|i| if i == 0 { R.canonical_gen() } else { R.zero() }));
    assert_correctly_preprocessed(&[
        [Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.neg_one(), Zt.zero(), Zt.zero()],
        [Zt.zero(), Zt.one(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero(), Zt.zero()],
    ], &db);
}

#[test]
fn test_base_pir_single() {
    let mut rng = StdRng::from_seed([0; 32]);

    let log2_N = 8;
    let N = 1 << log2_N;
    let q = FIXED_Q as u64;
    let sk_hwt = pir_wrapper::SK_HWT;
    let t = double_pir::PRIMARY_PLAIN_MODULUS as u64;
    let sigma = pir_wrapper::SIGMA;
    
    let C = CipherRing::new(Zn::new(q), N, [Zn::new(q).neg_one()]);
    let R = PlainRing::new(Zn::new(t), N, [Zn::new(t).neg_one()]);
    let mut memory = aligned_test_memory(PIRDatabase::required_memory_general(IndexGroup::full_group(N), 1 << log2_N));
    let mut db = PIRDatabase::create(R, C, IndexGroup::full_group(N), UsedSeeds::FirstSet, &mut memory);

    let data = (0..(N * SIMD_COUNT)).map(|i| R.int_hom().map(i as i32)).collect::<Vec<_>>();
    let start = Instant::now();
    db.set_db(data.iter().map(|x| R.clone_el(x)));
    let end = Instant::now();
    println!("Preprocessing done in {} us", (end - start).as_micros());
    let R = db.entry_ring();
    let C = db.ciphertext_ring();

    let sk = gen_sk(&C, &mut rng, sk_hwt);
    let gk_b = gen_gk_b(&C, &UsedSeeds::FirstSet.get_gk_seeds(), &mut rng, &sk, 5, GK_DIGITS, LOG2_B, sigma);
    let gk_b = from_fn(|i| &gk_b[i]);

    for idx in 0..32 {
        let qry_b = enc_sym_b(&R, &C, UsedSeeds::FirstSet.get_qry_seeds()[0], &mut rng, &sk, &R.pow(R.canonical_gen(), idx), sigma);
        let qry_conj_b = enc_sym_b(&R, &C, UsedSeeds::FirstSet.get_qry_seeds()[1], &mut rng, &sk, &R.pow(R.canonical_gen(), 2 * R.rank() - idx), sigma);
        let start = Instant::now();
        let [ra, _, _, _, _, _, _, _] = db.get_a();
        let [rb, _, _, _, _, _, _, _] = db.perform_pir(&qry_b, Some(&qry_conj_b), gk_b);
        let end = Instant::now();
        println!("Query done in {} us", (end - start).as_micros());
        let result = dec(&R, &C, &sk, &(ra, rb));
        assert_el_eq!(&R, &data[idx * SIMD_COUNT], result);
    }
}

#[test]
fn test_base_pir_nontrivial_quotient() {
    let mut rng = StdRng::from_seed([0; 32]);

    let log2_N = 8;
    let N = 1 << log2_N;
    let q = FIXED_Q as u64;
    let sk_hwt = pir_wrapper::SK_HWT;
    let t = double_pir::PRIMARY_PLAIN_MODULUS as u64;
    let sigma = pir_wrapper::SIGMA;
    
    let C = CipherRing::new(Zn::new(q), N, [Zn::new(q).neg_one()]);
    let R = PlainRing::new(Zn::new(t), N, [Zn::new(t).neg_one()]);
    let index_group = IndexGroup::subgroup_of_size(N, 32);
    let mut memory = aligned_test_memory(PIRDatabase::required_memory_general(index_group, 1 << log2_N));
    let mut db = PIRDatabase::create(R, C, index_group, UsedSeeds::FirstSet, &mut memory);

    let data = (0..(index_group.group_order() * SIMD_COUNT)).map(|i| R.int_hom().map(i as i32)).collect::<Vec<_>>();
    let start = Instant::now();
    db.set_db(data.iter().map(|x| R.clone_el(x)));
    let end = Instant::now();
    println!("Preprocessing done in {} us", (end - start).as_micros());
    let R = db.entry_ring();
    let C = db.ciphertext_ring();

    let sk = gen_sk(&C, &mut rng, sk_hwt);
    let gk_b = gen_gk_b(&C, &UsedSeeds::FirstSet.get_gk_seeds(), &mut rng, &sk, 5, GK_DIGITS, LOG2_B, sigma);
    let gk_b = from_fn(|i| &gk_b[i]);

    for idx in 0..32 {
        let qry_b_payload = R.pow(R.canonical_gen(), idx * index_group.query_base_power());
        let qry_b = enc_sym_b(&R, &C, UsedSeeds::FirstSet.get_qry_seeds()[0], &mut rng, &sk, &qry_b_payload, sigma);
        let start = Instant::now();
        let [ra, _, _, _, _, _, _, _] = db.get_a();
        let [rb, _, _, _, _, _, _, _] = db.perform_pir(&qry_b, None, gk_b);
        let end = Instant::now();
        println!("Query done in {} us", (end - start).as_micros());
        let result = dec(&R, &C, &sk, &(ra, rb));
        assert_el_eq!(&R, &data[idx as usize * SIMD_COUNT], result);
    }
}

#[test]
fn test_base_pir_split() {
    let mut rng = StdRng::from_seed([0; 32]);

    let log2_N = 8;
    let N = 1 << log2_N;
    let q = FIXED_Q as u64;
    let sk_hwt = pir_wrapper::SK_HWT;
    let t = double_pir::PRIMARY_PLAIN_MODULUS as u64;
    let sigma = pir_wrapper::SIGMA;
    
    let C = CipherRing::new(Zn::new(q), N, [Zn::new(q).neg_one()]);
    let R = PlainRing::new(Zn::new(t), N, [Zn::new(t).neg_one()]);
    let mut memory = aligned_test_memory(PIRDatabase::required_memory_general(IndexGroup::full_group(N), 1 << log2_N));
    let mut db = PIRDatabase::create(R, C, IndexGroup::full_group(N), UsedSeeds::FirstSet, &mut memory);

    let data = (0..(N * SIMD_COUNT)).map(|_| R.random_element(|| rng.next_u64())).collect::<Vec<_>>();
    let start = Instant::now();
    db.set_db(data.iter().map(|x| R.clone_el(x)));
    let end = Instant::now();
    println!("Preprocessing done in {} us", (end - start).as_micros());
    let R = db.entry_ring();
    let C = db.ciphertext_ring();

    let sks: [_; SIMD_COUNT] = from_fn(|_| gen_sk(&C, &mut rng, sk_hwt));
    let gk_bs: [_; SIMD_COUNT] = from_fn(|i| gen_gk_b(&C, &UsedSeeds::FirstSet.get_gk_seeds(), &mut rng, &sks[i], 5, GK_DIGITS, LOG2_B, sigma));
    let gk_bs: [[_; GK_DIGITS]; SIMD_COUNT] = from_fn(|j| from_fn(|i| &gk_bs[j][i]));

    for idx in 0..32 {
        let qry_bs: [_; SIMD_COUNT] = from_fn(|i| enc_sym_b(&R, &C, UsedSeeds::FirstSet.get_qry_seeds()[0], &mut rng, &sks[i], &R.pow(R.canonical_gen(), idx), sigma));
        let qry_conj_bs: [_; SIMD_COUNT] = from_fn(|i| enc_sym_b(&R, &C, UsedSeeds::FirstSet.get_qry_seeds()[1], &mut rng, &sks[i], &R.pow(R.canonical_gen(), 2 * R.rank() - idx), sigma));
        let start = Instant::now();
        let result_as = db.get_a();
        let result_bs = db.perform_pir_split(from_fn(|i| QueryRef { qry_b: &qry_bs[i], qry_conj_b: Some(&qry_conj_bs[i]), gk_b: gk_bs[i] }));
        let end = Instant::now();
        println!("Queries done in {} us", (end - start).as_micros());
        for (i, (ra, rb)) in result_as.into_iter().zip(result_bs.into_iter()).enumerate() {
            let result = dec(&R, &C, &sks[i], &(ra, rb));
            assert_el_eq!(&R, &data[idx * SIMD_COUNT + i], result);
        }
    }
}