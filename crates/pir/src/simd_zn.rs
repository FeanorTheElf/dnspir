//! Arithmetic in `Z/pZ` for `p = 2^K - alpha`, eight elements at a time.
//!
//! [`SpecialZnx8`] implements `feanor-math`'s ring traits on top of the
//! 8-lane vectors from [`crate::avx`], keeping representatives lazily
//! reduced (bounded by `2p`) so that the hot loops can skip a conditional
//! subtraction. [`CompletelyReducedSpecialZnx8`] is the same ring with
//! fully reduced representatives, which is what the NTT butterflies need.
//! [`CompressedZnx8El`] is the compact 32-bit-per-lane form the
//! preprocessed databases are stored in.
//!
//! The two moduli actually instantiated are the RNS factors `Q_FACTOR1`
//! (`K = 26`) and `Q_FACTOR2` (`K = 25`) from [`crate::base_pir`].

use feanor_math::algorithms::fft::cooley_tuckey::CooleyTuckeyButterfly;
use feanor_math::delegate::DelegateRing;
use feanor_math::divisibility::{DivisibilityRing, DivisibilityRingStore};
use feanor_math::homomorphism::Homomorphism;
use feanor_math::integer::IntegerRing;
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::*;
use feanor_math::rings::zn::zn_64::{Zn, ZnEl};
use feanor_math::rings::zn::ZnRingStore;

use crate::avx;

///
/// Arithmetic modulo `p = 2^K - alpha`, performed in SIMD fashion 
/// on 8 elements in parallel. `p` does not have to be prime, but this is a
/// common case.
/// 
#[derive(Clone)]
pub struct SpecialZnx8<const K: u32> {
    avx: avx::Context,
    alpha: i64,
    component_ring: Zn,
    // p in each entry of a 8x64
    modulusx8: avx::m512i,
    // 2 p in each entry of a 8x64
    double_modulusx8: avx::m512i,
    // reduce_from_u64_its: usize
}

#[derive(Copy, Clone)]
pub struct SpecialZnx8El<const K: u32> {
    // value should be a representative `<= 2 * p`
    value: avx::m512i
}

#[derive(Copy, Clone, Default)]
#[repr(C, align(32))]
pub struct CompressedZnx8El<const K: u32> {
    value: [i32; 8]
}

unsafe impl<const K: u32> bytemuck::Zeroable for CompressedZnx8El<K> {}
unsafe impl<const K: u32> bytemuck::Pod for CompressedZnx8El<K> {}

impl<const K: u32> CompressedZnx8El<K> {
    
    ///
    /// use with care!
    /// 
    #[allow(unused)]
    pub fn from_raw(value: avx::m256i) -> Self {
        Self { value: unsafe { std::mem::transmute(value) } }
    }
}

impl<const K: u32> SpecialZnx8El<K> {
    ///
    /// use with care!
    /// 
    pub fn to_raw(self) -> avx::m512i {
        self.value
    }

    ///
    /// use with care!
    /// 
    #[allow(unused)]
    pub fn from_raw(value: avx::m512i) -> Self {
        Self { value }
    }
}

impl<const K: u32> SpecialZnx8<K> {

    pub fn new(alpha: i64) -> Self {
        let avx = avx::Context::check_target_features();
        let modulus = (1 << K as usize) - alpha;
        let result = Self {
            alpha: alpha,
            component_ring: Zn::new(modulus as u64),
            modulusx8: avx.mm512_set1_epi64(modulus),
            double_modulusx8: avx.mm512_set1_epi64(2 * modulus),
            avx: avx
        };
        result.check_invariant();
        return result;
    }

    pub fn component_ring(&self) -> &Zn {
        &self.component_ring
    }

    pub fn get_components(&self, val: SpecialZnx8El<K>) -> [ZnEl; 8] {
        let hom = self.component_ring.int_hom();
        self.compress(val).value.map(|x| hom.map(x))
    }

    pub fn from_components(&self, values: [ZnEl; 8]) -> SpecialZnx8El<K> {
        SpecialZnx8El { value: self.avx.mm512_setr_epi64(
            self.component_ring.smallest_positive_lift(values[0]),
            self.component_ring.smallest_positive_lift(values[1]),
            self.component_ring.smallest_positive_lift(values[2]),
            self.component_ring.smallest_positive_lift(values[3]),
            self.component_ring.smallest_positive_lift(values[4]),
            self.component_ring.smallest_positive_lift(values[5]),
            self.component_ring.smallest_positive_lift(values[6]),
            self.component_ring.smallest_positive_lift(values[7])
        ) }
    }

    pub fn compress(&self, val: SpecialZnx8El<K>) -> CompressedZnx8El<K> {
        CompressedZnx8El {
            value: [
                self.avx.mm512_extract_epi64::<0>(val.value) as i32,
                self.avx.mm512_extract_epi64::<1>(val.value) as i32,
                self.avx.mm512_extract_epi64::<2>(val.value) as i32,
                self.avx.mm512_extract_epi64::<3>(val.value) as i32,
                self.avx.mm512_extract_epi64::<4>(val.value) as i32,
                self.avx.mm512_extract_epi64::<5>(val.value) as i32,
                self.avx.mm512_extract_epi64::<6>(val.value) as i32,
                self.avx.mm512_extract_epi64::<7>(val.value) as i32
            ]
        }
    }

    pub fn uncompress(&self, val: CompressedZnx8El<K>) -> SpecialZnx8El<K> {
        SpecialZnx8El { value: self.avx.mm512_setr_epi64(
            val.value[0] as i64,
            val.value[1] as i64,
            val.value[2] as i64,
            val.value[3] as i64,
            val.value[4] as i64,
            val.value[5] as i64,
            val.value[6] as i64,
            val.value[7] as i64
        ) }
    }

    pub fn modulus(&self) -> &i64 {
        self.component_ring().modulus()
    }

    pub fn modulus_avx(&self) -> avx::m512i {
        self.modulusx8
    }

    pub fn repr_bound(&self) -> i64 {
        *self.component_ring().modulus() * 2
    }

    fn check_invariant(&self) {
        assert!(self.alpha >= 0);
        assert!((2 << K as usize) <= u32::MAX as i64);
        assert!(2 * self.alpha + 2 * self.alpha * self.alpha <= (1 << K as usize));
        assert!(2 * self.alpha < *self.component_ring.modulus());
    }

    #[inline]
    fn check_repr(&self, _value: avx::m512i) {
        // assert!(self.avx.mm512_extract_epi64::<0>(value) >= 0);
        // assert!(self.avx.mm512_extract_epi64::<1>(value) >= 0);
        // assert!(self.avx.mm512_extract_epi64::<2>(value) >= 0);
        // assert!(self.avx.mm512_extract_epi64::<3>(value) >= 0);
        // assert!(self.avx.mm512_extract_epi64::<4>(value) >= 0);
        // assert!(self.avx.mm512_extract_epi64::<5>(value) >= 0);
        // assert!(self.avx.mm512_extract_epi64::<6>(value) >= 0);
        // assert!(self.avx.mm512_extract_epi64::<7>(value) >= 0);

        // assert!(self.avx.mm512_extract_epi64::<0>(value) <= self.repr_bound());
        // assert!(self.avx.mm512_extract_epi64::<1>(value) <= self.repr_bound());
        // assert!(self.avx.mm512_extract_epi64::<2>(value) <= self.repr_bound());
        // assert!(self.avx.mm512_extract_epi64::<3>(value) <= self.repr_bound());
        // assert!(self.avx.mm512_extract_epi64::<4>(value) <= self.repr_bound());
        // assert!(self.avx.mm512_extract_epi64::<5>(value) <= self.repr_bound());
        // assert!(self.avx.mm512_extract_epi64::<6>(value) <= self.repr_bound());
        // assert!(self.avx.mm512_extract_epi64::<7>(value) <= self.repr_bound());
    }

    ///
    /// The general relation is that if the input is `x`, the output
    /// is less than `alpha * x / 2^K + 2^K - alpha` and congruent to
    /// the input modulo `2^K - alpha`.
    /// 
    fn reduce_step(&self, data: avx::m512i) -> avx::m512i {
        let approx_quo = self.avx.mm512_srli_epi64::<K>(data);
        return self.avx.mm512_sub_epi64(data, self.avx.mm512_mul_epu32(approx_quo, self.modulusx8));
    }

    ///
    /// Reduces each of the packed u64 integers to an integer `<= 2 * p`
    /// and congruent to the input.
    /// 
    pub fn reduce_from_u64(&self, data: avx::m512i) -> SpecialZnx8El<K> {
        let hom = self.component_ring().can_hom(&StaticRing::<i128>::RING).unwrap();
        self.from_components([
            hom.map(self.avx.mm512_extract_epi64::<0>(data) as u64 as i128),
            hom.map(self.avx.mm512_extract_epi64::<1>(data) as u64 as i128),
            hom.map(self.avx.mm512_extract_epi64::<2>(data) as u64 as i128),
            hom.map(self.avx.mm512_extract_epi64::<3>(data) as u64 as i128),
            hom.map(self.avx.mm512_extract_epi64::<4>(data) as u64 as i128),
            hom.map(self.avx.mm512_extract_epi64::<5>(data) as u64 as i128),
            hom.map(self.avx.mm512_extract_epi64::<6>(data) as u64 as i128),
            hom.map(self.avx.mm512_extract_epi64::<7>(data) as u64 as i128)
        ])
    }

    ///
    /// Takes a packed integers `<= 2^(2 * K + 1)` and returns results congruent
    /// to the inputs modulo `p`, and bounded as `<= p`.
    /// 
    fn reduce_after_mul(&self, data: avx::m512i) -> avx::m512i {
        // `self.alpha + self.alpha * self.alpha <= 2 * (1 << K as usize)` implies
        // that the second reduction ends up with values `<= 2 * (2^K - alpha)`
        let reduced1 = self.reduce_step(data);
        return self.reduce_step(reduced1);
    }

    ///
    /// Takes packed integers `<= 2 * p` and returns a full reduction
    /// modulo `p`, i.e. a representative in `<= p`
    /// 
    fn reduce_from_2p(&self, data: avx::m512i) -> avx::m512i {
        let to_reduce = self.avx.mm512_cmpgt_epi64_mask(data, self.modulusx8);
        let subtract = self.avx.mm512_mask(self.modulusx8, to_reduce);
        return self.avx.mm512_sub_epi64(data, subtract);
    }

    ///
    /// Takes packed integers `<= 4 * p` and integers `<= 2 * p` congruent
    /// to the input modulo `p`.
    /// 
    fn reduce_from_4p(&self, data: avx::m512i) -> avx::m512i {
        let to_reduce = self.avx.mm512_cmpgt_epi64_mask(data, self.double_modulusx8);
        let subtract = self.avx.mm512_mask(self.double_modulusx8, to_reduce);
        return self.avx.mm512_sub_epi64(data, subtract);
    }

    pub fn simd_smallest_positive_lift(&self, el: SpecialZnx8El<K>) -> avx::m512i {
        let result = self.reduce_from_2p(el.value);
        // return result;
        return self.avx.mm512_sub_epi64(result, self.avx.mm512_mask(self.modulusx8, self.avx.mm512_cmpeq_epi64_mask(result, self.modulusx8)));
    }

    pub fn simd_from_p_sqr(&self, x: avx::m512i) -> SpecialZnx8El<K> {
        let result = self.reduce_after_mul(x);
        self.check_repr(result);
        SpecialZnx8El { value: result }
    }
}

impl<const K: u32> PartialEq for SpecialZnx8<K> {

    fn eq(&self, other: &Self) -> bool {
        self.component_ring.get_ring() == other.component_ring.get_ring()
    }
}

impl<const K: u32> RingBase for SpecialZnx8<K> {
    
    type Element = SpecialZnx8El<K>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        *val
    }
    
    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.check_repr(lhs.value);
        self.check_repr(rhs.value);
        let sum = self.avx.mm512_add_epi64(lhs.value, rhs.value);
        lhs.value = self.reduce_from_4p(sum);
        self.check_repr(lhs.value);
    }

    fn sub_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.check_repr(lhs.value);
        self.check_repr(rhs.value);
        let increased = self.avx.mm512_add_epi64(lhs.value, self.double_modulusx8);
        let diff = self.avx.mm512_sub_epi64(increased, rhs.value);
        lhs.value = self.reduce_from_4p(diff);
        self.check_repr(lhs.value);
    }

    fn negate_inplace(&self, lhs: &mut Self::Element) {
        self.check_repr(lhs.value);
        lhs.value = self.avx.mm512_sub_epi64(self.double_modulusx8, lhs.value);
        self.check_repr(lhs.value);
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.check_repr(lhs.value);
        self.check_repr(rhs.value);
        // unfortunately, reduce_after_mul() needs input to be bounded by `2 * p^2`, and
        // not `4 * p^2` which would be `repr_bound()^2`
        let prod = self.avx.mm512_mul_epu32(self.reduce_from_2p(lhs.value), rhs.value);
        let result = self.reduce_after_mul(prod);
        self.check_repr(result);
        lhs.value = result;
    }

    fn fma(&self, lhs: &Self::Element, rhs: &Self::Element, summand: Self::Element) -> Self::Element {
        self.check_repr(lhs.value);
        self.check_repr(rhs.value);
        self.check_repr(summand.value);
        // unfortunately, reduce_after_mul() needs input to be bounded by `2 * p^2`, and
        // not `4 * p^2` which would be `repr_bound()^2`
        let prod = self.avx.mm512_add_epi64(self.avx.mm512_mul_epu32(self.reduce_from_2p(lhs.value), rhs.value), summand.value);
        let result = self.reduce_after_mul(prod);
        self.check_repr(result);
        SpecialZnx8El { value: result }
    }

    fn from_int(&self, value: i32) -> Self::Element {
        let value = self.component_ring.smallest_positive_lift(self.component_ring.get_ring().from_int(value));
        SpecialZnx8El { value: self.avx.mm512_set1_epi64(value) }
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        // `reduce_from_2p()` necessary twice since `2p` is an allowed representative
        let diff = self.reduce_from_2p(self.reduce_from_2p(self.sub_ref(lhs, rhs).value));
        return self.get_components(SpecialZnx8El { value: diff }).iter().all(|x| self.component_ring.is_zero(x));
    }

    fn is_commutative(&self) -> bool { true }
    fn is_noetherian(&self) -> bool { true }
    fn is_approximate(&self) -> bool { false }

    fn dbg_within<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>, _env: EnvBindingStrength) -> std::fmt::Result {
        // `reduce_from_2p()` necessary twice since `2p` is an allowed representative
        let values = self.get_components(SpecialZnx8El { value: self.reduce_from_2p(self.reduce_from_2p(value.value)) });
        // write!(out, "({}, {}, {}, {}, {}, {}, {}, {})",
        //     self.component_ring.format(&values[0]),
        //     self.component_ring.format(&values[1]),
        //     self.component_ring.format(&values[2]),
        //     self.component_ring.format(&values[3]),
        //     self.component_ring.format(&values[4]),
        //     self.component_ring.format(&values[5]),
        //     self.component_ring.format(&values[6]),
        //     self.component_ring.format(&values[7])
        // )
        write!(out, "{}",
            self.component_ring.format(&values[0])
        )
    }

    fn sum<I>(&self, els: I) -> Self::Element 
        where I: IntoIterator<Item = Self::Element>
    {
        let mut result = self.zero();
        let summands_before_reduction = *self.modulus() as usize;
        assert!(summands_before_reduction >= 1);
        let mut i = 0;
        let mut it = els.into_iter();
        while let Some(x) = it.next() {
            if i == summands_before_reduction {
                i = 0;
                result.value = self.reduce_after_mul(result.value);
            }
            result.value = self.avx.mm512_add_epi64(result.value, x.value);
            i += 1;
        }
        result.value = self.reduce_after_mul(result.value);
        return result;
    }

    fn characteristic<I: RingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.component_ring.get_ring().characteristic(ZZ)
    }
}

impl<const K: u32> DivisibilityRing for SpecialZnx8<K> {

    fn checked_left_div(&self, lhs: &Self::Element, rhs: &Self::Element) -> Option<Self::Element> {
        let lhs = self.get_components(SpecialZnx8El { value: self.reduce_from_2p(lhs.value) });
        let rhs = self.get_components(SpecialZnx8El { value: self.reduce_from_2p(rhs.value) });
        return Some(self.from_components([
            self.component_ring().checked_div(&lhs[0], &rhs[0])?,
            self.component_ring().checked_div(&lhs[1], &rhs[1])?,
            self.component_ring().checked_div(&lhs[2], &rhs[2])?,
            self.component_ring().checked_div(&lhs[3], &rhs[3])?,
            self.component_ring().checked_div(&lhs[4], &rhs[4])?,
            self.component_ring().checked_div(&lhs[5], &rhs[5])?,
            self.component_ring().checked_div(&lhs[6], &rhs[6])?,
            self.component_ring().checked_div(&lhs[7], &rhs[7])?,
        ]).into());
    }
}

#[repr(transparent)]
#[derive(Clone)]
pub struct CompletelyReducedSpecialZnx8<const K: u32> {
    base: SpecialZnx8<K>
}

impl<const K: u32> CompletelyReducedSpecialZnx8<K> {

    pub fn from(base: SpecialZnx8<K>) -> Self {
        Self { base }
    }
}

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct CompletelyReducedSpecialZnx8El<const K: u32> {
    // representative should be `<= p`
    value: SpecialZnx8El<K>
}

impl<const K: u32> PartialEq for CompletelyReducedSpecialZnx8<K> {
    fn eq(&self, other: &Self) -> bool {
        self.base == other.base
    }
}

impl<const K: u32> DelegateRing for CompletelyReducedSpecialZnx8<K> {

    type Base = SpecialZnx8<K>;
    type Element = CompletelyReducedSpecialZnx8El<K>;
    
    fn get_delegate(&self) -> &Self::Base {
        &self.base
    }

    fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element {
        el.value
    }

    fn delegate_mut<'a>(&self, el: &'a mut Self::Element) -> &'a mut <Self::Base as RingBase>::Element {
        &mut el.value
    }

    fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element {
        &el.value
    }

    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element {
        CompletelyReducedSpecialZnx8El { value: SpecialZnx8El { value: self.base.reduce_from_2p(el.value) } }
    }
}

impl<const K: u32, const L: u32> CooleyTuckeyButterfly<CompletelyReducedSpecialZnx8<L>> for SpecialZnx8<K> {

    #[inline(always)]
    fn butterfly_new<H: Homomorphism<CompletelyReducedSpecialZnx8<L>, Self>>(hom: H, x: &mut Self::Element, y: &mut Self::Element, twiddle: &CompletelyReducedSpecialZnx8El<L>) {
        assert_eq!(K, L);
        let self_ = hom.codomain().get_ring();
        self_.check_repr(x.value);
        self_.check_repr(y.value);
        let twiddled_y = self_.reduce_after_mul(self_.avx.mm512_mul_epu32(twiddle.value.value, y.value));
        y.value = self_.reduce_from_4p(self_.avx.mm512_sub_epi64(self_.avx.mm512_add_epi64(x.value, self_.double_modulusx8), twiddled_y));
        x.value = self_.reduce_from_4p(self_.avx.mm512_add_epi64(x.value, twiddled_y));
        self_.check_repr(x.value);
        self_.check_repr(y.value);
    }
    
    #[inline(always)]
    fn inv_butterfly_new<H: Homomorphism<CompletelyReducedSpecialZnx8<L>, Self>>(hom: H, x: &mut Self::Element, y: &mut Self::Element, twiddle: &CompletelyReducedSpecialZnx8El<L>) {
        assert_eq!(K, L);
        let self_ = hom.codomain().get_ring();
        self_.check_repr(x.value);
        self_.check_repr(y.value);
        let x_new = self_.reduce_from_4p(self_.avx.mm512_add_epi64(x.value, y.value));
        y.value = self_.reduce_after_mul(self_.avx.mm512_mul_epu32(twiddle.value.value, self_.reduce_from_4p(self_.avx.mm512_sub_epi64(self_.avx.mm512_add_epi64(x.value, self_.double_modulusx8), y.value))));
        x.value = x_new;
        self_.check_repr(x.value);
        self_.check_repr(y.value);
    }

    #[inline(always)]
    default fn prepare_for_fft(&self, _value: &mut Self::Element) {}
    
    #[inline(always)]
    default fn prepare_for_inv_fft(&self, _value: &mut Self::Element) {}
}

#[ignore]
#[test]
fn test_reduce() {
    const K: u32 = 26;
    let ring: SpecialZnx8<K> = SpecialZnx8::new(1 << 12);

    for x in (0..=(1 << (2 * K + 1))).rev().step_by(8) {
        let input = ring.avx.mm512_setr_epi64(x, x + 1, x + 2, x + 3, x + 4, x + 5, x + 6, x + 7);
        let output = ring.reduce_after_mul(input);
        let extracted = [
            ring.avx.mm512_extract_epi64::<0>(output) as i64,
            ring.avx.mm512_extract_epi64::<1>(output) as i64,
            ring.avx.mm512_extract_epi64::<2>(output) as i64,
            ring.avx.mm512_extract_epi64::<3>(output) as i64,
            ring.avx.mm512_extract_epi64::<4>(output) as i64,
            ring.avx.mm512_extract_epi64::<5>(output) as i64,
            ring.avx.mm512_extract_epi64::<6>(output) as i64,
            ring.avx.mm512_extract_epi64::<7>(output) as i64,
        ];
        for i in 0..4 {
            if !(
                (x + i) % *ring.modulus() == extracted[i as usize] ||
                (x + i) % *ring.modulus() + *ring.modulus() == extracted[i as usize] ||
                (2 * *ring.modulus() == extracted[i as usize] && (x + i) % *ring.modulus() == 0)
            ) {
                println!("{}, {}, {}", x, i, extracted[i as usize]);
                unreachable!();
            }
        }
        let output_full = ring.reduce_from_2p(output);
        let extracted = [
            ring.avx.mm512_extract_epi64::<0>(output_full) as i64,
            ring.avx.mm512_extract_epi64::<1>(output_full) as i64,
            ring.avx.mm512_extract_epi64::<2>(output_full) as i64,
            ring.avx.mm512_extract_epi64::<3>(output_full) as i64,
            ring.avx.mm512_extract_epi64::<4>(output_full) as i64,
            ring.avx.mm512_extract_epi64::<5>(output_full) as i64,
            ring.avx.mm512_extract_epi64::<6>(output_full) as i64,
            ring.avx.mm512_extract_epi64::<7>(output_full) as i64,
        ];
        for i in 0..4 {
            if !(
                (x + i) % *ring.modulus() == extracted[i as usize] ||
                (*ring.modulus() == extracted[i as usize] && (x + i) % *ring.modulus() == 0)
            ) {
                println!("{}, {}, {}", x, i, extracted[i as usize]);
                unreachable!();
            }
        }
        if x % 1000000000 == 0 {
            println!("done {}", x);
        }
    }
}