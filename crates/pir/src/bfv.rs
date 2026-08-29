//! The BFV primitives the PIR scheme is built on.
//!
//! Everything here works over a power-of-two cyclotomic ring
//! `Z_q[X] / (X^N + 1)`, represented as `feanor-math`'s `FreeAlgebraImpl`
//! ([`CipherRing`] for ciphertexts, [`PlainRing`] for plaintexts).
//!
//! The module provides secret-key generation with a fixed Hamming weight
//! ([`gen_sk`]), symmetric encryption and decryption ([`enc_sym_b`],
//! [`dec`]), Galois-key generation ([`gen_gk_b`]), and the modulus switch
//! ([`mod_switch_encode`], [`mod_switch_decode`]) that shrinks ciphertexts
//! down to the bit width the wire format uses.
//!
//! Ciphertexts are transmitted as their `b`-part only: the `a`-part is
//! regenerated from a 32-byte [`Seed`] by [`expand`] on both sides.

use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::integer::IntegerRingStore;
use feanor_math::primitive_int::StaticRing;
use feanor_math::rings::extension::extension_impl::FreeAlgebraImpl;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::rings::finite::FiniteRingStore;
use feanor_math::rings::zn::ZnRingStore;
use feanor_math::seq::VectorFn;
use rand::rngs::StdRng;
use rand::{CryptoRng, Rng, RngCore, SeedableRng};
use feanor_math::ring::*;
use feanor_math::rings::zn::zn_64::Zn;
use rand_distr::{Distribution, StandardNormal};
use tracing::instrument;

pub type Seed = [u8; 32];
pub type CipherRing = FreeAlgebraImpl<Zn, [El<Zn>; 1]>;
pub type PlainRing = FreeAlgebraImpl<Zn, [El<Zn>; 1]>;

#[instrument(skip_all)]
pub fn gen_sk<R: Rng + CryptoRng>(C: &CipherRing, mut rng: R, hwt: usize) -> El<CipherRing> {
    let mut result_data = (0..C.rank()).map(|_| C.base_ring().zero()).collect::<Vec<_>>();
    for _ in 0..hwt {
        let idx = (rng.next_u64() % C.rank() as u64) as usize;
        result_data[idx] = C.base_ring().int_hom().map((rng.next_u32() % 3) as i32 - 1);
    }
    return C.from_canonical_basis(result_data);
}

pub fn expand(C: &CipherRing, seed: [u8; 32]) -> El<CipherRing> {
    let mut rng = StdRng::from_seed(seed);
    C.random_element(|| rng.next_u64())
}

#[instrument(skip_all)]
pub fn mod_switch_encode<const BITS: usize>(C: &CipherRing, x: &El<CipherRing>) -> Vec<u64> {
    assert!(BITS <= u64::BITS as usize);
    let q = *C.base_ring().modulus() as i128;
    return C.wrt_canonical_basis(x).iter().map(|c| (
        (StaticRing::<i128>::RING.rounded_div(C.base_ring().smallest_lift(c) as i128 * (1 << BITS), &q) + (1 << BITS)) % (1 << BITS)
    ).try_into().unwrap()).collect();
}

#[instrument(skip_all)]
pub fn mod_switch_decode<const BITS: usize>(C: &CipherRing, x: &[u64]) -> El<CipherRing> {
    assert!(BITS <= u64::BITS as usize);
    assert_eq!(C.rank(), x.len());
    let q = *C.base_ring().modulus();
    let mod_q = C.base_ring().can_hom(&StaticRing::<i64>::RING).unwrap();
    return C.from_canonical_basis(x.iter().map(|c| 
        mod_q.map(((*c as i128 * q as i128) / (1 << BITS)).try_into().unwrap())
    ));
}

#[instrument(skip_all)]
pub fn enc_sym_b<R: Rng + CryptoRng>(P: &PlainRing, C: &CipherRing, a: [u8; 32], mut rng: R, sk: &El<CipherRing>, m: &El<PlainRing>, sigma: f64) -> El<CipherRing> {
    let a = expand(C, a);
    let e = C.from_canonical_basis((0..C.rank()).map(|_| C.base_ring().int_hom().map((<_ as Distribution<f64>>::sample(&StandardNormal, &mut rng) * sigma).round() as i32)));
    let ZZ_to_Cbase = C.base_ring().can_hom(&StaticRing::<i64>::RING).unwrap();
    let ZZ_to_Pbase = P.base_ring().can_hom(&StaticRing::<i64>::RING).unwrap();
    let P_modulus_inv = C.base_ring().invert(&ZZ_to_Cbase.map(*P.base_ring().modulus())).unwrap();
    let rescale = |c| {
        let c_upscaled= P.base_ring().smallest_lift(c) * *C.base_ring().modulus();
        let delta = P.base_ring().smallest_lift(ZZ_to_Pbase.map(c_upscaled));
        let c_downscaled = C.base_ring().mul(ZZ_to_Cbase.map(c_upscaled - delta), P_modulus_inv);
        return c_downscaled;
    };
    let payload = C.from_canonical_basis(P.wrt_canonical_basis(m).iter().map(rescale));
    let b = C.add(C.sub(e, C.mul_ref_snd(a, &sk)), payload);
    return b;
}

#[instrument(skip_all)]
pub fn dec(P: &PlainRing, C: &CipherRing, sk: &El<CipherRing>, ct: &(El<CipherRing>, El<CipherRing>)) -> El<PlainRing> {
    let noisy_dec = C.add_ref_snd(C.mul_ref(&ct.0, sk), &ct.1);
    let ZZ_to_Cbase = C.base_ring().can_hom(&StaticRing::<i64>::RING).unwrap();
    let ZZ_to_Pbase = P.base_ring().can_hom(&StaticRing::<i64>::RING).unwrap();
    let C_modulus_inv = P.base_ring().invert(&ZZ_to_Pbase.map(*C.base_ring().modulus())).unwrap();
    let rescale = |c| {
        let c_upscaled= C.base_ring().smallest_lift(c) * *P.base_ring().modulus();
        let delta = C.base_ring().smallest_lift(ZZ_to_Cbase.map(c_upscaled));
        let c_downscaled = P.base_ring().mul(ZZ_to_Pbase.map(c_upscaled - delta), C_modulus_inv);
        return c_downscaled;
    };
    return P.from_canonical_basis(C.wrt_canonical_basis(&noisy_dec).iter().map(rescale));
}

#[instrument(skip_all)]
pub fn gen_gk_b<R: Rng + CryptoRng>(C: &CipherRing, seeds_a: &[[u8; 32]], mut rng: R, sk: &El<CipherRing>, g: usize, digits: usize, log2_B: usize, sigma: f64) -> Vec<El<CipherRing>> {
    let Gal_ring = Zn::new(2 * C.rank() as u64);
    let inv_g = Gal_ring.invert(&Gal_ring.coerce(&StaticRing::<i64>::RING, g as i64)).unwrap();
    let s_wrt_basis = C.wrt_canonical_basis(sk);
    let gal_s = C.from_canonical_basis((0..C.rank()).map(|i| {
        let new_idx: usize = Gal_ring.smallest_positive_lift(Gal_ring.mul(Gal_ring.coerce(&StaticRing::<i64>::RING, i as i64), inv_g)).try_into().unwrap();
        if new_idx > C.rank() {
            C.base_ring().negate(s_wrt_basis.at(new_idx - C.rank()))
        } else {
            s_wrt_basis.at(new_idx)
        }
    }));
    (0..digits).map(|i| {
        let seed_a = seeds_a[i];
        let a = expand(C, seed_a);
        let e = C.from_canonical_basis((0..C.rank()).map(|_| C.base_ring().int_hom().map((<_ as Distribution<f64>>::sample(&StandardNormal, &mut rng) * sigma).round() as i32)));
        let payload = C.inclusion().mul_map(C.clone_el(&gal_s), C.base_ring().pow(C.base_ring().coerce(&StaticRing::<i64>::RING, 1 << log2_B), i));
        let b = C.add(C.sub(e, C.mul_ref_snd(a, &sk)), payload);
        return b;
    }).collect()
}