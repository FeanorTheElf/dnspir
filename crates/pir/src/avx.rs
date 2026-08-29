//! Thin newtypes over the AVX-512 intrinsics used by the inner-product
//! loop, plus a portable fallback.
//!
//! Every operation exists twice: once as the real `x86_64` AVX-512
//! intrinsic, and once — behind the `emulate_avx512` feature — as a pair of
//! AVX2 half-registers implementing the same semantics. That fallback makes
//! the engine buildable and testable on hosts without AVX-512 (at a
//! significant cost in throughput); it is not the configuration the
//! benchmarks are meant to be run in.
//!
//! [`Context`] is a zero-sized capability token: obtaining one via
//! [`Context::check_target_features`] asserts once that the required CPU
//! features are present, so the individual operations can be `unsafe`-free
//! at their call sites.

use std::arch::x86_64::*;

#[derive(Clone, Copy)]
pub struct Context;

#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
#[cfg(feature = "emulate_avx512")]
#[repr(C)]
pub struct m512i {
    data0: __m256i,
    data1: __m256i
}

#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
#[cfg(not(feature = "emulate_avx512"))]
#[repr(transparent)]
pub struct m512i {
    data: __m512i
}

#[derive(Clone, Copy)]
#[allow(unused)]
#[allow(non_camel_case_types)]
pub struct m256i {
    data: __m256i
}

#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
#[cfg(feature = "emulate_avx512")]
pub struct m512_mask {
    data0: __m256i,
    data1: __m256i
}

#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
#[cfg(not(feature = "emulate_avx512"))]
pub struct m512_mask {
    data: u8
}

impl Context {
    
    #[cfg(feature = "emulate_avx512")]
    #[inline(always)]
    pub fn check_target_features() -> Self {
        assert!(cfg!(target_arch = "x86_64"));
        assert!(is_x86_feature_detected!("avx2"));
        return Context;
    }

    #[cfg(not(feature = "emulate_avx512"))]
    #[inline(always)]
    pub fn check_target_features() -> Self {
        assert!(cfg!(target_arch = "x86_64"));
        assert!(is_x86_feature_detected!("avx512f"));
        return Context;
    }

    #[cfg(feature = "emulate_avx512")]
    #[inline(always)]
    pub fn mm512_set1_epi64(self, a: i64) -> m512i {
        unsafe { m512i { data0: _mm256_set1_epi64x(a), data1: _mm256_set1_epi64x(a) } }
    }

    #[cfg(not(feature = "emulate_avx512"))]
    #[inline(always)]
    pub fn mm512_set1_epi64(self, a: i64) -> m512i {
        unsafe { m512i { data: _mm512_set1_epi64(a) } }
    }

    #[cfg(feature = "emulate_avx512")]
    #[inline(always)]
    pub fn mm512_extract_epi64<const K: u32>(self, a: m512i) -> i64 {
        match K {
            0 => unsafe { _mm256_extract_epi64::<0>(a.data0) },
            1 => unsafe { _mm256_extract_epi64::<1>(a.data0) },
            2 => unsafe { _mm256_extract_epi64::<2>(a.data0) },
            3 => unsafe { _mm256_extract_epi64::<3>(a.data0) },
            4 => unsafe { _mm256_extract_epi64::<0>(a.data1) },
            5 => unsafe { _mm256_extract_epi64::<1>(a.data1) },
            6 => unsafe { _mm256_extract_epi64::<2>(a.data1) },
            7 => unsafe { _mm256_extract_epi64::<3>(a.data1) },
            _ => unreachable!()
        }
    }

    #[cfg(not(feature = "emulate_avx512"))]
    #[inline(always)]
    pub fn mm512_extract_epi64<const K: u32>(self, a: m512i) -> i64 {
        match K {
            0 => unsafe { _mm256_extract_epi64::<0>(_mm512_extracti64x4_epi64::<0>(a.data)) },
            1 => unsafe { _mm256_extract_epi64::<1>(_mm512_extracti64x4_epi64::<0>(a.data)) },
            2 => unsafe { _mm256_extract_epi64::<2>(_mm512_extracti64x4_epi64::<0>(a.data)) },
            3 => unsafe { _mm256_extract_epi64::<3>(_mm512_extracti64x4_epi64::<0>(a.data)) },
            4 => unsafe { _mm256_extract_epi64::<0>(_mm512_extracti64x4_epi64::<1>(a.data)) },
            5 => unsafe { _mm256_extract_epi64::<1>(_mm512_extracti64x4_epi64::<1>(a.data)) },
            6 => unsafe { _mm256_extract_epi64::<2>(_mm512_extracti64x4_epi64::<1>(a.data)) },
            7 => unsafe { _mm256_extract_epi64::<3>(_mm512_extracti64x4_epi64::<1>(a.data)) },
            _ => unreachable!()
        }
    }

    #[cfg(feature = "emulate_avx512")]
    #[inline(always)]
    pub fn mm512_setr_epi64(self, e0: i64, e1: i64, e2: i64, e3: i64, e4: i64, e5: i64, e6: i64, e7: i64) -> m512i {
        unsafe { m512i { data0: _mm256_setr_epi64x(e0, e1, e2, e3), data1: _mm256_setr_epi64x(e4, e5, e6, e7) } }
    }

    #[cfg(not(feature = "emulate_avx512"))]
    #[inline(always)]
    pub fn mm512_setr_epi64(self, e0: i64, e1: i64, e2: i64, e3: i64, e4: i64, e5: i64, e6: i64, e7: i64) -> m512i {
        unsafe { m512i { data: _mm512_setr_epi64(e0, e1, e2, e3, e4, e5, e6, e7) } }
    }

    #[cfg(feature = "emulate_avx512")]
    #[inline(always)]
    pub fn mm512_srli_epi64<const K: u32>(self, x: m512i) -> m512i {
        macro_rules! impl_srli_k {
            ($actual:ident; $($num:literal),*) => {
                match $actual {
                    $($num => unsafe { m512i { data0: _mm256_srli_epi64::<$num>(x.data0), data1: _mm256_srli_epi64::<$num>(x.data1) } }),*,
                    _ => unreachable!()
                }
            };
        }
        impl_srli_k!(K; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
    }

    #[cfg(not(feature = "emulate_avx512"))]
    #[inline(always)]
    pub fn mm512_srli_epi64<const K: u32>(self, x: m512i) -> m512i {
        unsafe { m512i { data: _mm512_srli_epi64::<K>(x.data) } }
    }

    #[cfg(feature = "emulate_avx512")]
    #[inline(always)]
    pub fn mm512_srai_epi64<const K: u32>(self, x: m512i) -> m512i {
        macro_rules! impl_srai_k {
            ($actual:ident; $($num:literal),*) => {
                match $actual {
                    $($num => unsafe { 
                        let _mm256_srai_epi64 = |x: __m256i| {
                            let sign_mask = _mm256_slli_epi64::<{64 - $num}>(_mm256_cmpgt_epi64(_mm256_setzero_si256(), x));
                            return _mm256_or_epi64(_mm256_srli_epi64::<$num>(x), sign_mask);
                        };
                        m512i { data0: _mm256_srai_epi64(x.data0), data1: _mm256_srai_epi64(x.data1) } 
                    }),*,
                    _ => unreachable!()
                }
            };
        }
        impl_srai_k!(K; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
    }

    #[cfg(not(feature = "emulate_avx512"))]
    #[inline(always)]
    pub fn mm512_srai_epi64<const K: u32>(self, x: m512i) -> m512i {
        unsafe { m512i { data: _mm512_srai_epi64::<K>(x.data) } }
    }

    #[cfg(feature = "emulate_avx512")]
    #[inline(always)]
    pub fn mm512_slli_epi64<const K: u32>(self, x: m512i) -> m512i {
        macro_rules! impl_slli_k {
            ($actual:ident; $($num:literal),*) => {
                match $actual {
                    $($num => unsafe { m512i { data0: _mm256_slli_epi64::<$num>(x.data0), data1: _mm256_slli_epi64::<$num>(x.data1) } }),*,
                    _ => unreachable!()
                }
            };
        }
        impl_slli_k!(K; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
    }

    #[cfg(not(feature = "emulate_avx512"))]
    #[inline(always)]
    pub fn mm512_slli_epi64<const K: u32>(self, x: m512i) -> m512i {
        unsafe { m512i { data: _mm512_slli_epi64::<K>(x.data) } }
    }

    #[cfg(feature = "emulate_avx512")]
    #[inline(always)]
    pub fn mm512_sub_epi64(self, lhs: m512i, rhs: m512i) -> m512i {
        unsafe { m512i { data0: _mm256_sub_epi64(lhs.data0, rhs.data0), data1: _mm256_sub_epi64(lhs.data1, rhs.data1) } }
    }

    #[cfg(not(feature = "emulate_avx512"))]
    #[inline(always)]
    pub fn mm512_sub_epi64(self, lhs: m512i, rhs: m512i) -> m512i {
        unsafe { m512i { data: _mm512_sub_epi64(lhs.data, rhs.data) } }
    }

    #[cfg(feature = "emulate_avx512")]
    #[inline(always)]
    pub fn mm512_add_epi64(self, lhs: m512i, rhs: m512i) -> m512i {
        unsafe { m512i { data0: _mm256_add_epi64(lhs.data0, rhs.data0), data1: _mm256_add_epi64(lhs.data1, rhs.data1) } }
    }

    #[cfg(not(feature = "emulate_avx512"))]
    #[inline(always)]
    pub fn mm512_add_epi64(self, lhs: m512i, rhs: m512i) -> m512i {
        unsafe { m512i { data: _mm512_add_epi64(lhs.data, rhs.data) } }
    }

    #[cfg(feature = "emulate_avx512")]
    #[inline(always)]
    pub fn mm512_mul_epu32(self, lhs: m512i, rhs: m512i) -> m512i {
        unsafe { m512i { data0: _mm256_mul_epu32(lhs.data0, rhs.data0), data1: _mm256_mul_epu32(lhs.data1, rhs.data1) } }
    }

    #[cfg(not(feature = "emulate_avx512"))]
    #[inline(always)]
    pub fn mm512_mul_epu32(self, lhs: m512i, rhs: m512i) -> m512i {
        unsafe { m512i { data: _mm512_mul_epu32(lhs.data, rhs.data) } }
    }

    #[cfg(feature = "emulate_avx512")]
    #[inline(always)]
    pub fn mm512_cmpgt_epi64_mask(self, lhs: m512i, rhs: m512i) -> m512_mask {
        unsafe { m512_mask { data0: _mm256_cmpgt_epi64(lhs.data0, rhs.data0), data1: _mm256_cmpgt_epi64(lhs.data1, rhs.data1) } }
    }

    #[cfg(not(feature = "emulate_avx512"))]
    #[inline(always)]
    pub fn mm512_cmpgt_epi64_mask(self, lhs: m512i, rhs: m512i) -> m512_mask {
        unsafe { m512_mask { data: _mm512_cmpgt_epi64_mask(lhs.data, rhs.data) } }
    }

    #[cfg(feature = "emulate_avx512")]
    #[inline(always)]
    pub fn mm512_cmpeq_epi64_mask(self, lhs: m512i, rhs: m512i) -> m512_mask {
        unsafe { m512_mask { data0: _mm256_cmpeq_epi64(lhs.data0, rhs.data0), data1: _mm256_cmpeq_epi64(lhs.data1, rhs.data1) } }
    }

    #[cfg(not(feature = "emulate_avx512"))]
    #[inline(always)]
    pub fn mm512_cmpeq_epi64_mask(self, lhs: m512i, rhs: m512i) -> m512_mask {
        unsafe { m512_mask { data: _mm512_cmpeq_epi64_mask(lhs.data, rhs.data) } }
    }

    #[cfg(feature = "emulate_avx512")]
    #[inline(always)]
    pub fn mm512_mask(self, x: m512i, mask: m512_mask) -> m512i {
        unsafe { m512i { data0: _mm256_and_si256(x.data0, mask.data0), data1: _mm256_and_si256(x.data1, mask.data1) } }
    }

    #[cfg(not(feature = "emulate_avx512"))]
    #[inline(always)]
    pub fn mm512_mask(self, x: m512i, mask: m512_mask) -> m512i {
        unsafe { m512i { data: _mm512_and_epi64(x.data, _mm512_maskz_set1_epi64(mask.data, -1)) } }
    }

    #[cfg(feature = "emulate_avx512")]
    #[allow(unused)]
    #[inline(always)]
    pub fn mm512_cvtepi32_epi64(self, x: m256i) -> m512i {
        unsafe { m512i { data0: _mm256_cvtepi32_epi64(_mm256_extractf128_si256::<0>(x.data)), data1: _mm256_cvtepi32_epi64(_mm256_extractf128_si256::<1>(x.data)) } }
    }

    #[cfg(not(feature = "emulate_avx512"))]
    #[allow(unused)]
    #[inline(always)]
    pub fn mm512_cvtepi32_epi64(self, x: m256i) -> m512i {
        unsafe { m512i { data: _mm512_cvtepi32_epi64(x.data) } }
    }

    #[cfg(feature = "emulate_avx512")]
    #[allow(unused)]
    #[inline(always)]
    pub fn mm256_setr_epi32(self, e0: i32, e1: i32, e2: i32, e3: i32, e4: i32, e5: i32, e6: i32, e7: i32) -> m256i {
        unsafe { m256i { data: _mm256_setr_epi32(e0, e1, e2, e3, e4, e5, e6, e7) } }
    }

    #[cfg(not(feature = "emulate_avx512"))]
    #[allow(unused)]
    #[inline(always)]
    pub fn mm256_setr_epi32(self, e0: i32, e1: i32, e2: i32, e3: i32, e4: i32, e5: i32, e6: i32, e7: i32) -> m256i {
        unsafe { m256i { data: _mm256_setr_epi32(e0, e1, e2, e3, e4, e5, e6, e7) } }
    }

    #[cfg(not(feature = "emulate_avx512"))]
    #[inline(always)]
    pub fn mm512_extract_si256(self, data: m512i) -> (m256i, m256i) {
        unsafe {(
            m256i { data: _mm512_extracti64x4_epi64::<0>(data.data) },
            m256i { data: _mm512_extracti64x4_epi64::<1>(data.data) },
        )}
    }

    #[cfg(feature = "emulate_avx512")]
    #[inline(always)]
    pub fn mm512_extract_si256(self, data: m512i) -> (m256i, m256i) {
        (m256i { data: data.data0 }, m256i { data: data.data1 })
    }

    #[inline(always)]
    #[allow(unused)]
    pub fn mm_prefetch<const STRATEGY: i32>(self, address: *const i8) {
        unsafe { _mm_prefetch::<STRATEGY>(address) }
    }

}
