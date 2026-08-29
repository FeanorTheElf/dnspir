//! The DNSPIR private-information-retrieval engine.
//!
//! This crate is self-contained and knows nothing about DNS: it treats a
//! database as a flat array of fixed-size plaintext ring elements of
//! `Z_t[X] / (X^N + 1)` (`N = 2048`, `t = 1025`, i.e. 2560 B of payload per
//! entry) and lets a client retrieve one of them without the server
//! learning which.
//!
//! Retrieval is two-level: the *primary* index selects a slot inside each
//! shard of a fleet of preprocessed databases, the *secondary* index
//! selects the shard. Both levels are answered homomorphically under BFV;
//! the second level runs against a database that is assembled per query
//! from the first-level results.
//!
//! The modules, bottom-up:
//!
//! * [`avx`] — the AVX-512 intrinsics used by the inner-product loop, plus
//!   a portable emulation behind the `emulate_avx512` feature.
//! * [`simd_zn`] — an 8-lane SIMD wrapper around small rings `Z/qZ`.
//! * [`bfv`] — BFV ring types, modulus-switching encode/decode, symmetric
//!   encryption and decryption.
//! * [`base_pir`] — one preprocessed shard ([`base_pir::PIRDatabase`]), the
//!   Galois sub-group choice ([`base_pir::IndexGroup`]) and the query
//!   evaluation over a shard.
//! * [`double_pir`] — composition of a shard fleet into a single two-level
//!   retrieval.
//! * [`pir_wrapper`] — the byte-level API used by callers: query
//!   preparation, server-side evaluation, reply decoding, and the mapping
//!   from an entry count to the database shape.
//!
//! [`align`] and [`permute`] are small allocation/permutation helpers used
//! by the engine internals; they are public only because they appear in
//! the signatures of engine types.
//!
//! Callers normally only need [`pir_wrapper`]:
//!
//! ```text
//! let (query, seed) = pir_wrapper::prepare_query(rng, primary, secondary, num_entries);
//! let reply         = pir_wrapper::process_query(&databases, &query, None);
//! let coefficients  = pir_wrapper::process_reply(&reply, seed);
//! ```
#![feature(allocator_api)]
#![feature(ptr_alignment_type)]
#![feature(min_specialization)]
#![feature(iter_array_chunks)]
#![feature(pointer_is_aligned_to)]
#![feature(macro_metavar_expr_concat)]
#![allow(non_snake_case)]

pub mod align;
pub mod avx;
pub mod base_pir;
pub mod bfv;
pub mod double_pir;
pub mod permute;
pub mod pir_wrapper;
pub mod simd_zn;
