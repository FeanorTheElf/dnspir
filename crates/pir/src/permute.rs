//! In-place permutation of a vector, `values_new[i] = values[perm(i)]`.
//!
//! Copied from `feanor-math`, where the equivalent helpers are not public.
//! Used by [`crate::base_pir`] to reorder database entries into the order
//! the Galois-group-aware evaluation expects.

use feanor_math::seq::*;

use std::alloc::{Allocator, Global};

///
/// Computes `values_new[i] = values[perm(i)]`.
/// 
/// Copied from `feanor-math`.
/// 
pub fn permute<V, T, F>(values: V, perm: F)
    where V: SwappableVectorViewMut<T>, F: Fn(usize) -> usize
{
    permute_using_allocator(values, perm, Global)
}

///
/// Computes `values_new[i] = values[perm(i)]`.
/// 
/// Copied from `feanor-math`.
/// 
pub fn permute_using_allocator<V, T, F, A: Allocator>(mut values: V, perm: F, allocator: A)
    where V: SwappableVectorViewMut<T>, F: Fn(usize) -> usize
{
    let mut swapped_indices = Vec::with_capacity_in(values.len(), allocator);
    swapped_indices.extend((0..values.len()).map(|_| false));
    let mut start = 0;
    while start < values.len() {
        let mut current = start;
        let mut next = perm(current);
        while !swapped_indices[next] {
            swapped_indices[current] = true;
            values.swap(current, next);
            current = next;
            next = perm(current);
        }
        swapped_indices[current] = true;
        start += 1;
    }
}

///
/// Computes `values_new[perm(i)] = values[i]`.
/// This is the inverse operation to [`permute()`].
/// 
/// Copied from `feanor-math`.
/// 
#[allow(unused)]
pub fn permute_inv<V, T, F>(values: V, perm: F)
    where V: SwappableVectorViewMut<T>, F: Fn(usize) -> usize
{
    permute_inv_using_allocator(values, perm, Global)
}

///
/// Computes `values_new[perm(i)] = values[i]`.
/// This is the inverse operation to [`permute()`].
/// 
/// Copied from `feanor-math`.
/// 
pub fn permute_inv_using_allocator<V, T, F, A: Allocator>(mut values: V, perm: F, allocator: A)
    where V: SwappableVectorViewMut<T>, F: Fn(usize) -> usize
{
    let mut swapped_indices = Vec::with_capacity_in(values.len(), allocator);
    swapped_indices.extend((0..values.len()).map(|_| false));
    let mut start = 0;
    while start < values.len() {
        let mut current = perm(start);
        swapped_indices[start] = true;
        while !swapped_indices[current] {
            swapped_indices[current] = true;
            values.swap(current, start);
            current = perm(current);
        }
        start += 1;
    }
}
