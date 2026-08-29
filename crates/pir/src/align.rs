//! An allocator adaptor that raises the alignment of every allocation to a
//! fixed minimum.
//!
//! The inner-product loop in [`crate::base_pir`] loads database entries with
//! aligned AVX-512 moves, so the slabs holding them must start on a 64-byte
//! (or larger) boundary — which `Global` does not guarantee for the element
//! types used here. [`AligningAlloc`] over-allocates and offsets, and
//! [`zero_vec`] is the convenience constructor for a zero-initialised,
//! aligned `Vec` of ring elements.

use std::alloc::{Allocator, Global, Layout, AllocError};
use std::ptr::NonNull;
use std::cmp::max;

use feanor_math::integer::IntegerRingStore;
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::*;

pub struct AligningAlloc<A: Allocator = Global> {
    allocator: A,
    min_alignment: usize
}

impl<A: Allocator> AligningAlloc<A> {

    pub fn new(allocator: A, min_alignment: usize) -> Self {
        assert!(1 << StaticRing::<i64>::RING.abs_log2_ceil(&(min_alignment as i64)).unwrap() == min_alignment);
        assert!(min_alignment <= 1024);
        return Self { allocator, min_alignment };
    }
}

impl Default for AligningAlloc {
    fn default() -> Self {
        Self::new(Global, 64)
    }
}

unsafe impl<A: Allocator> Allocator for AligningAlloc<A> {
    
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.allocator.allocate(layout.align_to(max(layout.align(), self.min_alignment)).unwrap())
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.allocator.allocate_zeroed(layout.align_to(max(layout.align(), self.min_alignment)).unwrap())
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { self.allocator.deallocate(ptr, layout.align_to(max(layout.align(), self.min_alignment)).unwrap()) }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        unsafe { self.allocator.grow(ptr, old_layout.align_to(max(old_layout.align(), self.min_alignment)).unwrap(), new_layout.align_to(max(new_layout.align(), self.min_alignment)).unwrap()) }
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        unsafe { self.allocator.grow_zeroed(ptr, old_layout.align_to(max(old_layout.align(), self.min_alignment)).unwrap(), new_layout.align_to(max(new_layout.align(), self.min_alignment)).unwrap()) }
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        unsafe { self.allocator.shrink(ptr, old_layout.align_to(max(old_layout.align(), self.min_alignment)).unwrap(), new_layout.align_to(max(new_layout.align(), self.min_alignment)).unwrap()) }
    }

}

pub fn zero_vec<R>(ring: &R, len: usize) -> Vec<R::Element, AligningAlloc>
    where R: RingBase
{
    let mut result = Vec::with_capacity_in(len, AligningAlloc::default());
    result.extend((0..len).map(|_| ring.zero()));
    return result;
}