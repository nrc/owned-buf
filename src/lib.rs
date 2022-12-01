//! TODO docs

#![feature(allocator_api)]
#![feature(maybe_uninit_slice)]
#![feature(maybe_uninit_write_slice)]

use core::{
    alloc::Allocator,
    cmp::{max, min},
    mem::ManuallyDrop,
    mem::MaybeUninit,
    ptr, slice,
};
use std::{
    alloc::Global,
    io::{Result, Write},
    ops,
};

/// TODO docs
pub struct OwnedBuf<A: 'static + Allocator = Global> {
    data: *mut MaybeUninit<u8>,
    dtor: &'static dyn Fn(&mut OwnedBuf<A>),
    capacity: usize,
    /// The length of `self.buf` which is known to be filled.
    filled: usize,
    /// The length of `self.buf` which is known to be initialized.
    init: usize,
    allocator: A,
}

impl<A: 'static + Allocator> Into<OwnedSlice<A>> for OwnedBuf<A> {
    fn into(self) -> OwnedSlice<A> {
        self.filled()
    }
}

impl<A: 'static + Allocator> Into<OwnedCursor<A>> for OwnedBuf<A> {
    fn into(self) -> OwnedCursor<A> {
        self.unfilled()
    }
}

pub trait UsizeRange {
    fn absolute_indices(&self, start: usize, end: usize) -> (usize, usize);
}

impl UsizeRange for ops::Range<usize> {
    #[inline]
    fn absolute_indices(&self, start: usize, end: usize) -> (usize, usize) {
        assert!(start + self.start <= end);
        assert!(start + self.end <= end);
        (start + self.start, start + self.end)
    }
}

impl UsizeRange for ops::RangeFrom<usize> {
    #[inline]
    fn absolute_indices(&self, start: usize, end: usize) -> (usize, usize) {
        assert!(start + self.start <= end);
        (start + self.start, end)
    }
}

impl UsizeRange for ops::RangeFull {
    #[inline]
    fn absolute_indices(&self, start: usize, end: usize) -> (usize, usize) {
        (start, end)
    }
}

impl UsizeRange for ops::RangeInclusive<usize> {
    #[inline]
    fn absolute_indices(&self, start: usize, end: usize) -> (usize, usize) {
        assert!(start + self.start() <= end);
        assert!(start + self.end() + 1 <= end);
        (start + self.start(), start + self.end() + 1)
    }
}

impl UsizeRange for ops::RangeTo<usize> {
    #[inline]
    fn absolute_indices(&self, start: usize, end: usize) -> (usize, usize) {
        assert!(start + self.end <= end);
        (start, start + self.end)
    }
}

impl UsizeRange for ops::RangeToInclusive<usize> {
    #[inline]
    fn absolute_indices(&self, start: usize, end: usize) -> (usize, usize) {
        assert!(start + self.end + 1 <= end);
        (start, start + self.end + 1)
    }
}

impl<A: 'static + Allocator> OwnedBuf<A> {
    /// TODO docs
    #[inline]
    pub fn new(
        data: *mut MaybeUninit<u8>,
        dtor: &'static dyn Fn(&mut OwnedBuf),
        capacity: usize,
        filled: usize,
        init: usize,
    ) -> OwnedBuf<Global> {
        OwnedBuf::new_in(data, dtor, capacity, filled, init, Global)
    }

    /// TODO docs
    #[inline]
    pub fn new_in(
        data: *mut MaybeUninit<u8>,
        dtor: &'static dyn Fn(&mut OwnedBuf<A>),
        capacity: usize,
        filled: usize,
        init: usize,
        allocator: A,
    ) -> OwnedBuf<A> {
        OwnedBuf {
            data,
            dtor,
            capacity,
            filled,
            init,
            allocator,
        }
    }

    /// TODO docs
    #[inline]
    pub unsafe fn into_vec(self) -> Vec<u8, A> {
        let this = ManuallyDrop::new(self);
        Vec::from_raw_parts_in(this.data as *mut u8, this.filled, this.capacity, unsafe {
            ptr::read(&this.allocator)
        })
    }

    /// TODO docs
    #[inline]
    pub unsafe fn into_uninit_vec(self) -> Vec<MaybeUninit<u8>, A> {
        let this = ManuallyDrop::new(self);
        Vec::from_raw_parts_in(this.data, this.filled, this.capacity, unsafe {
            ptr::read(&this.allocator)
        })
    }

    /// Returns the length of the initialized part of the buffer.
    #[inline]
    pub fn init_len(&self) -> usize {
        self.init
    }

    /// TODO docs.
    #[inline]
    pub fn filled_bytes(&self) -> &[u8] {
        // SAFETY: We only slice the filled part of the buffer, which is always valid
        unsafe { MaybeUninit::slice_assume_init_ref(slice::from_raw_parts(self.data, self.filled)) }
    }

    /// TODO docs.
    #[inline]
    pub fn filled(self) -> OwnedSlice<A> {
        OwnedSlice {
            start: 0,
            end: self.filled,
            buf: self,
        }
    }

    /// TODO docs.
    #[inline]
    pub fn filled_slice(self, range: impl UsizeRange) -> OwnedSlice<A> {
        let (start, end) = range.absolute_indices(0, self.filled);
        OwnedSlice {
            start,
            end,
            buf: self,
        }
    }

    /// Returns a cursor over the unfilled part of the buffer.
    #[inline]
    pub fn unfilled(self) -> OwnedCursor<A> {
        OwnedCursor {
            end: self.capacity,
            buf: self,
        }
    }

    /// TODO docs.
    #[inline]
    pub fn unfilled_slice(self, range: ops::RangeTo<usize>) -> OwnedCursor<A> {
        assert!(range.end >= self.filled && range.end <= self.capacity);
        OwnedCursor {
            end: range.end,
            buf: self,
        }
    }

    /// Clears the buffer, resetting the filled region to empty.
    ///
    /// The number of initialized bytes is not changed, and the contents of the buffer are not modified.
    #[inline]
    pub fn clear(&mut self) -> &mut Self {
        self.filled = 0;
        self
    }

    /// Asserts that the first `n` bytes of the buffer are initialized.
    ///
    /// `OwnedBuf` assumes that bytes are never de-initialized, so this method does nothing when called with fewer
    /// bytes than are already known to be initialized.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the first `n` unfilled bytes of the buffer have already been initialized.
    #[inline]
    pub unsafe fn set_init(&mut self, n: usize) -> &mut Self {
        self.init = max(self.init, n);
        self
    }
}

fn drop_vec<A: 'static + Allocator>(buf: &mut OwnedBuf<A>) {
    let buf = ManuallyDrop::new(buf);
    let _vec = unsafe {
        Vec::from_raw_parts_in(
            buf.data,
            buf.filled,
            buf.capacity,
            ptr::read(&buf.allocator),
        )
    };
}

impl<A: 'static + Allocator> From<Vec<MaybeUninit<u8>, A>> for OwnedBuf<A> {
    fn from(v: Vec<MaybeUninit<u8>, A>) -> OwnedBuf<A> {
        let (data, len, capacity, allocator) = v.into_raw_parts_with_alloc();
        OwnedBuf {
            data,
            dtor: &drop_vec,
            capacity,
            filled: len,
            init: len,
            allocator,
        }
    }
}

impl<A: 'static + Allocator> From<Vec<u8, A>> for OwnedBuf<A> {
    fn from(v: Vec<u8, A>) -> OwnedBuf<A> {
        let (data, len, capacity, allocator) = v.into_raw_parts_with_alloc();
        OwnedBuf {
            data: data as *mut MaybeUninit<u8>,
            dtor: &drop_vec,
            capacity,
            filled: len,
            init: len,
            allocator,
        }
    }
}

impl<A: 'static + Allocator> Drop for OwnedBuf<A> {
    fn drop(&mut self) {
        (self.dtor)(self)
    }
}

/// TODO docs
pub struct OwnedSlice<A: 'static + Allocator> {
    buf: OwnedBuf<A>,
    // Invariant: start <= buf.filled
    start: usize,
    // Invariant: end >= start && end <= buf.filled
    end: usize,
}

/// TODO docs
pub struct OwnedCursor<A: 'static + Allocator> {
    buf: OwnedBuf<A>,
    // Invariant: end >= buf.filled && end <= buf.capacity
    end: usize,
}

impl<A: 'static + Allocator> OwnedSlice<A> {
    #[inline]
    pub fn slice(self, range: impl UsizeRange) -> Self {
        let (start, end) = range.absolute_indices(self.start, self.end);
        OwnedSlice {
            buf: self.buf,
            start,
            end,
        }
    }

    #[inline]
    pub fn into_buf(self) -> OwnedBuf<A> {
        self.buf
    }
}

impl<A: 'static + Allocator> ops::Deref for OwnedSlice<A> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: We only slice the filled part of the buffer, which is always valid
        unsafe {
            MaybeUninit::slice_assume_init_ref(slice::from_raw_parts(self.buf.data, self.end))
        }
    }
}

impl<A: 'static + Allocator> OwnedCursor<A> {
    #[inline]
    pub fn into_buf(self) -> OwnedBuf<A> {
        self.buf
    }

    /// Returns the available space in the slice.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.end - self.buf.filled
    }

    /// Returns a mutable reference to the initialized portion of the slice.
    #[inline]
    pub fn init_mut(&mut self) -> &mut [u8] {
        let start = self.buf.filled;
        let end = min(self.end, self.buf.init);

        // SAFETY: We only slice the initialized part of the buffer, which is always valid
        unsafe { MaybeUninit::slice_assume_init_mut(&mut self.buf_as_bytes()[start..end]) }
    }

    /// Returns a mutable reference to the uninitialized part of the slice.
    ///
    /// It is safe to uninitialize any of these bytes.
    #[inline]
    pub fn uninit_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        let start = self.buf.init;
        let end = max(self.end, self.buf.init);
        unsafe { &mut self.buf_as_bytes()[start..end] }
    }

    /// Returns a mutable reference to the whole slice.
    ///
    /// # Safety
    ///
    /// The caller must not uninitialize any bytes in the initialized portion of the slice.
    #[inline]
    pub unsafe fn as_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        let start = self.buf.filled;
        let end = self.end;
        &mut self.buf_as_bytes()[start..end]
    }

    #[inline]
    unsafe fn buf_as_bytes(&mut self) -> &mut [MaybeUninit<u8>] {
        slice::from_raw_parts_mut(self.buf.data, self.buf.capacity)
    }

    /// Advance the cursor by asserting that `n` bytes have been filled.
    ///
    /// After advancing, the `n` bytes are no longer accessible via the cursor and can only be
    /// accessed via the underlying buffer. I.e., the buffer's filled portion grows by `n` elements
    /// and its unfilled portion (and the capacity of this cursor) shrinks by `n` elements.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the first `n` bytes of the cursor have been properly
    /// initialised.
    #[inline]
    pub unsafe fn advance(&mut self, n: usize) -> &mut Self {
        self.buf.filled += n;
        self.buf.init = max(self.buf.init, self.buf.filled);
        self
    }

    /// Initializes all bytes in the slice.
    #[inline]
    pub fn ensure_init(&mut self) -> &mut Self {
        for byte in self.uninit_mut() {
            byte.write(0);
        }
        self.buf.init = max(self.end, self.buf.init);

        self
    }

    /// Asserts that the first `n` unfilled bytes of the slice are initialized.
    ///
    /// `OwnedBuf` assumes that bytes are never de-initialized, so this method does nothing when
    /// called with fewer bytes than are already known to be initialized.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the first `n` bytes of the slice have already been initialized.
    #[inline]
    pub unsafe fn set_init(&mut self, n: usize) -> &mut Self {
        self.buf.init = max(self.buf.init, self.buf.filled + n);
        self
    }

    /// Copies `data` to the start of the slice and advances the slice (TODO what does that mean?)
    ///
    /// # Panics
    ///
    /// Panics if `self.capacity()` is less than `buf.len()`.
    #[inline]
    pub fn write_slice(&mut self, data: &[u8]) {
        assert!(self.capacity() >= data.len());

        // SAFETY: we do not de-initialize any of the elements of the slice
        unsafe {
            MaybeUninit::write_slice(&mut self.as_mut()[..data.len()], data);
        }

        // SAFETY: We just added the entire contents of data.
        unsafe {
            self.advance(data.len());
        }
    }
}

impl<A: 'static + Allocator> Write for OwnedCursor<A> {
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        let len = min(self.capacity(), buf.len());
        self.write_slice(&buf[..len]);
        Ok(len)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_buf_smoke() {
        let v: Vec<u8> = vec![1, 2, 3];
        let buf: OwnedBuf = v.into();
        let filled = buf.filled();
        assert_eq!(&*filled, &[1, 2, 3]);
        let mut buf = filled.into_buf();
        buf.clear();
        assert_eq!(buf.filled_bytes().len(), 0);
        let mut cursor = buf.unfilled();
        cursor.write_slice(&[6, 5, 4]);
        let v = unsafe { cursor.into_buf().into_vec() };
        assert_eq!(&*v, &[6, 5, 4]);
    }
}
