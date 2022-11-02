//! TODO docs

#![feature(allocator_api)]
#![feature(maybe_uninit_slice)]
#![feature(maybe_uninit_write_slice)]

use core::{alloc::Allocator, cmp::max, mem::ManuallyDrop, mem::MaybeUninit, ptr, slice};
use std::{
    alloc::Global,
    io::{Result, Write},
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

    /// Returns a shared reference to the filled portion of the buffer.
    #[inline]
    pub fn filled(&self) -> &[u8] {
        // SAFETY: We only slice the filled part of the buffer, which is always valid
        unsafe { MaybeUninit::slice_assume_init_ref(slice::from_raw_parts(self.data, self.filled)) }
    }

    /// Returns a cursor over the unfilled part of the buffer.
    #[inline]
    pub fn unfilled(&mut self) -> OwnedCursor<'_, A> {
        OwnedCursor {
            start: self.filled,
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
pub struct OwnedCursor<'buf, A: 'static + Allocator> {
    buf: &'buf mut OwnedBuf<A>,
    start: usize,
}

impl<'a, A: 'static + Allocator> OwnedCursor<'a, A> {
    /// Reborrow this cursor by cloning it with a smaller lifetime.
    ///
    /// Since a cursor maintains unique access to its underlying buffer, the borrowed cursor is
    /// not accessible while the new cursor exists.
    #[inline]
    pub fn reborrow<'this>(&'this mut self) -> OwnedCursor<'this, A> {
        OwnedCursor {
            buf: self.buf,
            start: self.start,
        }
    }

    /// Returns the available space in the cursor.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.capacity - self.buf.filled
    }

    /// Returns the number of bytes written to this cursor since it was created from a `BorrowedBuf`.
    ///
    /// Note that if this cursor is a reborrowed clone of another, then the count returned is the
    /// count written via either cursor, not the count since the cursor was reborrowed.
    #[inline]
    pub fn written(&self) -> usize {
        self.buf.filled - self.start
    }

    /// Returns a shared reference to the initialized portion of the cursor.
    #[inline]
    pub fn init_ref(&self) -> &[u8] {
        let filled = self.buf.filled;
        // SAFETY: We only slice the initialized part of the buffer, which is always valid
        unsafe {
            MaybeUninit::slice_assume_init_ref(
                &slice::from_raw_parts(self.buf.data, self.buf.init)[filled..],
            )
        }
    }

    /// Returns a mutable reference to the initialized portion of the cursor.
    #[inline]
    pub fn init_mut(&mut self) -> &mut [u8] {
        let filled = self.buf.filled;
        let init = self.buf.init;
        // SAFETY: We only slice the initialized part of the buffer, which is always valid
        unsafe { MaybeUninit::slice_assume_init_mut(&mut self.buf_as_slice()[filled..init]) }
    }

    /// Returns a mutable reference to the uninitialized part of the cursor.
    ///
    /// It is safe to uninitialize any of these bytes.
    #[inline]
    pub fn uninit_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        let init = self.buf.init;
        unsafe { &mut self.buf_as_slice()[init..] }
    }

    /// Returns a mutable reference to the whole cursor.
    ///
    /// # Safety
    ///
    /// The caller must not uninitialize any bytes in the initialized portion of the cursor.
    #[inline]
    pub unsafe fn as_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        let filled = self.buf.filled;
        &mut self.buf_as_slice()[filled..]
    }

    #[inline]
    unsafe fn buf_as_slice(&mut self) -> &mut [MaybeUninit<u8>] {
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

    /// Initializes all bytes in the cursor.
    #[inline]
    pub fn ensure_init(&mut self) -> &mut Self {
        for byte in self.uninit_mut() {
            byte.write(0);
        }
        self.buf.init = self.buf.capacity;

        self
    }

    /// Asserts that the first `n` unfilled bytes of the cursor are initialized.
    ///
    /// `BorrowedBuf` assumes that bytes are never de-initialized, so this method does nothing when
    /// called with fewer bytes than are already known to be initialized.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the first `n` bytes of the buffer have already been initialized.
    #[inline]
    pub unsafe fn set_init(&mut self, n: usize) -> &mut Self {
        self.buf.init = max(self.buf.init, self.buf.filled + n);
        self
    }

    /// Appends data to the cursor, advancing position within its buffer.
    ///
    /// # Panics
    ///
    /// Panics if `self.capacity()` is less than `buf.len()`.
    #[inline]
    pub fn append(&mut self, buf: &[u8]) {
        assert!(self.capacity() >= buf.len());

        // SAFETY: we do not de-initialize any of the elements of the slice
        unsafe {
            MaybeUninit::write_slice(&mut self.as_mut()[..buf.len()], buf);
        }

        // SAFETY: We just added the entire contents of buf to the filled section.
        unsafe {
            self.set_init(buf.len());
        }
        self.buf.filled += buf.len();
    }
}

impl<'a, A: 'static + Allocator> Write for OwnedCursor<'a, A> {
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        self.append(buf);
        Ok(buf.len())
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
        let mut buf: OwnedBuf = v.into();
        assert_eq!(buf.filled(), &[1, 2, 3]);
        let _cursor = buf.unfilled();
    }
}
