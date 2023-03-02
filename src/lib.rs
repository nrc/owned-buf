//! An owned, fixed length buffer of bytes (`u8`s) and helper types.
//!
//! The primary type is [`OwnedBuf`] which is a sort-of abstract wrapper type for an owned
//! sequence of bytes. [`OwnedSlice`] and [`OwnedCursor`] give read access to the filled
//! part of an `OwnedBuf` and write access to the unfilled part of an `OwnedBuf`, respectively.
//!
//! An `OwnedBuf` is primarily useful where an IO operation takes ownership of supplied buffers, for
//! example [`async OwnedRead`](TODO).
//!
//! An `OwnedBuf`'s data can be in either filled (i.e., data has been written to a byte) or unfilled.
//! Unfilled data may be with initialized or uninitialized. An `OwnedBuf` may contain all three
//! kinds of data at the same time and tracks the state of bytes.
//!
//! ## Lifecycle
//!
//! An `OwnedBuf` is never directly created. Instead, an existing sequence of bytes (e.g., a
//! `Vec<u8>`) is transformed into an `OwnedBuf`. Unlike when taking a slice, the previous sequence
//! type is consumed and ownership of its memory is transferred to the `OwnedBuf` (the memory itself
//! does not move).
//!
//! The `OwnedBuf` is then used. It is transformed into an `OwnedCursor` to write to it and into an
//! `OwnedSlice` to read from it. These types can be transformed back into an `OwnedBuf` as needed.
//! With each transformation, ownership is passed to the new type. If needed the `OwnedBuf` can be
//! reset.
//!
//! An `OwnedBuf` can be transformed back into the original sequence type. Any data written to the
//! `OwnedBuf` can be read. The sequence type can be used in the usual way, including its
//! destruction. If an `OwnedBuf` is not transformed back into the previous type, then its destructor
//! calls a function supplied when the buffer is created. Typically, that converts the buffer into
//! the original sequence type and calls it destructor.
//!
//! ### Destructor safety
//!
//! An `OwnedBuf` takes a destructor as a function pointer and calls it from the `OwnedBuf`'s
//! destructor. The function pointer is marked as `unsafe` and the safety invariant is that the
//! `OwnedBuf` was created from the collection type expected by the destructor function. This
//! invariant must be ensured when the `OwnedBuf` is created, thus `OwnedBuf::new` is `unsafe`.
//!
//! ## Conversion from user types
//!
//! This module includes functionality to transform an `OwnedBuf` from and to a `Vec<u8>` or
//! `Vec<MaybeUninit<u8>>`. `OwnedBuf` is designed to be usable with any owned sequence of bytes. To
//! create an `OwnedBuf` use [`OwnedBuf::new`], you'll supply a data pointer, some metadata, and a
//! destructor function (see below). To transform from an `OwnedBuf`, you use `into_raw_parts` to
//! get the internal state of the `OwnedBuf` and then create the sequence type in the usual way.
//!
//! An `OwnedBuf`'s destructor function has type `&'static dyn Fn(&mut OwnedBuf<A>)`, it passes a
//! mutable reference to the destructor, however, it is guaranteed that the `OwnedBuf` will not be
//! accessed after calling the destructor function. Typically, the destructor function will use
//! [`std::ptr::read`] to get the `OwnedBuf` by value, transform it into the original sequence type,
//! and (implicitly) call its destructor.
//!
//! # Examples
//!
//! Creating an `OwnedBuf` from a `Vec`.
//!
//! ```
//! # use owned_buf::OwnedBuf;
//! let vec = vec![1u8, 2, 3];
//! let buf: OwnedBuf = vec.into();
//!
//! // Use `filled` to view `buf` as an `OwnedSlice` to read from it.
//! assert_eq!(1, buf.filled()[0]);
//!
//! // `Vec`'s destructor will be called via `OwnedBuf`'s.
//! ```
//!
//! Writing into an `OwnedBuf` and converting it back into a `Vec`.
//!
//! ```
//! # use owned_buf::OwnedBuf;
//! let vec: Vec<u8> = Vec::with_capacity(8);
//! let buf: OwnedBuf = vec.into();
//! assert_eq!(0, buf.filled_len());
//! assert_eq!(8, buf.capacity());
//!
//! // Get a cursor to write into the `OwnedBuf`.
//! let mut cursor = buf.unfilled();
//! cursor.write_slice(&[0, 1, 2, 3]);
//!
//! // Convert the cursor back into an `OwnedBuf`.
//! let buf = cursor.into_buf();
//! assert_eq!(4, buf.filled_len());
//!
//! let vec = unsafe { buf.into_vec() };
//! assert_eq!(4, vec.len());
//! assert_eq!(8, vec.capacity());
//! assert_eq!(3, vec[3]);
//! ```

#![feature(maybe_uninit_slice)]
#![feature(maybe_uninit_write_slice)]
#![feature(vec_into_raw_parts)]

use core::{
    cmp::{max, min},
    mem::{ManuallyDrop, MaybeUninit},
    ops, ptr, slice,
};
use std::io::{Result, Write};

/// An owned, fixed length buffer of bytes.
///
/// An `OwnedBuf` is created from an existing collection type, it is an abstract
/// wrapper of sorts. See the [module docs](self) for details and examples.
///
/// This type is a sort of "double cursor". It tracks three regions in the buffer: a region at the beginning of the
/// buffer that has been logically filled with data, a region that has been initialized at some point but not yet
/// logically filled, and a region at the end that is fully uninitialized. The filled region is guaranteed to be a
/// subset of the initialized region. The filled part can grow or shrink, the initialized part can
/// only grow, and capacity of the buffer (i.e., its total length) is fixed.
///
/// In summary, the contents of the buffer can be visualized as:
/// ```not_rust
/// [             capacity              ]
/// [ filled |         unfilled         ]
/// [    initialized    | uninitialized ]
/// ```
///
/// The API of `OwnedBuf` is focussed on creation and destruction. To read from the filled part,
/// get a view of the buffer using an `OwnedSlice`. To write into the unfilled part, get a view of
/// the buffer using and `OwnedCursor`. Both of these types can be converted back into an `OwnedBuf`.
pub struct OwnedBuf {
    data: *mut MaybeUninit<u8>,
    dtor: unsafe fn(&mut OwnedBuf),
    user_data: *const (),
    capacity: usize,
    /// The length of `self.data` which is known to be filled.
    filled: usize,
    /// The length of `self.data` which is known to be initialized.
    init: usize,
}

impl Into<OwnedSlice> for OwnedBuf {
    fn into(self) -> OwnedSlice {
        self.filled()
    }
}

impl Into<OwnedCursor> for OwnedBuf {
    fn into(self) -> OwnedCursor {
        self.unfilled()
    }
}

impl OwnedBuf {
    /// Create a new OwnedBuf.
    ///
    /// # Safety
    ///
    /// See module docs for safety requirements.
    #[inline]
    pub unsafe fn new(
        data: *mut MaybeUninit<u8>,
        dtor: unsafe fn(&mut OwnedBuf),
        user_data: *const (),
        capacity: usize,
        filled: usize,
        init: usize,
    ) -> OwnedBuf {
        OwnedBuf {
            data,
            dtor,
            user_data,
            capacity,
            filled,
            init,
        }
    }

    /// Convert this buffer into a `Vec` of `u8`.
    ///
    /// # Safety
    ///
    /// It is only safe to use this method if the buffer was created from a `Vec<u8>`.
    #[inline]
    pub unsafe fn into_vec(self) -> Vec<u8> {
        let (data, _, _, filled, _, capacity) = self.into_raw_parts();
        Vec::from_raw_parts(data as *mut u8, filled, capacity)
    }

    /// Convert this buffer into a `Vec` of `MaybeUninit<u8>`.
    ///
    /// # Safety
    ///
    /// It is only safe to use this method if the buffer was created from a `Vec<MaybeUninit<u8>>`.
    #[inline]
    pub unsafe fn into_maybe_uninit_vec(self) -> Vec<MaybeUninit<u8>> {
        let (data, _, _, filled, _, capacity) = self.into_raw_parts();
        Vec::from_raw_parts(data, filled, capacity)
    }

    /// Returns the total size of the buffer.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the length of the filled part of the buffer.
    #[inline]
    pub fn filled_len(&self) -> usize {
        self.filled
    }

    /// Returns the length of the initialized part of the buffer.
    #[inline]
    pub fn init_len(&self) -> usize {
        self.init
    }

    /// Returns an `OwnedSlice` covering the filled part of this buffer.
    #[inline]
    pub fn filled(self) -> OwnedSlice {
        OwnedSlice {
            start: 0,
            end: self.filled,
            buf: self,
        }
    }

    /// Returns an `OwnedSlice` covering a slice of the filled part of this buffer.
    ///
    /// The supplied range must be within the filled part of the buffer.
    ///
    /// # Panics
    ///
    /// This function will panic if `range` is outside the bounds of the filled part of the buffer.
    #[inline]
    pub fn filled_slice(self, range: impl UsizeRange) -> OwnedSlice {
        let (start, end) = range.absolute_indices(0, self.filled);
        OwnedSlice {
            start,
            end,
            buf: self,
        }
    }

    /// Returns a cursor over the unfilled part of the buffer.
    #[inline]
    pub fn unfilled(self) -> OwnedCursor {
        OwnedCursor {
            end: self.capacity,
            buf: self,
        }
    }

    /// Returns a cursor over a slice of the unfilled part of the buffer.
    ///
    /// The supplied range covers the whole buffer, not just the unfilled part. The upper bound
    /// of the range must be within the unfilled part.
    ///
    /// # Examples
    ///
    /// Where `buf` has a capacity (total length) of 16 bytes and the first 4 bytes are filled,
    /// `buf.unfilled_slice(..8)` will return a cursor over the first 4 unfilled bytes.
    #[inline]
    pub fn unfilled_slice(self, range: ops::RangeTo<usize>) -> OwnedCursor {
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

    /// Decomposes this `OwnedBuf` into its raw components.
    ///
    /// Returns `(data, dtor, filled, init, capacity)` where `data` is a pointer to the buffer's
    /// data in memory, `dtor` is the buffer's destructor function, `filled` is the number of bytes
    /// which are filled with data, `init` is the number of bytes which have been initialised (i.e.,
    /// which are safe to treat as `u8` rather than `MaybeUninit<u8>`), and `capacity` is the total
    /// length of the buffer. `filled <= init <= capacity`.
    ///
    /// # Safety
    ///
    /// See module docs for safety requirements on the returned destructor function.
    #[inline]
    pub fn into_raw_parts(
        self,
    ) -> (
        *mut MaybeUninit<u8>,
        unsafe fn(&mut OwnedBuf),
        *const (),
        usize,
        usize,
        usize,
    ) {
        let this = ManuallyDrop::new(self);
        (
            this.data,
            this.dtor,
            this.user_data,
            this.filled,
            this.init,
            this.capacity,
        )
    }
}

impl Drop for OwnedBuf {
    fn drop(&mut self) {
        unsafe { (self.dtor)(self) }
    }
}

unsafe fn drop_vec(buf: &mut OwnedBuf) {
    let (data, _, _, filled, _, capacity) = unsafe { ptr::read(buf) }.into_raw_parts();
    let _vec = Vec::from_raw_parts(data, filled, capacity);
}

impl From<Vec<MaybeUninit<u8>>> for OwnedBuf {
    fn from(v: Vec<MaybeUninit<u8>>) -> OwnedBuf {
        let (data, len, capacity) = v.into_raw_parts();
        OwnedBuf {
            data,
            dtor: drop_vec,
            user_data: ptr::null(),
            capacity,
            filled: len,
            init: len,
        }
    }
}

impl From<Vec<u8>> for OwnedBuf {
    fn from(v: Vec<u8>) -> OwnedBuf {
        let (data, len, capacity) = v.into_raw_parts();
        OwnedBuf {
            data: data as *mut MaybeUninit<u8>,
            dtor: drop_vec,
            user_data: ptr::null(),
            capacity,
            filled: len,
            init: len,
        }
    }
}

/// An owned slice of an `OwnedBuf`.
///
/// Like a regular slice, an `OwnedSlice` may be a view of the whole buffer or
/// a subslice of it. An `OwnedSlice` can be further shrunk. An `OwnedSlice` is
/// always immutable. An `OwnedSlice` takes ownership of the `OwnedBuf` and its
/// underlying data when it is created, it can always be converted back into an
/// `OwnedBuf` (which will be the full, original buffer, not a subslice of it).
///
/// The primary way to read an `OwnedSlice` is as a `&[u8]` via dereferencing. E.g.,
///
/// ```
/// # use owned_buf::OwnedBuf;
/// let buf: OwnedBuf = vec![1u8, 2, 3, 4].into();
/// let slice = buf.filled();
/// assert_eq!(slice[1], 2);
/// ```
pub struct OwnedSlice {
    buf: OwnedBuf,
    // Invariant: start <= buf.filled
    start: usize,
    // Invariant: end >= start && end <= buf.filled
    end: usize,
}

impl OwnedSlice {
    /// Take an (owned) subslice of this `OwnedSlice`.
    #[inline]
    pub fn slice(self, range: impl UsizeRange) -> Self {
        let (start, end) = range.absolute_indices(self.start, self.end);
        OwnedSlice {
            buf: self.buf,
            start,
            end,
        }
    }

    /// Convert this slice back into an `OwnedBuf`.
    #[inline]
    pub fn into_buf(self) -> OwnedBuf {
        self.buf
    }
}

impl ops::Deref for OwnedSlice {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: We only slice the filled part of the buffer, which is always valid
        unsafe {
            MaybeUninit::slice_assume_init_ref(slice::from_raw_parts(self.buf.data, self.end))
        }
    }
}

/// A write-only cursor over an `OwnedBuf`.
///
/// Created via `OwnedBuf::unfilled` or using `Into`.
///
/// An `OwnedCursor` represents the unfilled portion of an `OwnedBuf`, as data is filled it becomes
/// inaccessible via the cursor, so the start of the cursor (or current position, if you think of it
/// that way) is always the first unfilled byte of the underlying buffer. An `OwnedCursor` can contain
/// both initialized and uninitialized bytes.
///
/// An `OwnedCursor` takes ownership of the `OwnedBuf` when it is created. It can always be converted
/// back into and `OwnedBuf`.
pub struct OwnedCursor {
    buf: OwnedBuf,
    // Invariant: end >= buf.filled && end <= buf.capacity
    end: usize,
}

impl OwnedCursor {
    /// Convert this cursor back into its underlying buffer.
    #[inline]
    pub fn into_buf(self) -> OwnedBuf {
        self.buf
    }

    /// Returns the available space in the cursor.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.end - self.buf.filled
    }

    /// Returns a mutable reference to the initialized portion of the cursor.
    #[inline]
    pub fn init_mut(&mut self) -> &mut [u8] {
        let start = self.buf.filled;
        let end = min(self.end, self.buf.init);

        // SAFETY: We only slice the initialized part of the buffer, which is always valid
        unsafe { MaybeUninit::slice_assume_init_mut(&mut self.buf_as_bytes()[start..end]) }
    }

    /// Returns a mutable reference to the uninitialized part of the cursor.
    ///
    /// It is safe to uninitialize any of these bytes.
    #[inline]
    pub fn uninit_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        let start = self.buf.init;
        let end = max(self.end, self.buf.init);
        unsafe { &mut self.buf_as_bytes()[start..end] }
    }

    /// Returns a mutable reference to the whole cursor.
    ///
    /// # Safety
    ///
    /// The caller must not uninitialize any bytes in the initialized portion of the cursor.
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

    /// Initializes all bytes in the cursor.
    #[inline]
    pub fn ensure_init(&mut self) -> &mut Self {
        for byte in self.uninit_mut() {
            byte.write(0);
        }
        self.buf.init = max(self.end, self.buf.init);

        self
    }

    /// Asserts that the first `n` unfilled bytes of the cursor are initialized.
    ///
    /// `OwnedBuf` assumes that bytes are never de-initialized, so this method does nothing when
    /// called with fewer bytes than are already known to be initialized.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the first `n` bytes of the cursor have already been initialized.
    #[inline]
    pub unsafe fn set_init(&mut self, n: usize) -> &mut Self {
        self.buf.init = max(self.buf.init, self.buf.filled + n);
        self
    }

    /// Copies `data` to the start of the cursor and advances the cursor.
    ///
    /// # Panics
    ///
    /// Panics if `self.capacity()` is less than `data.len()`.
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

impl Write for OwnedCursor {
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        let len = min(self.capacity(), buf.len());
        self.write_slice(&buf[..len]);
        Ok(len)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Any range with `usize` bounds.
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
        let slice = buf.filled();
        assert_eq!(slice.len(), 0);
        let mut cursor = slice.into_buf().unfilled();
        cursor.write_slice(&[6, 5, 4]);
        let v = unsafe { cursor.into_buf().into_vec() };
        assert_eq!(&*v, &[6, 5, 4]);
    }
}
