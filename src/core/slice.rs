use parking_lot::Mutex;
use std::{
    alloc::{alloc, dealloc, Layout},
    collections::BTreeMap,
    marker::PhantomData,
    mem::{needs_drop, replace},
    ptr::{self, drop_in_place},
    slice::from_raw_parts,
    sync::atomic::{AtomicPtr, Ordering},
};

pub struct Slice<T> {
    data: AtomicPtr<u8>,
    offset: usize,
    seen: Mutex<usize>,
    _marker: PhantomData<[T]>,
}

pub struct Guard<'a, T>(&'a Slice<T>, *mut u8);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Key(*mut u8);
unsafe impl Send for Key {}
unsafe impl<T: Send> Send for Guard<'_, T> {}
unsafe impl<T: Sync> Sync for Guard<'_, T> {}

static STASH: Mutex<BTreeMap<Key, usize>> = Mutex::new(BTreeMap::new());

impl<T: Clone + Send + Sync> Slice<T> {
    pub fn new(items: &[T]) -> Self {
        let (data, offset) = unsafe { allocate_with(items, &[]) };
        Slice {
            data: AtomicPtr::new(data),
            offset,
            seen: 0.into(),
            _marker: PhantomData,
        }
    }

    pub fn guard(&self) -> Guard<T> {
        let mut count = self.seen.lock();
        *count += 1;
        // Load pointer under the lock to ensure that it is not modified at the same time.
        Guard(self, self.data.load(Ordering::Relaxed).into())
    }
}

impl<'a, T> Guard<'a, T> {
    #[inline]
    pub fn get(&mut self) -> &[T] {
        self.update();
        unsafe { self.get_unchecked() }
    }

    #[inline]
    pub unsafe fn get_unchecked(&self) -> &[T] {
        unsafe {
            from_raw_parts(
                self.1.add(self.0.offset).cast::<T>(),
                self.1.cast::<usize>().read(),
            )
        }
    }

    pub fn update(&mut self) -> bool {
        let data = self.0.data.load(Ordering::Relaxed);
        if ptr::eq(self.1, data) {
            // Fast path.
            return false;
        }

        // The pointer has changed since last read.
        let mut seen = self.0.seen.lock();
        // Load the pointer again in case it changed between the `load` and `lock` above.
        let data = self.0.data.load(Ordering::Relaxed);
        *seen += 1;
        drop(seen);

        Self::try_free(self.1);
        self.1 = data;
        true
    }

    // Frees or decrements the count of the given pointer.
    fn try_free(data: *mut u8) {
        let key = Key(data);
        let mut stash = STASH.lock();
        match stash.get_mut(&key) {
            Some(count) if *count == 1 => {
                unsafe { free::<T>(key.0) };
                stash.remove(&key);
            }
            Some(count) => *count -= 1,
            None => unreachable!("A reader is holding an invalid pointer."),
        }
    }

    fn set(&mut self, old: *mut u8, new: *mut u8, seen: &mut usize) {
        debug_assert!(*seen > 0);
        if *seen == 1 {
            unsafe { free::<T>(old) };
        } else {
            let count = replace(seen, 1) - 1;
            let value = STASH.lock().insert(Key(old), count);
            debug_assert!(value.is_none());
        }
        self.1 = new;
    }
}

impl<T: Clone> Guard<'_, T> {
    pub fn truncate(&mut self, len: usize) -> &[T] {
        let mut seen = self.0.seen.lock();
        // Call `self.get` under the lock such that the data pointer is not modified while truncating.
        self.update();
        let slice = unsafe { self.get_unchecked() };
        if slice.len() > len {
            let (new, offset) = unsafe { allocate_with(&slice[..len], &[]) };
            let old = self.0.data.swap(new, Ordering::Relaxed);
            debug_assert_eq!(offset, self.0.offset);
            self.set(old, new, &mut seen);
        }
        unsafe { self.get_unchecked() }
    }

    pub fn extend<I: IntoIterator<Item = T>>(&mut self, items: I) -> &[T] {
        let items: Box<[T]> = items.into_iter().collect();
        self.extend_with(|_| items)
    }

    pub fn extend_with<F: FnOnce(&[T]) -> Box<[T]>>(&mut self, with: F) -> &[T] {
        let mut seen = self.0.seen.lock();
        self.update();
        self.append(&with(unsafe { self.get_unchecked() }), &mut seen)
    }

    pub fn extend_filter<F: FnOnce(&[T]) -> Option<Box<[T]>>>(&mut self, with: F) -> Option<&[T]> {
        let mut seen = self.0.seen.lock();
        self.update();
        Some(self.append(&with(unsafe { self.get_unchecked() })?, &mut seen))
    }

    pub fn push_with<F: FnOnce(&[T]) -> T>(&mut self, with: F) -> &[T] {
        let mut seen = self.0.seen.lock();
        self.update();
        self.append(&[with(unsafe { self.get_unchecked() })], &mut seen)
    }

    pub fn push_if_same(&mut self, item: T) -> Option<&[T]> {
        let mut seen = self.0.seen.lock();
        if self.update() {
            None
        } else {
            Some(self.append(&[item], &mut seen))
        }
    }

    pub fn push_filter<F: FnOnce(&[T]) -> Option<T>>(&mut self, with: F) -> Option<&[T]> {
        let mut seen = self.0.seen.lock();
        self.update();
        Some(self.append(&[with(unsafe { self.get_unchecked() })?], &mut seen))
    }

    fn append(&mut self, items: &[T], seen: &mut usize) -> &[T] {
        if items.len() > 0 {
            let (new, offset) = unsafe { allocate_with(self.get_unchecked(), items) };
            let old = self.0.data.swap(new, Ordering::Relaxed);
            debug_assert_eq!(offset, self.0.offset);
            self.set(old, new, seen);
        }
        unsafe { self.get_unchecked() }
    }
}

impl<T: Clone + Send + Sync> Clone for Guard<'_, T> {
    fn clone(&self) -> Self {
        self.0.guard()
    }
}

impl<T> Drop for Slice<T> {
    fn drop(&mut self) {
        debug_assert_eq!(*self.seen.get_mut(), 0);
        // This guard is the last one. Free the pointer.
        unsafe { free::<T>(*self.data.get_mut()) };
    }
}

impl<T> Drop for Guard<'_, T> {
    fn drop(&mut self) {
        let mut seen = self.0.seen.lock();
        let data = self.0.data.load(Ordering::Relaxed);
        if ptr::eq(self.1, data) {
            *seen -= 1;
        } else {
            drop(seen);
            Self::try_free(self.1);
        }
    }
}

fn layout<T>(len: usize) -> (Layout, usize) {
    Layout::new::<usize>()
        .extend(Layout::array::<T>(len).unwrap())
        .unwrap()
}

unsafe fn allocate<T>(len: usize, initialize: impl FnOnce(*mut T)) -> (*mut u8, usize) {
    let (layout, offset) = layout::<T>(len);
    let data = alloc(layout);
    data.cast::<usize>().write(len);
    initialize(data.add(offset).cast::<T>());
    (data, offset)
}

unsafe fn allocate_with<T: Clone>(left: &[T], right: &[T]) -> (*mut u8, usize) {
    // TODO: Deal with `panic` in a `Clone` implementation. Will need to wrap these operations in a `impl Drop` structure.
    allocate::<T>(left.len() + right.len(), |data| {
        for (i, item) in left.iter().enumerate() {
            data.add(i).write(item.clone());
        }
        for (i, item) in right.iter().enumerate() {
            data.add(left.len() + i).write(item.clone());
        }
    })
}

unsafe fn free<T>(data: *mut u8) {
    let len = data.cast::<usize>().read();
    let (layout, offset) = layout::<T>(len);
    if needs_drop::<T>() {
        let slice = data.add(offset).cast::<T>();
        for i in 0..len {
            drop_in_place(slice.add(i));
        }
    }
    dealloc(data, layout);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// When testing the `STASH`, this global lock forces tests to run sequentially.
    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn stash_stays_empty() {
        let _lock = LOCK.lock();
        let slice = Slice::new(&[1]);
        let mut a = slice.guard();
        assert_eq!(STASH.lock().len(), 0);
        assert_eq!(a.push_with(|_| 2), &[1, 2]);
        assert_eq!(STASH.lock().len(), 0);
        assert_eq!(a.push_with(|_| 3), &[1, 2, 3]);
        assert_eq!(STASH.lock().len(), 0);
        drop(a);
        assert_eq!(STASH.lock().len(), 0);
    }

    #[test]
    fn stash_is_properly_emptied_with_2_guards() {
        let _lock = LOCK.lock();
        let slice = Slice::new(&[1]);
        let mut a = slice.guard();
        let b = slice.guard();
        assert_eq!(STASH.lock().len(), 0);
        assert_eq!(a.push_with(|_| 2), &[1, 2]);
        assert_eq!(STASH.lock().len(), 1);
        assert_eq!(a.push_with(|_| 3), &[1, 2, 3]);
        assert_eq!(STASH.lock().len(), 1);
        drop(b);
        assert_eq!(STASH.lock().len(), 0);
    }

    #[test]
    fn get_updates_the_slice() {
        let _lock = LOCK.lock();
        let slice = Slice::new(&[1]);
        let mut a = slice.guard();
        let mut b = slice.guard();

        assert_eq!(STASH.lock().len(), 0);
        assert_eq!(unsafe { a.get_unchecked() }, &[1]);
        assert_eq!(unsafe { b.get_unchecked() }, &[1]);
        assert_eq!(a.get(), &[1]);
        assert_eq!(b.get(), &[1]);

        assert_eq!(a.push_with(|_| 2), &[1, 2]);
        assert_eq!(STASH.lock().len(), 1);
        assert_eq!(unsafe { a.get_unchecked() }, &[1, 2]);
        assert_eq!(unsafe { b.get_unchecked() }, &[1]);
        assert_eq!(STASH.lock().len(), 1);
        assert_eq!(a.get(), &[1, 2]);
        assert_eq!(b.get(), &[1, 2]);
        assert_eq!(STASH.lock().len(), 0);
    }

    #[test]
    fn stash_is_properly_emptied_with_3_guards() {
        let _lock = LOCK.lock();
        let slice = Slice::new(&[1]);
        let mut a = slice.guard();
        let mut b = slice.guard();
        let c = slice.guard();
        assert_eq!(a.push_with(|_| 2), &[1, 2]);
        assert_eq!(b.get(), &[1, 2]);
        assert_eq!(a.push_with(|_| 3), &[1, 2, 3]);
        assert_eq!(STASH.lock().len(), 2);
        assert_eq!(unsafe { a.get_unchecked() }, &[1, 2, 3]);
        assert_eq!(unsafe { b.get_unchecked() }, &[1, 2]);
        assert_eq!(unsafe { c.get_unchecked() }, &[1]);

        drop(a);
        assert_eq!(STASH.lock().len(), 2);
        drop(b);
        assert_eq!(STASH.lock().len(), 1);
        drop(c);
        assert_eq!(STASH.lock().len(), 0);
    }
}
