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

/// A concurrent contiguous collection designed for frequent reads and rare writes.
///
/// Access to the vector in done through [`View`]s which allow for concurrent read and write operations at the cost of consistency.
/// [`View`]s must manually call [`View::update`] when they want to synchronize with recent writes.
///
/// Write operations require the type `T` to be [`Clone`] and will reallocate the backing storage every time.
/// A typical usage pattern is to use [`ViewVec<Arc<T>>`] in place of a [`Mutex<Vec<Arc<T>>>`] to remove the requirement for locking.
pub struct ViewVec<T> {
    data: AtomicPtr<u8>,
    offset: usize, // TODO: This field is likely not required since the offset should be constant.
    seen: Mutex<usize>,
    _marker: PhantomData<[T]>,
}
lice
pub struct View<'a, T>(&'a ViewVec<T>, *mut u8);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Key(*mut u8);
unsafe impl Send for Key {}
unsafe impl<T: Send> Send for View<'_, T> {}
unsafe impl<T: Sync> Sync for View<'_, T> {}

static STASH: Mutex<BTreeMap<Key, usize>> = Mutex::new(BTreeMap::new());

impl<T: Clone> ViewVec<T> {
    pub fn new(items: &[T]) -> Self {
        let (data, offset) = unsafe { allocate_with(items.iter().cloned(), [].into_iter()) };
        ViewVec {
            data: AtomicPtr::new(data),
            offset,
            seen: 0.into(),
            _marker: PhantomData,
        }
    }

    pub fn view(&self) -> View<T> {
        let mut seen = self.seen.lock();
        *seen += 1;
        // Load pointer under the lock to ensure that it is not modified at the same time.
        let guard = View(self, self.data.load(Ordering::Relaxed));
        drop(seen);
        guard
    }
}

impl<'a, T> View<'a, T> {
    /// Returns the locally cached version of the slice at 0 cost (no indirection, locking or atomic operations).
    /// The returned slice may not be up to date with recent changes. Use [`Self::update`] to catch up on recent changes.
    #[inline]
    pub const fn get(&self) -> &[T] {
        // SAFETY:
        // The pointer held in `self.1` is guaranteed to be always valid for as long as this guard is alive. This is ensured
        // through the `seen` and `STASH` mechanisms.
        // Note that the pointer might point to data that is not up to date with recent changes.
        unsafe {
            from_raw_parts(
                self.1.add(self.0.offset).cast::<T>(),
                self.1.cast::<usize>().read(),
            )
        }
    }

    /// Updates the locally cached version of the slice at a synchronization cost.
    /// - If there are no changes to the slice, the cost is a single atomic load.
    /// - If there are changes to the slice, the cost is much larger.
    #[inline]
    pub fn update(&mut self) -> bool {
        self.update_locked(None)
    }

    /// Returns an updated version of the slice.
    /// This is equivalent to calling [`Self::update`] then [`Self::get`].
    #[inline]
    pub fn get_updated(&mut self) -> &[T] {
        self.update();
        self.get()
    }

    fn update_locked(&mut self, seen: Option<&mut usize>) -> bool {
        let Some(data) = self.load() else {
            return false;
        };
        let data = match seen {
            Some(seen) => {
                *seen += 1;
                data
            }
            None => {
                // The pointer has changed since last read.
                let mut seen = self.0.seen.lock();
                // Load the pointer again in case it changed between the `load` and `lock` above.
                let data = self.0.data.load(Ordering::Relaxed);
                *seen += 1;
                drop(seen);
                data
            }
        };
        Self::try_free(self.1);
        self.1 = data;
        true
    }

    fn load(&self) -> Option<*mut u8> {
        let data = self.0.data.load(Ordering::Relaxed);
        if ptr::eq(self.1, data) {
            None
        } else {
            Some(data)
        }
    }

    /// Frees or decrements the count of the given pointer.
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
        drop(stash);
    }

    fn set_locked(&mut self, old: *mut u8, new: *mut u8, seen: &mut usize) {
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

impl<T: Clone> View<'_, T> {
    pub fn truncate(&mut self, len: usize) -> &[T] {
        let mut seen = self.0.seen.lock();
        self.truncate_locked(len, &mut seen)
    }

    pub fn extend<I: IntoIterator<IntoIter = impl ExactSizeIterator<Item = T>>>(
        &mut self,
        items: I,
    ) -> &[T] {
        self.extend_locked(items, true, &mut self.0.seen.lock())
    }

    pub fn extend_with<
        I: IntoIterator<IntoIter = impl ExactSizeIterator<Item = T>>,
        F: FnOnce(&[T]) -> I,
    >(
        &mut self,
        with: F,
    ) -> &[T] {
        self.extend_locked(with(self.get()), true, &mut self.0.seen.lock())
    }

    pub fn push(&mut self, item: T) -> &[T] {
        self.extend_locked([item], true, &mut self.0.seen.lock())
    }

    pub fn push_with<F: FnOnce(&[T]) -> T>(&mut self, with: F) -> &[T] {
        self.extend_locked([with(self.get())], true, &mut self.0.seen.lock())
    }

    /// Pushes the `item` only if the slice wasn't modified since the last `update` of this `Guard`.
    /// Note that almost all operations on this `Guard` (including this one) will cause an `update`.
    pub fn push_if_same(&mut self, item: T) -> Option<&[T]> {
        let mut seen = self.0.seen.lock();
        if self.update_locked(Some(&mut seen)) {
            None
        } else {
            Some(self.extend_locked([item], false, &mut seen))
        }
    }

    pub fn push_filter<F: FnOnce(&[T]) -> Option<T>>(&mut self, with: F) -> Option<&[T]> {
        Some(self.extend_locked([with(self.get())?], true, &mut self.0.seen.lock()))
    }

    fn truncate_locked(&mut self, len: usize, seen: &mut usize) -> &[T] {
        self.update_locked(Some(seen));

        // Call `self.get_weak` under the lock such that the data pointer is not modified while truncating.
        let slice = self.get();
        if slice.len() > len {
            let (new, offset) =
                unsafe { allocate_with(slice[..len].iter().cloned(), [].into_iter()) };
            let old = self.0.data.swap(new, Ordering::Relaxed);
            debug_assert_eq!(offset, self.0.offset);
            self.set_locked(old, new, seen);
        }
        self.get()
    }

    fn extend_locked<I: IntoIterator<IntoIter = impl ExactSizeIterator<Item = T>>>(
        &mut self,
        items: I,
        update: bool,
        seen: &mut usize,
    ) -> &[T] {
        if update {
            self.update_locked(Some(seen));
        }

        let items = items.into_iter();
        if items.len() > 0 {
            let (new, offset) = unsafe { allocate_with(self.get().iter().cloned(), items) };
            let old = self.0.data.swap(new, Ordering::Relaxed);
            debug_assert_eq!(offset, self.0.offset);
            self.set_locked(old, new, seen);
        }

        self.get()
    }
}

impl<T: Clone + Send + Sync> Clone for View<'_, T> {
    fn clone(&self) -> Self {
        self.0.view()
    }
}

impl<T> Drop for ViewVec<T> {
    fn drop(&mut self) {
        debug_assert_eq!(*self.seen.get_mut(), 0);
        // This guard is the last one. Free the pointer.
        unsafe { free::<T>(*self.data.get_mut()) };
    }
}

impl<T> Drop for View<'_, T> {
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

unsafe fn allocate_with<L: ExactSizeIterator, R: ExactSizeIterator<Item = L::Item>>(
    left: L,
    right: R,
) -> (*mut u8, usize) {
    let left = left.into_iter();
    let right = right.into_iter();
    let counts = (left.len(), right.len());
    // TODO: Deal with `panic` in a `Iterator` implementation. Will need to wrap these operations in a `impl Drop` structure.
    allocate::<L::Item>(counts.0 + counts.1, |data| {
        for (i, item) in left.enumerate() {
            data.add(i).write(item);
        }
        for (i, item) in right.enumerate() {
            data.add(counts.0 + i).write(item);
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
        let slice = ViewVec::new(&[1]);
        let mut a = slice.view();
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
        let slice = ViewVec::new(&[1]);
        let mut a = slice.view();
        let b = slice.view();
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
        let slice = ViewVec::new(&[1]);
        let mut a = slice.view();
        let mut b = slice.view();

        assert_eq!(STASH.lock().len(), 0);
        assert_eq!(a.get(), &[1]);
        assert_eq!(b.get(), &[1]);
        assert_eq!(a.get_updated(), &[1]);
        assert_eq!(b.get_updated(), &[1]);

        assert_eq!(a.push_with(|_| 2), &[1, 2]);
        assert_eq!(STASH.lock().len(), 1);
        assert_eq!(a.get(), &[1, 2]);
        assert_eq!(b.get(), &[1]);
        assert_eq!(STASH.lock().len(), 1);
        assert_eq!(a.get_updated(), &[1, 2]);
        assert_eq!(b.get_updated(), &[1, 2]);
        assert_eq!(STASH.lock().len(), 0);
    }

    #[test]
    fn stash_is_properly_emptied_with_3_guards() {
        let _lock = LOCK.lock();
        let slice = ViewVec::new(&[1]);
        let mut a = slice.view();
        let mut b = slice.view();
        let c = slice.view();
        assert_eq!(a.push_with(|_| 2), &[1, 2]);
        assert_eq!(b.get_updated(), &[1, 2]);
        assert_eq!(a.push_with(|_| 3), &[1, 2, 3]);
        assert_eq!(STASH.lock().len(), 2);
        assert_eq!(a.get(), &[1, 2, 3]);
        assert_eq!(b.get(), &[1, 2]);
        assert_eq!(c.get(), &[1]);

        drop(a);
        assert_eq!(STASH.lock().len(), 2);
        drop(b);
        assert_eq!(STASH.lock().len(), 1);
        drop(c);
        assert_eq!(STASH.lock().len(), 0);
    }
}
