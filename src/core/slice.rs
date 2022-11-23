use parking_lot::Mutex;
use std::{
    alloc::{alloc, dealloc, Layout},
    collections::BTreeMap,
    marker::PhantomData,
    mem::needs_drop,
    ptr::drop_in_place,
    slice::from_raw_parts,
    sync::atomic::{AtomicPtr, Ordering},
};

pub struct Slice<T> {
    data: AtomicPtr<u8>,
    offset: usize,
    count: Mutex<usize>,
    _marker: PhantomData<[T]>,
}

pub struct Guard<'a, T>(&'a Slice<T>, *mut u8);

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key(*mut u8);
unsafe impl Send for Key {}
unsafe impl<T: Send> Send for Guard<'_, T> {}
unsafe impl<T: Sync> Sync for Guard<'_, T> {}

type Free = unsafe fn(*mut u8);
static STASH: Mutex<BTreeMap<Key, (usize, Free)>> = Mutex::new(BTreeMap::new());

impl<T: Clone + Send + Sync> Slice<T> {
    pub fn new(items: &[T]) -> Self {
        let (data, offset) = unsafe { allocate_with(items, &[]) };
        Slice {
            data: AtomicPtr::new(data),
            offset,
            count: 0.into(),
            _marker: PhantomData,
        }
    }

    pub fn guard(&self) -> Guard<T> {
        let mut count = self.count.lock();
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

    pub fn update(&mut self) {
        let data = self.0.data.load(Ordering::Relaxed);
        if self.1 == data {
            return;
        }
        // The pointer has changed since last read.
        let key = Key(self.1);
        let mut stash = STASH.lock();
        if let Some((count, free)) = stash.get_mut(&key) {
            debug_assert!(*count > 0);
            if *count == 1 {
                unsafe { free(key.0) };
                stash.remove(&key);
            } else {
                *count -= 1;
            }
        }
        self.1 = data;
    }
}

impl<T: Clone> Guard<'_, T> {
    pub fn truncate(&mut self, len: usize) -> &[T] {
        let guards = self.0.count.lock();
        // Call `self.get` under the lock such that the data pointer is not modified while truncating.
        self.update();
        let slice = unsafe { self.get_unchecked() };
        if slice.len() > len {
            let (new, offset) = unsafe { allocate_with(&slice[..len], &[]) };
            let old = self.0.data.swap(new, Ordering::Relaxed);
            debug_assert_eq!(offset, self.0.offset);
            debug_assert!(*guards > 0);
            if *guards == 1 {
                unsafe { free::<T>(old) };
            } else {
                STASH.lock().insert(Key(old), (*guards - 1, free::<T>));
            }
            drop(slice);
            self.1 = new;
        }
        unsafe { self.get_unchecked() }
    }

    pub fn extend<I: IntoIterator<Item = T>>(&mut self, items: I) -> &[T] {
        let items: Box<[T]> = items.into_iter().collect();
        self.try_extend_with(|_| Some(items))
    }

    pub fn extend_with<F: FnOnce(&[T]) -> Box<[T]>>(&mut self, with: F) -> &[T] {
        self.try_extend_with(|items| Some(with(items)))
    }

    pub fn try_extend_with<F: FnOnce(&[T]) -> Option<Box<[T]>>>(&mut self, with: F) -> &[T] {
        let count = self.0.count.lock();
        self.update();
        let slice = unsafe { self.get_unchecked() };
        if let Some(items) = with(slice) {
            if items.len() > 0 {
                let (new, offset) = unsafe { allocate_with(slice, &items) };
                let old = self.0.data.swap(new, Ordering::Relaxed);
                debug_assert_eq!(offset, self.0.offset);
                debug_assert!(*count > 0);
                if *count == 1 {
                    unsafe { free::<T>(old) };
                } else {
                    let value = STASH.lock().insert(Key(old), (*count - 1, free::<T>));
                    debug_assert!(value.is_none());
                }
                self.1 = new;
            }
        }
        unsafe { self.get_unchecked() }
    }

    pub fn push_with<F: FnOnce(&[T]) -> Option<T>>(&mut self, with: F) -> &[T] {
        let count = self.0.count.lock();
        self.update();
        let slice = unsafe { self.get_unchecked() };
        if let Some(item) = with(slice) {
            let (new, offset) = unsafe { allocate_with(slice, &[item]) };
            let old = self.0.data.swap(new, Ordering::Relaxed);
            debug_assert_eq!(offset, self.0.offset);
            debug_assert!(*count > 0);
            if *count == 1 {
                unsafe { free::<T>(old) };
            } else {
                let value = STASH.lock().insert(Key(old), (*count - 1, free::<T>));
                debug_assert!(value.is_none());
            }
            self.1 = new;
        }
        unsafe { self.get_unchecked() }
    }
}

impl<T: Clone + Send + Sync> Clone for Guard<'_, T> {
    fn clone(&self) -> Self {
        self.0.guard()
    }
}

impl<T> Drop for Guard<'_, T> {
    fn drop(&mut self) {
        let mut count = self.0.count.lock();
        debug_assert!(*count > 0);
        *count -= 1;
        // Load pointer under the lock to ensure that it is not modified at the same time.
        self.update();
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
