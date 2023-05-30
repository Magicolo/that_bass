#![feature(array_zip)]

use parking_lot::Mutex;
use rayon::prelude::*;
use std::{
    collections::HashSet,
    iter::FusedIterator,
    mem::{swap, transmute},
    num::NonZeroU32,
    sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering},
};

pub struct Store<T> {
    keys: Keys,
    reads: Vec<Option<T>>,
    writes: Vec<Option<T>>,
    inserts: Mutex<Vec<(Key, T)>>,
    removes: Mutex<HashSet<Key>>,
}

struct Keys {
    next: AtomicU64,
    entries: Vec<AtomicU64>,
    slots: (Vec<Slot>, AtomicU32),
    free: (Vec<Key>, AtomicI64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    generation: u32,
    index: u32,
}

pub enum Entry {
    Vacant(Key),
    Occupied(NonZeroU32),
}

#[derive(Default)]
pub struct Slot {
    generation: u32,
}

pub struct Defer<'a, T> {
    keys: &'a Keys,
    inserts: &'a Mutex<Vec<(Key, T)>>,
    removes: &'a Mutex<HashSet<Key>>,
}

pub struct Others<'a, T>(usize, &'a Keys, &'a [Option<T>]);

pub struct Inserts<'a, P: Iterator<Item = (Key, T)>, T> {
    inserts: P,
    keys: &'a mut Keys,
    reads: &'a mut [Option<T>],
}

pub struct Removes<'a, K: Iterator<Item = Key>, T> {
    removes: K,
    keys: &'a mut Keys,
    reads: &'a mut [Option<T>],
    writes: &'a mut [Option<T>],
}

impl Keys {
    pub const fn new() -> Self {
        Self {
            next: AtomicU64::new(0),
            entries: Vec::new(),
            slots: (Vec::new(), AtomicU32::new(0)),
            free: (Vec::new(), AtomicI64::new(0)),
        }
    }

    fn reserve_2(&self) -> Key {
        match self
            .next
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                let next = Key::from_u64(next);
                match self.entries.get(next.index as usize) {
                    Some(entry) => Some(entry.load(Ordering::Relaxed)),
                    None => Some(Key::into_u64(Key::new(0, next.index.checked_add(1)?))),
                }
            }) {
            Ok(next) => Key::from_u64(next),
            Err(_) => panic!("Too many entries were created."),
        }
    }

    fn release_2(&self, key: Key) -> bool {
        if let Some(key) = key.increment() {
            if let Some(entry) = self.entries.get(key.index as usize) {
                let key = Key::into_u64(key);
                return self
                    .next
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                        entry.store(next, Ordering::Relaxed);
                        Some(key)
                    })
                    .is_ok();
            }
        }
        false
    }

    fn reserve(&self, keys: &mut [Key]) {
        let tail = self.free.1.fetch_sub(keys.len() as _, Ordering::Relaxed);
        let keys = if tail > 0 {
            let tail = tail as usize;
            let free = tail.min(keys.len());
            keys[..free].copy_from_slice(&self.free.0[tail - free..tail]);
            &mut keys[free..]
        } else {
            keys
        };

        if !keys.is_empty() {
            let index = self.slots.1.fetch_add(keys.len() as _, Ordering::Relaxed);
            for (i, key) in keys.iter_mut().enumerate() {
                *key = Key::new(0, index.wrapping_add(i as _));
            }
        }
    }

    fn reserve_mut(&mut self, keys: &mut [Key]) -> usize {
        let mut count = *self.free.1.get_mut();
        let keys = if count > 0 {
            count -= keys.len() as i64;
            let tail = count as usize;
            let free = tail.min(keys.len());
            keys[..free].copy_from_slice(&self.free.0[tail - free..free]);
            &mut keys[free..]
        } else {
            keys
        };

        let capacity = if !keys.is_empty() {
            let capacity = self.slots.1.get_mut();
            *capacity = capacity.wrapping_add(keys.len() as _);
            let index = *capacity;
            for (i, key) in keys.iter_mut().enumerate() {
                *key = Key::new(0, index.wrapping_add(i as _));
            }
            *capacity as usize
        } else {
            *self.slots.1.get_mut() as usize
        };

        self.free.0.truncate(count.max(0) as _);
        *self.free.1.get_mut() = self.free.0.len() as _;
        capacity
    }

    fn release(&mut self, keys: impl IntoIterator<Item = Key>) -> &[Key] {
        let count = *self.free.1.get_mut();
        self.free.0.truncate(count.max(0) as _);
        let index = self.free.0.len();
        for key in keys {
            if let Some(slot) = self.slots.0.get_mut(key.index as usize) {
                if slot.generation == key.generation {
                    if let Some(generation) = slot.generation.checked_add(1) {
                        slot.generation = generation;
                    }
                    self.free.0.push(key);
                }
            }
        }
        *self.free.1.get_mut() = self.free.0.len() as _;
        &self.free.0[index..]
    }

    fn resolve(&mut self) {
        let count = *self.free.1.get_mut();
        self.free.0.truncate(count.max(0) as _);
        *self.free.1.get_mut() = self.free.0.len() as _;
    }
}

impl<A> Store<A> {
    pub fn new() -> Self {
        Self {
            keys: Keys::new(),
            reads: Vec::new(),
            writes: Vec::new(),
            inserts: Mutex::new(Vec::new()),
            removes: Mutex::new(HashSet::new()),
        }
    }
}

impl<A> Default for Store<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, P: Iterator<Item = (Key, T)>, T> Inserts<'a, P, T> {
    fn new(inserts: P, keys: &'a mut Keys, reads: &'a mut Vec<Option<T>>) -> Self {
        let capacity = *keys.slots.1.get_mut() as _;
        Store::ensure(capacity, keys, reads);
        Self {
            inserts,
            keys,
            reads,
        }
    }

    fn insert(&mut self, key: Key, value: T) -> Result<(), T> {
        let index = key.index as usize;
        match (self.keys.slots.0.get_mut(index), self.reads.get_mut(index)) {
            (Some(slot), Some(read @ None)) => {
                slot.generation = key.generation;
                *read = Some(value);
                Ok(())
            }
            _ => Err(value),
        }
    }
}

impl<'a, P: Iterator<Item = (Key, T)>, T> Iterator for Inserts<'a, P, T> {
    type Item = Result<(), T>;

    fn next(&mut self) -> Option<Self::Item> {
        let (key, value) = self.inserts.next()?;
        Some(self.insert(key, value))
    }
}

impl<'a, P: DoubleEndedIterator<Item = (Key, T)>, T> DoubleEndedIterator for Inserts<'a, P, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let (key, value) = self.inserts.next_back()?;
        Some(self.insert(key, value))
    }
}

impl<'a, P: ExactSizeIterator<Item = (Key, T)>, T> ExactSizeIterator for Inserts<'a, P, T> {
    fn len(&self) -> usize {
        self.inserts.len()
    }
}

impl<'a, P: FusedIterator<Item = (Key, T)>, T> FusedIterator for Inserts<'a, P, T> {}

impl<'a, K: Iterator<Item = Key>, T> Removes<'a, K, T> {
    fn new(
        removes: K,
        keys: &'a mut Keys,
        reads: &'a mut [Option<T>],
        writes: &'a mut [Option<T>],
    ) -> Self {
        let count = *keys.free.1.get_mut();
        keys.free.0.truncate(count.max(0) as _);
        Self {
            removes,
            keys,
            reads,
            writes,
        }
    }

    fn remove(&mut self, key: Key) -> Result<(Key, T), Key> {
        let index = key.index as usize;
        if let Some(slot) = self.keys.slots.0.get_mut(index) {
            if slot.generation == key.generation {
                if let Some(read) = self.reads.get_mut(index) {
                    if let Some(value) = read.take() {
                        if let Some(write @ Some(_)) = self.writes.get_mut(index) {
                            *write = None;
                        }
                        if let Some(key) = key.increment() {
                            self.keys.free.0.push(key);
                        }
                        return Ok((key, value));
                    }
                }
            }
        }
        Err(key)
    }
}

impl<'a, K: Iterator<Item = Key>, T> Iterator for Removes<'a, K, T> {
    type Item = Result<(Key, T), Key>;

    fn next(&mut self) -> Option<Self::Item> {
        let key = self.removes.next()?;
        Some(self.remove(key))
    }
}

impl<'a, K: DoubleEndedIterator<Item = Key>, T> DoubleEndedIterator for Removes<'a, K, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let key = self.removes.next_back()?;
        Some(self.remove(key))
    }
}

impl<'a, K: ExactSizeIterator<Item = Key>, T> ExactSizeIterator for Removes<'a, K, T> {
    fn len(&self) -> usize {
        self.removes.len()
    }
}

impl<'a, K: FusedIterator<Item = Key>, T> FusedIterator for Removes<'a, K, T> {}

impl<'a, K: Iterator<Item = Key>, T> Drop for Removes<'a, K, T> {
    fn drop(&mut self) {
        for _ in self.by_ref() {}
        *self.keys.free.1.get_mut() = self.keys.free.0.len() as _;
    }
}

impl Key {
    pub(crate) const ZERO: Key = Key::new(0, 0);

    pub(crate) const fn new(generation: u32, index: u32) -> Self {
        Self { generation, index }
    }

    pub const fn into_u64(self) -> u64 {
        unsafe { transmute(self) }
    }

    pub const fn from_u64(value: u64) -> Key {
        unsafe { transmute(value) }
    }

    #[inline]
    pub(crate) fn increment(self) -> Option<Key> {
        Some(Key::new(self.generation.checked_add(1)?, self.index))
    }
}

impl<T> Defer<'_, T> {
    pub fn insert(&self, value: T) -> Key {
        let mut keys = [Key::ZERO];
        self.keys.reserve(&mut keys);
        let key = keys[0];
        self.inserts.lock().push((key, value));
        key
    }

    pub fn insert_n<const N: usize>(&self, values: [T; N]) -> [Key; N] {
        let mut keys = [Key::ZERO; N];
        self.keys.reserve(&mut keys);
        self.inserts.lock().extend(keys.zip(values));
        keys
    }

    pub fn try_insert<P: IntoIterator<Item = (Key, T)>>(&self, pairs: P) {
        self.inserts.lock().extend(pairs)
    }

    pub fn remove<K: IntoIterator<Item = Key>>(&self, keys: K) {
        self.removes.lock().extend(keys);
    }
}

impl<T> Store<T> {
    pub fn len(&self) -> usize {
        self.reads.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, key: Key) -> Option<&T> {
        let index = key.index as usize;
        let slot = self.keys.slots.0.get(index)?;
        if slot.generation == key.generation {
            self.reads[index].as_ref()
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, key: Key) -> Option<&mut T> {
        let index = key.index as usize;
        let slot = self.keys.slots.0.get(index)?;
        if slot.generation == key.generation {
            self.reads[index].as_mut()
        } else {
            None
        }
    }

    pub fn insert(&mut self, value: T) -> (Key, &mut T) {
        let mut keys = [Key::ZERO];
        let capacity = self.keys.reserve_mut(&mut keys);
        Self::ensure(capacity, &mut self.keys, &mut self.reads);
        let key = keys[0];
        let slot = &mut self.keys.slots.0[key.index as usize];
        debug_assert!(key.generation > slot.generation);
        slot.generation = key.generation;
        (key, self.reads[key.index as usize].insert(value))
    }

    pub fn insert_n<const N: usize>(&mut self, values: [T; N]) -> [Key; N] {
        let mut keys = [Key::ZERO; N];
        let capacity = self.keys.reserve_mut(&mut keys);
        Self::ensure(capacity, &mut self.keys, &mut self.reads);
        for (key, value) in keys.zip(values) {
            let slot = &mut self.keys.slots.0[key.index as usize];
            debug_assert!(key.generation > slot.generation);
            slot.generation = key.generation;
            self.reads[key.index as usize] = Some(value);
        }
        keys
    }

    pub fn try_insert<P: IntoIterator<Item = (Key, T)>>(
        &mut self,
        pairs: P,
    ) -> Inserts<P::IntoIter, T> {
        Inserts::new(pairs.into_iter(), &mut self.keys, &mut self.reads)
    }

    pub fn remove<K: IntoIterator<Item = Key>>(&mut self, keys: K) -> Removes<K::IntoIter, T> {
        Removes::new(
            keys.into_iter(),
            &mut self.keys,
            &mut self.reads,
            &mut self.writes,
        )
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (Key, &T)> {
        self.reads.iter().enumerate().filter_map(|(i, read)| {
            let read = read.as_ref()?;
            let slot = &self.keys.slots.0[i];
            let key = Key::new(slot.generation, i as _);
            Some((key, read))
        })
    }

    pub fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = (Key, &mut T)> {
        self.reads.iter_mut().enumerate().filter_map(|(i, read)| {
            let read = read.as_mut()?;
            let slot = &self.keys.slots.0[i];
            let key = Key::new(slot.generation, i as _);
            Some((key, read))
        })
    }

    pub fn each<E: FnMut(Key, &T)>(&self, mut each: E) {
        self.iter().for_each(|(key, value)| each(key, value))
    }

    pub fn each_mut<E: FnMut(Key, &mut T)>(&mut self, mut each: E) {
        self.iter_mut().for_each(|(key, value)| each(key, value))
    }

    pub fn each_with<E: FnMut(Key, &T, Others<T>, Defer<T>)>(&mut self, mut each: E) {
        self.reads.iter().enumerate().for_each(|(i, read)| {
            if let Some((key, value, others, defer)) = Self::item_with(
                i,
                &self.keys,
                &self.reads,
                read,
                &self.inserts,
                &self.removes,
            ) {
                each(key, value, others, defer);
            }
        });
        Self::resolve_defer(
            &mut self.keys,
            &mut self.reads,
            &mut self.writes,
            &mut self.inserts,
            &mut self.removes,
        );
    }

    fn ensure(capacity: usize, keys: &mut Keys, reads: &mut Vec<Option<T>>) {
        keys.slots.0.resize_with(capacity, Slot::default);
        reads.resize_with(capacity, || None);
    }

    fn resolve_defer(
        keys: &mut Keys,
        reads: &mut Vec<Option<T>>,
        writes: &mut [Option<T>],
        inserts: &mut Mutex<Vec<(Key, T)>>,
        removes: &mut Mutex<HashSet<Key>>,
    ) {
        drop(Inserts::new(inserts.get_mut().drain(..), keys, reads));
        drop(Removes::new(removes.get_mut().drain(), keys, reads, writes));
    }

    fn item_with<'a>(
        index: usize,
        keys: &'a Keys,
        reads: &'a [Option<T>],
        read: &'a Option<T>,
        inserts: &'a Mutex<Vec<(Key, T)>>,
        removes: &'a Mutex<HashSet<Key>>,
    ) -> Option<(Key, &'a T, Others<'a, T>, Defer<'a, T>)> {
        let value = read.as_ref()?;
        let slot = &keys.slots.0[index];
        let key = Key::new(slot.generation, index as _);
        let reads = Others(index, keys, reads);
        let defer = Defer {
            keys,
            inserts,
            removes,
        };
        Some((key, value, reads, defer))
    }
}

impl<T: Clone> Store<T> {
    pub fn each_mut_with<E: FnMut(Key, &mut T, Others<T>, Defer<T>)>(&mut self, mut each: E) {
        self.writes.resize_with(self.reads.len(), || None);
        self.writes.iter_mut().enumerate().for_each(|(i, write)| {
            if let Some((key, value, others, defer)) = Self::item_mut_with(
                i,
                &self.keys,
                &self.reads,
                write,
                &self.inserts,
                &self.removes,
            ) {
                each(key, value, others, defer);
            }
        });
        swap(&mut self.reads, &mut self.writes);
        Self::resolve_defer(
            &mut self.keys,
            &mut self.reads,
            &mut self.writes,
            &mut self.inserts,
            &mut self.removes,
        );
    }

    fn item_mut_with<'a>(
        index: usize,
        keys: &'a Keys,
        reads: &'a [Option<T>],
        write: &'a mut Option<T>,
        inserts: &'a Mutex<Vec<(Key, T)>>,
        removes: &'a Mutex<HashSet<Key>>,
    ) -> Option<(Key, &'a mut T, Others<'a, T>, Defer<'a, T>)> {
        let read = reads[index].as_ref()?;
        let slot = &keys.slots.0[index];
        let key = Key::new(slot.generation, index as _);
        let value = write.insert(read.clone());
        let reads = Others(index, keys, reads);
        let defer = Defer {
            keys,
            inserts,
            removes,
        };
        Some((key, value, reads, defer))
    }
}

impl<T: Send + Sync> Store<T> {
    pub fn par_iter(&self) -> impl ParallelIterator<Item = (Key, &T)> {
        self.reads.par_iter().enumerate().filter_map(|(i, read)| {
            let read = read.as_ref()?;
            let slot = &self.keys.slots.0[i];
            let key = Key::new(slot.generation, i as _);
            Some((key, read))
        })
    }

    pub fn par_iter_mut(&mut self) -> impl ParallelIterator<Item = (Key, &mut T)> {
        self.reads
            .par_iter_mut()
            .enumerate()
            .filter_map(|(i, read)| {
                let read = read.as_mut()?;
                let slot = &self.keys.slots.0[i];
                let key = Key::new(slot.generation, i as _);
                Some((key, read))
            })
    }

    pub fn par_each<E: Fn(Key, &T) + Sync>(&self, each: E) {
        self.par_iter().for_each(|(key, value)| each(key, value))
    }

    pub fn par_each_with<E: Fn(Key, &T, Others<T>, Defer<T>) + Sync>(&mut self, each: E) {
        self.reads.par_iter().enumerate().for_each(|(i, read)| {
            if let Some((key, value, others, defer)) = Self::item_with(
                i,
                &self.keys,
                &self.reads,
                read,
                &self.inserts,
                &self.removes,
            ) {
                each(key, value, others, defer);
            }
        });
        Self::resolve_defer(
            &mut self.keys,
            &mut self.reads,
            &mut self.writes,
            &mut self.inserts,
            &mut self.removes,
        );
    }

    pub fn par_each_mut<E: Fn(Key, &mut T) + Sync>(&mut self, each: E) {
        self.par_iter_mut()
            .for_each(|(key, value)| each(key, value))
    }
}

impl<T: Clone + Send + Sync> Store<T> {
    pub fn par_each_mut_with<E: Fn(Key, &mut T, Others<T>, Defer<T>) + Sync>(&mut self, each: E) {
        self.writes.resize_with(self.reads.len(), || None);
        self.writes
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, write)| {
                if let Some((key, value, others, defer)) = Self::item_mut_with(
                    i,
                    &self.keys,
                    &self.reads,
                    write,
                    &self.inserts,
                    &self.removes,
                ) {
                    each(key, value, others, defer);
                }
            });
        swap(&mut self.reads, &mut self.writes);
        Self::resolve_defer(
            &mut self.keys,
            &mut self.reads,
            &mut self.writes,
            &mut self.inserts,
            &mut self.removes,
        );
    }
}

impl<T> Others<'_, T> {
    pub fn get(&self, key: Key) -> Option<&T> {
        let index = key.index as usize;
        if self.0 == index {
            return None;
        }
        let slot = self.1.slots.0.get(index)?;
        if slot.generation == key.generation {
            self.2[index].as_ref()
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (Key, &T)> {
        self.2.iter().enumerate().filter_map(|(i, read)| {
            if self.0 == i {
                return None;
            }
            let read = read.as_ref()?;
            let slot = &self.1.slots.0[i];
            let key = Key::new(slot.generation, i as _);
            Some((key, read))
        })
    }
}

impl<T: Sync> Others<'_, T> {
    pub fn par_iter(&self) -> impl ParallelIterator<Item = (Key, &T)> {
        self.2.par_iter().enumerate().filter_map(|(i, read)| {
            if i == self.0 {
                return None;
            }
            let read = read.as_ref()?;
            let slot = &self.1.slots.0[i];
            let key = Key::new(slot.generation, i as _);
            Some((key, read))
        })
    }
}

#[test]
fn fett() {
    #[derive(Clone)]
    enum Entity {
        Player {
            position: [f64; 2],
            velocity: [f64; 2],
            target: Option<Key>,
        },
        Enemy {
            position: [f64; 2],
            velocity: [f64; 2],
        },
    }

    let mut store = Store::new();
    store.insert(Entity::Player {
        position: [0.0; 2],
        velocity: [1.0; 2],
        target: None,
    });
    store.insert(Entity::Enemy {
        position: [0.0; 2],
        velocity: [1.0; 2],
    });

    store.par_each_mut(|_, entity| match entity {
        Entity::Player {
            position, velocity, ..
        }
        | Entity::Enemy { position, velocity } => {
            position[0] += velocity[0];
            position[1] += velocity[1];
        }
    });

    store.par_each_mut_with(|_, entity, others, defer| {
        let Entity::Player { position: source_position, target, .. } = entity else { return; };
        let target_pair = others.par_iter().min_by_key(|(_, target)| match target {
            Entity::Enemy {
                position: target_position,
                ..
            } => Some((distance(source_position, target_position) * 1_000_000.0) as u64),
            _ => None,
        });
        let target_key = match target_pair {
            Some((key, _)) => key,
            None => defer.insert(Entity::Enemy {
                position: [0.0; 2],
                velocity: [0.0; 2],
            }),
        };
        *target = Some(target_key);
    });
}

fn distance(left: &[f64; 2], right: &[f64; 2]) -> f64 {
    let x = left[0] + right[0];
    let y = left[1] + right[1];
    (x * x + y * y).sqrt()
}
