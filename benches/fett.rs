use parking_lot::Mutex;
use rayon::prelude::*;
use std::{
    collections::HashSet,
    mem::swap,
    num::NonZeroU32,
    sync::atomic::{AtomicI64, AtomicU32, Ordering},
};

pub struct Store<T> {
    keys: Keys,
    reads: Vec<Option<T>>,
    writes: Vec<Option<T>>,
    inserts: Mutex<Vec<(Key, T)>>,
    removes: Mutex<HashSet<Key>>,
}

struct Keys {
    slots: (Vec<Slot>, AtomicU32),
    free: (Vec<Key>, AtomicI64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    generation: NonZeroU32,
    index: u32,
}

pub struct Slot {
    generation: u32,
}

pub struct Defer<'a, T> {
    keys: &'a Keys,
    inserts: &'a Mutex<Vec<(Key, T)>>,
    removes: &'a Mutex<HashSet<Key>>,
}

impl Keys {
    pub const fn new() -> Self {
        Self {
            slots: (Vec::new(), AtomicU32::new(0)),
            free: (Vec::new(), AtomicI64::new(0)),
        }
    }

    fn reserve(&self) -> Key {
        let tail = self.free.1.fetch_sub(1, Ordering::Relaxed);
        if tail > 0 {
            return self.free.0[tail as usize - 1];
        }

        let index = self.slots.1.fetch_add(1, Ordering::Relaxed);
        assert!(index < u32::MAX);
        Key {
            generation: NonZeroU32::MIN,
            index,
        }
    }

    fn release(&mut self, keys: impl IntoIterator<Item = Key>) {
        let count = *self.free.1.get_mut();
        self.free.0.truncate(count.max(0) as _);
        self.free.0.extend(keys);
        *self.free.1.get_mut() = self.free.0.len() as _;
    }
}

impl<T> Store<T> {
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

impl<T> Default for Store<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Others<'a, T>(usize, &'a Keys, &'a [Option<T>]);

impl Key {
    pub(crate) const fn new(generation: NonZeroU32, index: u32) -> Self {
        Self { generation, index }
    }

    #[inline]
    pub(crate) fn increment(self) -> Option<Key> {
        Some(Key::new(self.generation.checked_add(1)?, self.index))
    }
}

impl<T> Defer<'_, T> {
    pub fn insert(&self, value: T) -> Key {
        let key = self.keys.reserve();
        self.inserts.lock().push((key, value));
        key
    }

    pub fn remove(&self, key: Key) {
        self.removes.lock().insert(key);
    }
}

impl<T> Store<T> {
    pub fn get(&self, key: Key) -> Option<&T> {
        let index = key.index as usize;
        let slot = self.keys.slots.0.get(index)?;
        if slot.generation == key.generation.get() {
            self.reads[index].as_ref()
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, key: Key) -> Option<&mut T> {
        let index = key.index as usize;
        let slot = self.keys.slots.0.get(index)?;
        if slot.generation == key.generation.get() {
            self.reads[index].as_mut()
        } else {
            None
        }
    }

    pub fn insert(&mut self, value: T) -> (Key, &mut T) {
        let key = self.next_key();
        let slot = &mut self.keys.slots.0[key.index as usize];
        debug_assert!(key.generation.get() > slot.generation);
        slot.generation = key.generation.get();
        let read = &mut self.reads[key.index as usize];
        debug_assert!(read.is_none());
        (key, read.insert(value))
    }

    pub fn insert_with(&mut self, key: Key, value: T) -> Result<&mut T, T> {
        Self::try_insert_with(
            key,
            value,
            &mut self.keys,
            &mut self.reads,
            &mut self.writes,
        )
    }

    pub fn remove(&mut self, key: Key) -> Option<T> {
        Self::try_remove(key, &mut self.keys, &mut self.reads, &mut self.writes)
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (Key, &T)> {
        self.reads.iter().enumerate().filter_map(|(i, read)| {
            let read = read.as_ref()?;
            let slot = &self.keys.slots.0[i];
            let key = Key::new(NonZeroU32::new(slot.generation)?, i as _);
            Some((key, read))
        })
    }

    pub fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = (Key, &mut T)> {
        self.reads.iter_mut().enumerate().filter_map(|(i, read)| {
            let read = read.as_mut()?;
            let slot = &self.keys.slots.0[i];
            let key = Key::new(NonZeroU32::new(slot.generation)?, i as _);
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

    fn try_insert_with<'a>(
        key: Key,
        value: T,
        keys: &mut Keys,
        reads: &'a mut Vec<Option<T>>,
        writes: &mut Vec<Option<T>>,
    ) -> Result<&'a mut T, T> {
        Self::resolve_slots(keys, reads, writes);
        match keys.slots.0.get_mut(key.index as usize) {
            Some(slot) if key.generation.get() == slot.generation => {
                let read = &mut reads[key.index as usize];
                debug_assert!(read.is_some());
                Ok(read.insert(value))
            }
            Some(slot) if slot.generation == 0 => {
                slot.generation = key.generation.get();
                let read = &mut reads[key.index as usize];
                debug_assert!(read.is_none());
                Ok(read.insert(value))
            }
            _ => Err(value),
        }
    }

    fn try_remove(
        key: Key,
        keys: &mut Keys,
        reads: &mut [Option<T>],
        writes: &mut [Option<T>],
    ) -> Option<T> {
        match keys.slots.0.get_mut(key.index as usize) {
            Some(slot) if key.generation.get() == slot.generation => {
                slot.generation = 0;
                writes[key.index as usize] = None;
                keys.release(key.increment());
                let read = &mut reads[key.index as usize];
                debug_assert!(read.is_some());
                read.take()
            }
            _ => None,
        }
    }

    fn next_key(&mut self) -> Key {
        self.resolve_free();
        if let Some(key) = self.keys.free.0.pop() {
            key
        } else {
            let capacity = self.keys.slots.1.get_mut();
            let index = *capacity;
            assert!(index < u32::MAX);
            *capacity += 1;
            Self::resolve_slots(&mut self.keys, &mut self.reads, &mut self.writes);
            Key::new(NonZeroU32::MIN, index as _)
        }
    }

    fn resolve_slots(keys: &mut Keys, reads: &mut Vec<Option<T>>, writes: &mut Vec<Option<T>>) {
        let capacity = *keys.slots.1.get_mut() as _;
        keys.slots
            .0
            .resize_with(capacity, || Slot { generation: 0 });
        reads.resize_with(capacity, || None);
        writes.resize_with(capacity, || None);
    }

    fn resolve_free(&mut self) {
        let count = *self.keys.free.1.get_mut();
        self.keys.free.0.truncate(count.max(0) as _);
        *self.keys.free.1.get_mut() = self.keys.free.0.len() as _;
    }

    fn resolve_defer(
        keys: &mut Keys,
        reads: &mut Vec<Option<T>>,
        writes: &mut Vec<Option<T>>,
        inserts: &mut Mutex<Vec<(Key, T)>>,
        removes: &mut Mutex<HashSet<Key>>,
    ) {
        for (key, value) in inserts.get_mut().drain(..) {
            let result = Self::try_insert_with(key, value, keys, reads, writes);
            debug_assert!(result.is_ok());
        }
        for key in removes.get_mut().drain() {
            Self::try_remove(key, keys, reads, writes);
        }
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
        let key = Key::new(NonZeroU32::new(slot.generation)?, index as _);
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
        let key = Key::new(NonZeroU32::new(slot.generation)?, index as _);
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
            let key = Key::new(NonZeroU32::new(slot.generation)?, i as _);
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
                let key = Key::new(NonZeroU32::new(slot.generation)?, i as _);
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
        if slot.generation == key.generation.get() {
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
            let key = Key::new(NonZeroU32::new(slot.generation)?, i as _);
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
            let key = Key::new(NonZeroU32::new(slot.generation)?, i as _);
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
