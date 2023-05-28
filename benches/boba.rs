// use parking_lot::{Mutex, RwLock};
// use rayon::prelude::*;
// use std::{
//     mem::{replace, MaybeUninit},
//     num::NonZeroU32,
//     sync::{
//         atomic::{AtomicI64, AtomicU32, AtomicUsize, Ordering},
//         Arc,
//     },
// };
// use that_bass::core::{slice::Slice, utility::fold_swap};

// pub struct Store<T, const N: usize = 32> {
//     slots: (Vec<Slot>, AtomicU32),
//     // TODO: It may be possible to have a `Vec<u32>` instead of keeping the generation which is redundantly kept in the slot.
//     free: (Vec<Key>, AtomicI64),
//     chunks: Slice<Arc<Chunk<T, N>>>,
// }

// pub struct Chunk<T, const N: usize> {
//     count: AtomicUsize,
//     items: RwLock<[MaybeUninit<(Key, T)>; N]>,
// }

// pub struct Slot {
//     generation: u32,
//     indices: AtomicU32,
// }

// #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
// pub struct Key {
//     generation: NonZeroU32,
//     index: u32,
// }

// pub struct Defer<T> {
//     inserts: Mutex<Vec<(Key, T)>>,
//     removes: Mutex<Vec<Key>>,
//     sets: Mutex<Vec<(Key, Box<dyn FnOnce(Key, &mut T) + Send>)>>,
// }

// impl Key {
//     #[inline]
//     pub(crate) fn increment(self) -> Option<Key> {
//         Some(Key {
//             generation: self.generation.checked_add(1)?,
//             index: self.index,
//         })
//     }
// }

// impl<T> Defer<T> {
//     pub fn new() -> Self {
//         Self {
//             inserts: Mutex::new(Vec::new()),
//             removes: Mutex::new(Vec::new()),
//             sets: Mutex::new(Vec::new()),
//         }
//     }

//     pub fn insert(&self, key: Key, value: T) {
//         self.inserts.lock().push((key, value))
//     }

//     pub fn remove(&self, key: Key) {
//         self.removes.lock().push(key)
//     }

//     pub fn set<S: FnOnce(Key, &mut T) + Send + 'static>(&self, key: Key, set: S) {
//         self.sets.lock().push((key, Box::new(set)));
//     }

//     pub fn resolve(&mut self, store: &mut Store<T>) {
//         for (key, value) in self.inserts.get_mut().drain(..) {
//             store.insert_with(key, value);
//         }
//         for key in self.removes.get_mut().drain(..) {
//             store.remove(key);
//         }
//         for (key, set) in self.sets.get_mut().drain(..) {
//             if let Some(value) = store.get_mut(key) {
//                 set(key, value);
//             }
//         }
//     }
// }

// impl<T> Store<T> {
//     pub const fn new() -> Self {
//         Self {
//             slots: (Vec::new(), AtomicU32::new(0)),
//             free: (Vec::new(), AtomicI64::new(0)),
//             chunks: Slice::new(&[]),
//         }
//     }

//     pub fn reserve(&self) -> Key {
//         let tail = self.free.1.fetch_sub(1, Ordering::Relaxed);
//         if tail > 0 {
//             return self.free.0[tail as usize - 1];
//         }

//         let index = self.slots.1.fetch_add(1, Ordering::Relaxed);
//         assert!(index < u32::MAX);
//         Key {
//             generation: NonZeroU32::MIN,
//             index,
//         }
//     }

//     pub fn reserve_in<E: Extend<Key>>(&self, count: usize, keys: &mut E) {
//         if count == 0 {
//             return;
//         }

//         let tail = self.free.1.fetch_sub(count as _, Ordering::Relaxed);
//         let done = if tail > 0 {
//             let tail = tail as usize;
//             let count = tail.min(count);
//             let head = tail - count;
//             keys.extend(self.free.0[head..tail].iter().copied());
//             count
//         } else {
//             0
//         };

//         if done < count {
//             let count = count - done;
//             let head = self.slots.1.fetch_add(count as _, Ordering::Relaxed);
//             assert!(head <= u32::MAX - count as u32);
//             keys.extend((0..count).map(|i| Key {
//                 generation: NonZeroU32::MIN,
//                 index: head + i as u32,
//             }));
//         }
//     }

//     pub fn get(&self, key: Key) -> Option<&T> {
//         match self.slots.0.get(key.index as usize) {
//             Some(slot) if key.generation.get() > slot.generation => slot.value.as_ref(),
//             _ => None,
//         }
//     }

//     pub fn get_mut(&mut self, key: Key) -> Option<&mut T> {
//         match self.slots.0.get_mut(key.index as usize) {
//             Some(slot) if key.generation.get() > slot.generation => slot.value.as_mut(),
//             _ => None,
//         }
//     }

//     pub fn insert(&mut self, value: T) -> Key {
//         self.resolve_free();
//         let key = match self.free.0.pop() {
//             Some(key) => key,
//             None => {
//                 let capacity = self.slots.1.get_mut();
//                 let index = *capacity;
//                 assert!(index < u32::MAX);
//                 *capacity += 1;
//                 Key {
//                     generation: NonZeroU32::MIN,
//                     index,
//                 }
//             }
//         };
//         self.resolve_slots();
//         let slot = &mut self.slots.0[key.index as usize];
//         debug_assert!(key.generation.get() > slot.generation);
//         debug_assert!(slot.value.is_none());
//         slot.generation = key.generation.get();
//         slot.value = Some(value);
//         key
//     }

//     pub fn insert_with(&mut self, key: Key, value: T) -> Result<(), T> {
//         self.resolve_slots();
//         // TODO: Provide a more descriptive error.
//         match self.slots.0.get_mut(key.index as usize) {
//             Some(slot) if slot.value.is_none() && key.generation.get() > slot.generation => {
//                 slot.generation = key.generation.get();
//                 slot.value = Some(value);
//                 Ok(())
//             }
//             _ => Err(value),
//         }
//     }

//     pub fn remove(&mut self, key: Key) -> Option<T> {
//         match self.slots.0.get_mut(key.index as usize) {
//             Some(slot) if slot.value.is_some() && key.generation.get() == slot.generation => {
//                 if let Some(key) = key.increment() {
//                     self.free.0.truncate((*self.free.1.get_mut()).max(0) as _);
//                     self.free.0.push(key);
//                     *self.free.1.get_mut() = self.free.0.len() as _;
//                 }
//                 replace(&mut slot.value, None)
//             }
//             _ => None,
//         }
//     }

//     pub fn reads(&self) -> impl Iterator<Item = (Key, &T)> {
//         let mut guard = self.chunks.guard();
//         let chunks = guard.get();
//         let mut counts = chunks.iter().enumerate().map(|(index,chunk)| (index, chunk.count.load(Ordering::Relaxed))).collect::<Vec<_>>();
//         fold_swap(&mut counts, (), (), |_, _, _| Ok(()), |_, _, _| {});
//         [].into_iter()
//     }

//     pub fn iter(&self) -> impl Iterator<Item = (Key, &T)> {
//         self.chunks.guard().get().iter().map(|chunk| {
//             chunk.c
//         })
//         self.slots.0.iter().enumerate().filter_map(|(index, slot)| {
//             Some((
//                 Key {
//                     generation: NonZeroU32::new(slot.generation)?,
//                     index: index as _,
//                 },
//                 slot.value.as_ref()?,
//             ))
//         })
//     }

//     pub fn par_iter(&self) -> impl ParallelIterator<Item = (Key, &T)>
//     where
//         T: Sync,
//     {
//         self.slots
//             .0
//             .par_iter()
//             .enumerate()
//             .filter_map(|(index, slot)| {
//                 Some((
//                     Key {
//                         generation: NonZeroU32::new(slot.generation)?,
//                         index: index as _,
//                     },
//                     slot.value.as_ref()?,
//                 ))
//             })
//     }

//     pub fn iter_mut(&mut self) -> impl Iterator<Item = (Key, &mut T)> {
//         self.slots
//             .0
//             .iter_mut()
//             .enumerate()
//             .filter_map(|(index, slot)| {
//                 Some((
//                     Key {
//                         generation: NonZeroU32::new(slot.generation)?,
//                         index: index as _,
//                     },
//                     slot.value.as_mut()?,
//                 ))
//             })
//     }

//     pub fn par_iter_mut(&mut self) -> impl ParallelIterator<Item = (Key, &mut T)>
//     where
//         T: Sync + Send,
//     {
//         self.slots
//             .0
//             .par_iter_mut()
//             .enumerate()
//             .filter_map(|(index, slot)| {
//                 Some((
//                     Key {
//                         generation: NonZeroU32::new(slot.generation)?,
//                         index: index as _,
//                     },
//                     slot.value.as_mut()?,
//                 ))
//             })
//     }

//     // TODO
//     pub fn extend(&mut self) {}
//     // TODO
//     pub fn drain(&mut self) {}

//     fn resolve_slots(&mut self) {
//         let capacity = *self.slots.1.get_mut() as _;
//         self.slots.0.resize_with(capacity, || Slot {
//             generation: 0,
//             value: None,
//         });
//     }

//     fn resolve_free(&mut self) {
//         self.free.0.truncate((*self.free.1.get_mut()).max(0) as _);
//         *self.free.1.get_mut() = self.free.0.len() as _;
//     }
// }

// fn boba() {
//     enum Entity {
//         Player {
//             position: [f64; 2],
//             velocity: [f64; 2],
//             target: Option<Key>,
//             health: f64,
//             damage: f64,
//             range: f64,
//         },
//         Enemy {
//             position: [f64; 2],
//             velocity: [f64; 2],
//             health: f64,
//         },
//     }

//     struct Void;
//     impl<A> Extend<A> for Void {
//         fn extend<T: IntoIterator<Item = A>>(&mut self, _: T) {}
//     }

//     let mut store = Store::new();
//     let key = store.insert(Entity::Player {
//         position: [0.0, 0.0],
//         velocity: [0.0, 0.0],
//         target: None,
//         damage: 1.0,
//         health: 10.0,
//         range: 5.0,
//     });
//     store.remove(key);

//     let mut defer = Defer::new();
//     store.par_iter_mut().for_each(|(key, entity)| match entity {
//         Entity::Player { health, .. } | Entity::Enemy { health, .. } if *health <= 0.0 => {
//             defer.remove(key)
//         }
//         Entity::Player {
//             position, velocity, ..
//         }
//         | Entity::Enemy {
//             position, velocity, ..
//         } => {
//             position[0] += velocity[0];
//             position[1] += velocity[1];
//         }
//         _ => {}
//     });

//     fn distance(left: &[f64; 2], right: &[f64; 2]) -> f64 {
//         todo!()
//     }

//     store.par_iter().for_each(|source| {
//         let (source_key, Entity::Player { position: left, range, damage, .. }) = source else { return; };
//         let target = store.iter().min_by_key(|(_, target)| match target {
//             Entity::Enemy {
//                 position: right, ..
//             } => Some((distance(left, right) * 1_000_000.0) as u64),
//             _ => None,
//         });
//         let target_key = match target {
//             Some((target, _)) => target,
//             None => {
//                 let key = store.reserve();
//                 defer.insert(
//                     key,
//                     Entity::Enemy {
//                         position: [0.0; 2],
//                         velocity: [0.0; 2],
//                         health: 10.0,
//                     },
//                 );
//                 key
//             }
//         };
//         defer.set(source_key, move |_, source| {
//             if let Entity::Player { target, .. } = source {
//                 *target = Some(target_key);
//             }
//         });

//         if let Some(Entity::Enemy {
//             position: right,
//             ..
//         }) = store.get(target_key)
//         {
//             if distance(left, right) < *range {
//                 let damage = *damage;
//                 defer.set(target_key, move |_, target| {
//                     if let Entity::Enemy { health, .. } = target {
//                         *health -= damage;
//                     }
//                 })
//             }
//         }
//     });
//     defer.resolve(&mut store);
// }
