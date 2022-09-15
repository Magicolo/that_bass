use crate::{
    database::{Database, Inner, Key, Table},
    Datum, Error,
};
use parking_lot::{
    MappedRwLockReadGuard, MappedRwLockWriteGuard, RwLock, RwLockReadGuard,
    RwLockUpgradableReadGuard, RwLockWriteGuard,
};
use std::{
    any::{type_name, TypeId},
    cell::UnsafeCell,
    collections::{HashMap, HashSet, VecDeque},
    iter::from_generator,
    marker::PhantomData,
    mem::{forget, needs_drop, replace, size_of},
    ops::{Deref, DerefMut},
    ptr::{copy, drop_in_place, slice_from_raw_parts_mut, NonNull},
    slice::{from_raw_parts, from_raw_parts_mut, SliceIndex},
    sync::{
        atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering::*},
        Arc,
    },
};

pub struct Query<I: Item, F: Filter = ()> {
    inner: Arc<Inner>,
    index: usize,
    indices: HashMap<usize, usize>,
    states: Vec<(usize, I::State)>,
    queue: VecDeque<usize>,
    filter: F,
    _marker: PhantomData<fn(I)>,
}

pub trait Filter {
    fn filter(&mut self, table: &Table) -> bool;
}

pub struct Not<F: Filter>(F);

pub struct Has<D: Datum>(PhantomData<D>);

pub trait Item {
    type State: for<'a> At<'a>;
    fn initialize(table: &Table) -> Option<Self::State>;
}

pub struct Read<C>(usize, PhantomData<C>);

pub trait At<'a> {
    type State;
    type Chunk;
    type Item;

    fn try_get(&self, table: &Table) -> Option<Self::State>;
    fn get(&self, table: &Table) -> Self::State;
    unsafe fn chunk(state: &mut Self::State) -> Self::Chunk;
    unsafe fn item(state: &mut Self::State, index: usize) -> Self::Item;
}

impl<I: Item, F: Filter> Query<I, F> {
    // pub fn item(&mut self, key: Key) -> Option<Guard<<I::State as At>::Item>> {
    //     self.with(key, |item, table| Guard(item, table))
    // }

    pub fn item_with<T>(
        &mut self,
        key: Key,
        with: impl FnOnce(<I::State as At>::Item) -> T,
    ) -> Option<T> {
        self.with(key, |item, _| with(item))
    }

    pub fn items(&mut self) -> impl Iterator<Item = <I::State as At>::Item> {
        self.iterate().flat_map(|(mut state, table)| {
            (0..table.count()).map(move |i| unsafe { I::State::item(&mut state, i as usize) })
        })
    }

    pub fn items_with(&mut self, mut each: impl FnMut(<I::State as At>::Item)) {
        self.each(|mut state, table| {
            for i in 0..table.count() {
                each(unsafe { I::State::item(&mut state, i as usize) });
            }
        })
    }

    pub fn chunks(&mut self) -> impl Iterator<Item = <I::State as At>::Chunk> {
        self.iterate()
            .map(|(mut state, _)| unsafe { I::State::chunk(&mut state) })
    }

    pub fn chunks_with(&mut self, mut each: impl FnMut(<I::State as At>::Chunk)) {
        self.each(|mut state, _| each(unsafe { I::State::chunk(&mut state) }));
    }

    /// Ensure that all tables have been filtered or initialized.
    fn update(&mut self) {
        while let Some(table) = self.inner.tables.get(self.index) {
            let table_read = table.read();
            if self.filter.filter(&table_read) {
                if let Some(state) = I::initialize(&table_read) {
                    drop(table_read);
                    self.indices.insert(self.index, self.states.len());
                    self.states.push((self.index, state));
                } else {
                    drop(table_read);
                }
            } else {
                drop(table_read);
            }
            self.index += 1;
        }
    }

    fn with<T>(
        &mut self,
        key: Key,
        with: impl FnOnce(<I::State as At>::Item, RwLockReadGuard<Table>) -> T,
    ) -> Option<T> {
        Some(loop {
            self.update();
            let Self {
                inner,
                indices,
                states,
                ..
            } = &*self;
            let slot = self.inner.slot(key)?;
            let (table_index, store_index) = slot.indices();
            let &state_index = match indices.get(&(table_index as usize)) {
                Some(index) => index,
                None => return None,
            };
            let (_, item_state) = unsafe { states.get_unchecked(state_index) };
            let table = unsafe { inner.tables.get_unchecked(table_index as usize) };
            let table_read = table.read();
            if slot.indices() != (table_index, store_index) {
                continue;
            }

            match I::State::try_get(item_state, &table_read) {
                Some(mut at_state) => {
                    let item = unsafe { I::State::item(&mut at_state, store_index as _) };
                    break with(item, table_read);
                }
                None => {
                    drop(table_read);
                    // It is allowed that there be interleaving of other thread operations here as long as the
                    // `slot.indices` are checked each time a table lock is acquired.
                    let table_write = table.write();
                    if slot.indices() != (table_index, store_index) {
                        continue;
                    }

                    let mut at_state = I::State::get(item_state, &table_write);
                    let table_read = RwLockWriteGuard::downgrade(table_write);
                    let item = unsafe { I::State::item(&mut at_state, store_index as _) };
                    break with(item, table_read);
                }
            }
        })
    }

    fn iterate(
        &mut self,
    ) -> impl Iterator<Item = (<I::State as At>::State, RwLockReadGuard<Table>)> {
        from_generator(|| {
            self.update();

            // Try to execute the query using only read locks on tables. This should succeed unless there is contention over
            // the store locks which would cause the `I::State::get` call to fail.
            for (state_index, (table_index, item_state)) in self.states.iter().enumerate() {
                if let Some(table) = self.inner.tables.get(*table_index) {
                    let table_read = table.read();
                    match I::State::try_get(item_state, &table_read) {
                        Some(at_state) => yield (at_state, table_read),
                        None => {
                            drop(table_read);
                            self.queue.push_back(state_index);
                        }
                    }
                }
            }

            // Try again to execute the tables that previously failed to take their store locks by still using only read
            // locks on tables hoping that there is no more contention.
            let mut count = self.queue.len();
            while let Some(state_index) = self.queue.pop_front() {
                if let Some((table_index, item_state)) = self.states.get(state_index) {
                    if let Some(table) = self.inner.tables.get(*table_index) {
                        let table_read = table.read();
                        match I::State::try_get(item_state, &table_read) {
                            Some(at_state) => {
                                count = self.queue.len();
                                yield (at_state, table_read);
                            }
                            None if count == 0 => {
                                drop(table_read);
                                // Since no table can make progress, escalate to a write lock.
                                let table_write = table.write();
                                let at_state = I::State::get(item_state, &table_write);
                                count = self.queue.len();
                                let table_read = RwLockWriteGuard::downgrade(table_write);
                                yield (at_state, table_read);
                            }
                            None => {
                                drop(table_read);
                                self.queue.push_back(state_index);
                                count -= 1;
                            }
                        }
                    }
                }
            }
        })
    }

    fn each(&mut self, mut each: impl FnMut(<I::State as At>::State, &Table)) {
        self.update();

        // Try to execute the query using only read locks on tables. This should succeed unless there is contention over
        // the store locks which would cause the `I::State::get` call to fail.
        for (state_index, (table_index, item_state)) in self.states.iter().enumerate() {
            if let Some(table) = self.inner.tables.get(*table_index) {
                let table_read = table.read();
                match I::State::try_get(item_state, &table_read) {
                    Some(at_state) => {
                        each(at_state, &table_read);
                        drop(table_read);
                    }
                    None => {
                        drop(table_read);
                        self.queue.push_back(state_index);
                    }
                }
            }
        }

        // Try again to execute the tables that previously failed to take their store locks by still using only read
        // locks on tables hoping that there is no more contention.
        let mut count = self.queue.len();
        while let Some(state_index) = self.queue.pop_front() {
            if let Some((table_index, item_state)) = self.states.get(state_index) {
                if let Some(table) = self.inner.tables.get(*table_index) {
                    let table_read = table.read();
                    match I::State::try_get(item_state, &table_read) {
                        Some(at_state) => {
                            each(at_state, &table_read);
                            drop(table);
                            count = self.queue.len();
                        }
                        None if count == 0 => {
                            drop(table_read);
                            // Since no table can make progress, escalate to a write lock.
                            let table_write = table.write();
                            let at_state = I::State::get(item_state, &table_write);
                            let table_read = RwLockWriteGuard::downgrade(table_write);
                            each(at_state, &table_read);
                            drop(table_read);
                            count = self.queue.len();
                        }
                        None => {
                            drop(table_read);
                            self.queue.push_back(state_index);
                            count -= 1;
                        }
                    }
                }
            }
        }
    }
}

impl Filter for () {
    fn filter(&mut self, table: &Table) -> bool {
        todo!()
    }
}

impl<F: Filter> Filter for Not<F> {
    fn filter(&mut self, table: &Table) -> bool {
        !self.0.filter(table)
    }
}

impl<D: Datum> Filter for Has<D> {
    fn filter(&mut self, table: &Table) -> bool {
        table.index(TypeId::of::<D>()).is_some()
    }
}

impl<F: FnMut(&Table) -> bool> Filter for F {
    fn filter(&mut self, table: &Table) -> bool {
        self(table)
    }
}

impl<D: Datum> Item for &D {
    type State = Read<D>;

    fn initialize(table: &Table) -> Option<Self::State> {
        Some(Read(table.index(TypeId::of::<D>())?, PhantomData))
    }
}

impl<'a, D: Datum> At<'a> for Read<D> {
    type State = MappedRwLockReadGuard<'a, [D]>;
    type Chunk = &'a [D];
    type Item = &'a D;

    #[inline]
    fn try_get(&self, table: &Table) -> Option<Self::State> {
        todo!()
        // unsafe { from_raw_parts(state.cast::<C>().as_ptr(), *count) }
        // Some((stores[self.0].data.try_read()?, count))
    }

    #[inline]
    fn get(&self, table: &Table) -> Self::State {
        todo!()
        // unsafe { from_raw_parts(state.cast::<C>().as_ptr(), *count) }
        // Some((stores[self.0].data.try_read()?, count))
    }

    #[inline]
    unsafe fn chunk(state: &mut Self::State) -> Self::Chunk {
        todo!()
        // state.as_ref()
    }

    #[inline]
    unsafe fn item(state: &mut Self::State, index: usize) -> Self::Item {
        todo!()
        // unsafe { state.get_unchecked(index) }
    }
}

impl Item for Key {
    type State = Self;

    fn initialize(table: &Table) -> Option<Self::State> {
        todo!()
    }
}

impl<'a> At<'a> for Key {
    type State = MappedRwLockReadGuard<'a, [Key]>;
    type Chunk = &'a [Key];
    type Item = Key;

    fn try_get(&self, table: &Table) -> Option<Self::State> {
        todo!()
    }

    fn get(&self, table: &Table) -> Self::State {
        todo!()
    }

    unsafe fn chunk(state: &mut Self::State) -> Self::Chunk {
        todo!()
    }

    unsafe fn item(state: &mut Self::State, index: usize) -> Self::Item {
        todo!()
    }
}

impl Database {
    pub fn query<I: Item>(&self) -> Result<Query<I>, Error> {
        // TODO: Fail when an invalid query is detected (ex: `Query<(&mut Position, &mut Position)>`).
        todo!()
    }

    pub fn query_with<I: Item, F: Filter>(&self, filter: F) -> Result<Query<I, F>, Error> {
        // TODO: Fail when an invalid query is detected (ex: `Query<(&mut Position, &mut Position)>`).
        todo!()
    }
}
