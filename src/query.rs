use crate::{
    database::{Database, Inner},
    key::Key,
    table::Table,
    Datum, Error,
};
use parking_lot::{RwLockReadGuard, RwLockWriteGuard};
use std::{
    any::TypeId,
    collections::{HashMap, VecDeque},
    marker::PhantomData,
    ops::{Deref, DerefMut},
    slice::{from_raw_parts, from_raw_parts_mut},
};

pub struct Query<'a, I: Item, F: Filter = ()> {
    inner: &'a Inner,
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

pub struct Read<T>(usize, PhantomData<T>);
pub struct Write<T>(usize, PhantomData<T>);

pub trait At<'a> {
    type State;
    type Chunk;
    type Item;

    fn try_get(&self, table: &Table) -> Option<Self::State>;
    fn get(&self, table: &Table) -> Self::State;
    unsafe fn chunk(state: &mut Self::State) -> Self::Chunk;
    unsafe fn item(state: &mut Self::State, index: usize) -> Self::Item;
}

/// This trait exists solely for the purpose of helping the compiler reason about lifetimes.
trait With<'a, I: Item> {
    type Value;
    fn with(
        self,
        item: <I::State as At<'a>>::Item,
        table: RwLockReadGuard<'a, Table>,
    ) -> Self::Value;
}

impl<'a, I: Item, F: Filter> Query<'a, I, F> {
    #[inline]
    pub fn item(&mut self, key: Key) -> Option<impl DerefMut<Target = <I::State as At<'a>>::Item>> {
        struct Guard<T, L>(T, L);
        struct WithItem;

        impl<T, L> Deref for Guard<T, L> {
            type Target = T;

            #[inline]
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl<T, L> DerefMut for Guard<T, L> {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl<'a, I: Item> With<'a, I> for WithItem {
            type Value = Guard<<I::State as At<'a>>::Item, RwLockReadGuard<'a, Table>>;
            fn with(
                self,
                item: <<I as Item>::State as At<'a>>::Item,
                table: RwLockReadGuard<'a, Table>,
            ) -> Self::Value {
                Guard(item, table)
            }
        }
        self.with(key, WithItem)
    }

    #[inline]
    pub fn item_with<T>(
        &mut self,
        key: Key,
        with: impl FnOnce(<I::State as At>::Item) -> T,
    ) -> Option<T> {
        struct WithItem<F, T>(F, PhantomData<T>);
        impl<'a, I: Item, T, F: FnOnce(<I::State as At<'a>>::Item) -> T> With<'a, I> for WithItem<F, T> {
            type Value = T;
            fn with(
                self,
                item: <<I as Item>::State as At<'a>>::Item,
                _: RwLockReadGuard<'a, Table>,
            ) -> Self::Value {
                self.0(item)
            }
        }
        self.with(key, WithItem(with, PhantomData))
    }

    // pub fn items(&mut self) -> impl Iterator<Item = <I::State as At>::Item> {
    //     self.iterate().flat_map(|(mut state, table)| {
    //         (0..table.count()).map(move |i| unsafe { I::State::item(&mut state, i as usize) })
    //     })
    // }

    #[inline]
    pub fn items_with(&mut self, mut each: impl FnMut(<I::State as At>::Item)) {
        self.each(|mut state, table| {
            for i in 0..table.count() {
                each(unsafe { I::State::item(&mut state, i as usize) });
            }
        })
    }

    // pub fn chunks(&mut self) -> impl Iterator<Item = <I::State as At>::Chunk> {
    //     self.iterate()
    //         .map(|(mut state, _)| unsafe { I::State::chunk(&mut state) })
    // }

    #[inline]
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

    fn with<W: With<'a, I>>(&mut self, key: Key, with: W) -> Option<W::Value> {
        Some(loop {
            self.update();
            let Self {
                inner,
                indices,
                states,
                ..
            } = &*self;
            let slot = self.inner.slots.get(key)?;
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

            if let Some(mut at_state) = I::State::try_get(item_state, &table_read) {
                let item = unsafe { I::State::item(&mut at_state, store_index as _) };
                break with.with(item, table_read);
            }

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
            break with.with(item, table_read);
        })
    }

    // fn iterate(
    //     &mut self,
    // ) -> impl Iterator<Item = (<I::State as At>::State, RwLockReadGuard<Table>)> {
    //     from_generator(|| {
    //         self.update();

    //         // Try to execute the query using only read locks on tables. This should succeed unless there is contention over
    //         // the store locks which would cause the `I::State::get` call to fail.
    //         for (state_index, (table_index, item_state)) in self.states.iter().enumerate() {
    //             if let Some(table) = self.inner.tables.get(*table_index) {
    //                 let table_read = table.read();
    //                 match I::State::try_get(item_state, &table_read) {
    //                     Some(at_state) => yield (at_state, table_read),
    //                     None => {
    //                         drop(table_read);
    //                         self.queue.push_back(state_index);
    //                     }
    //                 }
    //             }
    //         }

    //         // Try again to execute the tables that previously failed to take their store locks by still using only read
    //         // locks on tables hoping that there is no more contention.
    //         let mut count = self.queue.len();
    //         while let Some(state_index) = self.queue.pop_front() {
    //             if let Some((table_index, item_state)) = self.states.get(state_index) {
    //                 if let Some(table) = self.inner.tables.get(*table_index) {
    //                     let table_read = table.read();
    //                     match I::State::try_get(item_state, &table_read) {
    //                         Some(at_state) => {
    //                             count = self.queue.len();
    //                             yield (at_state, table_read);
    //                         }
    //                         None if count == 0 => {
    //                             drop(table_read);
    //                             // Since no table can make progress, escalate to a write lock.
    //                             let table_write = table.write();
    //                             let at_state = I::State::get(item_state, &table_write);
    //                             count = self.queue.len();
    //                             let table_read = RwLockWriteGuard::downgrade(table_write);
    //                             yield (at_state, table_read);
    //                         }
    //                         None => {
    //                             drop(table_read);
    //                             self.queue.push_back(state_index);
    //                             count -= 1;
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     })
    // }

    fn each(&mut self, mut each: impl FnMut(<I::State as At>::State, &Table)) {
        self.update();

        // Try to execute the query using only read locks on tables. This should succeed unless there is contention over
        // the store locks which would cause the `I::State::get` call to fail.
        for (state_index, (table_index, item_state)) in self.states.iter().enumerate() {
            if let Some(table) = self.inner.tables.get(*table_index) {
                let table_read = table.read();
                if let Some(at_state) = I::State::try_get(item_state, &table_read) {
                    each(at_state, &table_read);
                    continue;
                }
                drop(table_read);
                self.queue.push_back(state_index);
            }
        }

        // Try again to execute the tables that previously failed to take their store locks by still using only read
        // locks on tables hoping that there is no more contention.
        let mut count = self.queue.len();
        while let Some(state_index) = self.queue.pop_front() {
            if let Some((table_index, item_state)) = self.states.get(state_index) {
                if let Some(table) = self.inner.tables.get(*table_index) {
                    let table_read = table.read();
                    if let Some(at_state) = I::State::try_get(item_state, &table_read) {
                        each(at_state, &table_read);
                        drop(table);
                        count = self.queue.len();
                        continue;
                    }

                    if count == 0 {
                        drop(table_read);
                        // Since no table can make progress, escalate to a write lock.
                        let table_write = table.write();
                        let at_state = I::State::get(item_state, &table_write);
                        let table_read = RwLockWriteGuard::downgrade(table_write);
                        each(at_state, &table_read);
                        drop(table_read);
                        count = self.queue.len();
                        continue;
                    }

                    drop(table_read);
                    self.queue.push_back(state_index);
                    count -= 1;
                }
            }
        }
    }
}

impl Filter for () {
    fn filter(&mut self, _: &Table) -> bool {
        true
    }
}

impl<F: Filter> Filter for Not<F> {
    fn filter(&mut self, table: &Table) -> bool {
        !self.0.filter(table)
    }
}

impl<D: Datum> Filter for Has<D> {
    fn filter(&mut self, table: &Table) -> bool {
        table.has(TypeId::of::<D>())
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
        let (index, _) = table
            .stores()
            .iter()
            .enumerate()
            .find(|(_, store)| store.meta().identifier == TypeId::of::<D>())?;
        Some(Read(index, PhantomData))
    }
}

impl<'a, D: Datum> At<'a> for Read<D> {
    type State = (*const D, usize);
    type Chunk = &'a [D];
    type Item = &'a D;

    #[inline]
    fn try_get(&self, table: &Table) -> Option<Self::State> {
        todo!()
        // unsafe {
        //     table
        //         .stores()
        //         .get_unchecked(self.0)
        //         .try_read(.., table.count() as _)
        // }
    }

    #[inline]
    fn get(&self, table: &Table) -> Self::State {
        todo!()
        // unsafe {
        //     table
        //         .stores()
        //         .get_unchecked(self.0)
        //         .read(.., table.count() as _)
        // }
    }

    #[inline]
    unsafe fn chunk(state: &mut Self::State) -> Self::Chunk {
        from_raw_parts(state.0, state.1)
    }

    #[inline]
    unsafe fn item(state: &mut Self::State, index: usize) -> Self::Item {
        Self::chunk(state).get_unchecked(index)
    }
}

impl<D: Datum> Item for &mut D {
    type State = Write<D>;

    fn initialize(table: &Table) -> Option<Self::State> {
        let (index, _) = table
            .stores()
            .iter()
            .enumerate()
            .find(|(_, store)| store.meta().identifier == TypeId::of::<D>())?;
        Some(Write(index, PhantomData))
    }
}

impl<'a, D: Datum> At<'a> for Write<D> {
    type State = (*mut D, usize);
    type Chunk = &'a mut [D];
    type Item = &'a mut D;

    #[inline]
    fn try_get(&self, table: &Table) -> Option<Self::State> {
        todo!()
        // unsafe {
        //     table
        //         .stores()
        //         .get_unchecked(self.0)
        //         .try_read(.., table.count() as _)
        // }
    }

    #[inline]
    fn get(&self, table: &Table) -> Self::State {
        todo!()
        // unsafe {
        //     table
        //         .stores()
        //         .get_unchecked(self.0)
        //         .read(.., table.count() as _)
        // }
    }

    #[inline]
    unsafe fn chunk(state: &mut Self::State) -> Self::Chunk {
        from_raw_parts_mut(state.0, state.1)
    }

    #[inline]
    unsafe fn item(state: &mut Self::State, index: usize) -> Self::Item {
        Self::chunk(state).get_unchecked_mut(index)
    }
}

impl Item for () {
    type State = ();

    fn initialize(_: &Table) -> Option<Self::State> {
        Some(())
    }
}

impl<'a> At<'a> for () {
    type State = ();
    type Chunk = ();
    type Item = ();

    #[inline]
    fn try_get(&self, _: &Table) -> Option<Self::State> {
        Some(())
    }
    #[inline]
    fn get(&self, _: &Table) -> Self::State {}
    #[inline]
    unsafe fn chunk(_: &mut Self::State) -> Self::Chunk {}
    #[inline]
    unsafe fn item(_: &mut Self::State, _: usize) -> Self::Item {}
}

impl<I1: Item, I2: Item> Item for (I1, I2) {
    type State = (I1::State, I2::State);

    fn initialize(table: &Table) -> Option<Self::State> {
        Some((I1::initialize(table)?, I2::initialize(table)?))
    }
}

impl<'a, A1: At<'a>, A2: At<'a>> At<'a> for (A1, A2) {
    type State = (A1::State, A2::State);
    type Chunk = (A1::Chunk, A2::Chunk);
    type Item = (A1::Item, A2::Item);

    #[inline]
    fn try_get(&self, table: &Table) -> Option<Self::State> {
        Some((self.0.try_get(table)?, self.1.try_get(table)?))
    }
    #[inline]
    fn get(&self, table: &Table) -> Self::State {
        (self.0.get(table), self.1.get(table))
    }
    #[inline]
    unsafe fn chunk(state: &mut Self::State) -> Self::Chunk {
        (A1::chunk(&mut state.0), A2::chunk(&mut state.1))
    }
    #[inline]
    unsafe fn item(state: &mut Self::State, index: usize) -> Self::Item {
        (A1::item(&mut state.0, index), A2::item(&mut state.1, index))
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
