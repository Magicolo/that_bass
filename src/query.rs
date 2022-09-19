use parking_lot::{MappedRwLockReadGuard, MappedRwLockWriteGuard};

use crate::{
    database::{Database, TableRead},
    key::Key,
    table::{Store, Table},
    Datum, Error,
};
use std::{
    any::TypeId,
    collections::{HashMap, HashSet, VecDeque},
    iter::from_generator,
    marker::PhantomData,
    mem::transmute,
    ops::{Deref, DerefMut},
    sync::Arc,
};

pub struct Query<'a, I: Item, F: Filter = ()> {
    database: &'a Database,
    index: usize,
    indices: HashMap<usize, usize>,
    states: Vec<(Arc<Table>, I::State)>,
    queue: VecDeque<usize>,
    filter: F,
    _marker: PhantomData<fn(I)>,
}

pub struct Context<'a> {
    read: &'a mut HashSet<TypeId>,
    write: &'a mut HashSet<TypeId>,
}

pub trait Filter {
    fn filter(&mut self, table: &Table) -> bool;
}

pub struct Not<F: Filter>(F);

pub struct Has<D: Datum>(PhantomData<D>);

pub unsafe trait Item {
    type State: for<'a> At<'a>;
    fn initialize(table: &Table) -> Option<Self::State>;
    fn validate(context: Context) -> Result<(), Error>;
}

pub struct Read<T>(usize, PhantomData<T>);
pub struct Write<T>(usize, PhantomData<T>);

pub trait At<'a> {
    type State;
    type Chunk;
    type Item;

    fn try_get(&self, keys: &[Key], stores: &[Store]) -> Option<Self::State>;
    fn get(&self, keys: &[Key], stores: &[Store]) -> Self::State;
    unsafe fn chunk(state: &mut Self::State) -> Self::Chunk;
    unsafe fn item(state: &mut Self::State, index: usize) -> Self::Item;
}

/// This trait exists solely for the purpose of helping the compiler reason about lifetimes.
trait With<'a, I: Item> {
    type Value;
    fn with(self, item: <I::State as At<'a>>::Item, table: TableRead<'a>) -> Self::Value;
}

impl Context<'_> {
    pub fn own(&mut self) -> Context {
        Context {
            read: self.read,
            write: self.write,
        }
    }

    pub fn read<T: 'static>(&mut self) -> Result<(), Error> {
        let identifier = TypeId::of::<T>();
        if self.write.contains(&identifier) {
            Err(Error::ReadWriteConflict)
        } else {
            self.read.insert(identifier);
            Ok(())
        }
    }

    pub fn write<T: 'static>(&mut self) -> Result<(), Error> {
        let identifier = TypeId::of::<T>();
        if self.read.contains(&identifier) {
            Err(Error::ReadWriteConflict)
        } else if self.write.insert(identifier) {
            Ok(())
        } else {
            Err(Error::WriteWriteConflict)
        }
    }
}

pub struct Guard<'a, T>(T, TableRead<'a>);

impl<T> Deref for Guard<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Guard<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, I: Item, F: Filter> Query<'a, I, F> {
    #[inline]
    pub fn item(&mut self, key: Key) -> Option<Guard<<I::State as At<'a>>::Item>> {
        struct WithItem;
        impl<'a, I: Item> With<'a, I> for WithItem {
            type Value = Guard<'a, <I::State as At<'a>>::Item>;
            #[inline]
            fn with(
                self,
                item: <<I as Item>::State as At<'a>>::Item,
                table: TableRead<'a>,
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
        struct WithItem<'a, F, T>(&'a Database, F, PhantomData<T>);
        impl<'a, I: Item, T, F: FnOnce(<I::State as At<'a>>::Item) -> T> With<'a, I>
            for WithItem<'a, F, T>
        {
            type Value = T;
            #[inline]
            fn with(
                self,
                item: <<I as Item>::State as At<'a>>::Item,
                _: TableRead<'a>,
            ) -> Self::Value {
                self.1(item)
            }
        }
        self.with(key, WithItem(self.database, with, PhantomData))
    }

    #[inline]
    pub fn items(&mut self) -> impl Iterator<Item = <I::State as At<'a>>::Item> + '_ {
        from_generator(|| {
            for (mut at_state, table_read) in self.iterate() {
                for i in 0..table_read.inner().count() {
                    yield unsafe { I::State::item(&mut at_state, i as usize) };
                }
                drop(table_read);
            }
        })
    }

    #[inline]
    pub fn items_with(&mut self, mut each: impl FnMut(<I::State as At>::Item)) {
        for (mut at_state, table_read) in self.iterate() {
            for i in 0..table_read.inner().count() {
                each(unsafe { I::State::item(&mut at_state, i as usize) });
            }
            drop(table_read);
        }
    }

    #[inline]
    pub fn chunks(&mut self) -> impl Iterator<Item = <I::State as At<'a>>::Chunk> + '_ {
        from_generator(|| {
            for (mut at_state, table_read) in self.iterate() {
                yield unsafe { I::State::chunk(&mut at_state) };
                drop(table_read);
            }
        })
    }

    #[inline]
    pub fn chunks_with(&mut self, mut each: impl FnMut(<I::State as At>::Chunk)) {
        for (mut at_state, table_read) in self.iterate() {
            each(unsafe { I::State::chunk(&mut at_state) });
            drop(table_read);
        }
    }

    /// Ensure that all tables have been filtered or initialized.
    fn update(&mut self) {
        while let Some(table) = self.database.tables.get_shared(self.index) {
            if self.filter.filter(&table) {
                if let Some(state) = I::initialize(&table) {
                    self.indices.insert(self.index, self.states.len());
                    self.states.push((table, state));
                }
            }
            self.index += 1;
        }
    }

    fn with<W: With<'a, I>>(&mut self, key: Key, with: W) -> Option<W::Value> {
        Some(loop {
            self.update();
            let Self {
                database,
                indices,
                states,
                ..
            } = &*self;
            let slot = self.database.keys.get(key)?;
            let (table_index, store_index) = slot.indices();
            let &state_index = match indices.get(&(table_index as usize)) {
                Some(index) => index,
                None => return None,
            };
            let (_, item_state) = unsafe { states.get_unchecked(state_index) };
            let table = unsafe { database.tables.get_unchecked(table_index as usize) };
            let table_read = database.table_read(table);
            if slot.indices() != (table_index, store_index) {
                continue;
            }

            if let Some(mut at_state) =
                I::State::try_get(item_state, table_read.keys(), table_read.stores())
            {
                let item = unsafe { I::State::item(&mut at_state, store_index as _) };
                break with.with(item, table_read);
            }

            drop(table_read);
            // It is allowed that there be interleaving of other thread operations here as long as the
            // `slot.indices` are checked each time a table lock is acquired.
            let table_write = database.table_write(table);
            if slot.indices() != (table_index, store_index) {
                continue;
            }

            let mut at_state = I::State::get(item_state, table_write.keys(), table_write.stores());
            let table_read = table_write.downgrade();
            let item = unsafe { I::State::item(&mut at_state, store_index as _) };
            break with.with(item, table_read);
        })
    }

    #[inline]
    fn iterate(&mut self) -> impl Iterator<Item = (<I::State as At<'a>>::State, TableRead)> {
        from_generator(|| {
            self.update();

            // Try to execute the query using only read locks on tables. This should succeed unless there is contention over
            // the store locks which would cause the `I::State::get` call to fail.
            for (state_index, (table, item_state)) in self.states.iter().enumerate() {
                match self.database.table_try_read(table) {
                    Some(table_read) => {
                        match I::State::try_get(item_state, table_read.keys(), table_read.stores())
                        {
                            Some(at_state) => yield (at_state, table_read),
                            None => drop(table_read),
                        }
                    }
                    None => self.queue.push_back(state_index),
                }
            }

            // Try again to execute the tables that previously failed to take their store locks by still using only read
            // locks on tables hoping that there is no more contention.
            let mut count = self.queue.len();
            while let Some(state_index) = self.queue.pop_front() {
                debug_assert!(state_index < self.states.len());
                let (table, item_state) = unsafe { self.states.get_unchecked(state_index) };
                let table_read = self.database.table_read(table);
                match I::State::try_get(item_state, table_read.keys() as _, table_read.stores()) {
                    Some(at_state) => {
                        count = self.queue.len();
                        yield (at_state, table_read);
                    }
                    None if count == 0 => {
                        drop(table_read);
                        count = self.queue.len();
                        // Since no table can make progress, escalate to a write lock.
                        let table_write = self.database.table_write(table);
                        let at_state =
                            I::State::get(item_state, table_write.keys(), table_write.stores());
                        let table_read = table_write.downgrade();
                        yield (at_state, table_read);
                    }
                    None => {
                        drop(table_read);
                        self.queue.push_back(state_index);
                        count -= 1;
                    }
                }
            }
        })
    }
}

impl Filter for () {
    fn filter(&mut self, _: &Table) -> bool {
        true
    }
}

impl Filter for bool {
    fn filter(&mut self, _: &Table) -> bool {
        *self
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

unsafe impl<D: Datum> Item for &D {
    type State = Read<D>;

    fn initialize(table: &Table) -> Option<Self::State> {
        Some(Read(table.store(TypeId::of::<D>())?, PhantomData))
    }

    fn validate(mut context: Context) -> Result<(), Error> {
        context.read::<D>()
    }
}

impl<'a, D: Datum> At<'a> for Read<D> {
    type State = MappedRwLockReadGuard<'a, [D]>;
    type Chunk = &'a [D];
    type Item = &'a D;

    #[inline]
    fn try_get(&self, keys: &[Key], stores: &[Store]) -> Option<Self::State> {
        debug_assert!(self.0 < stores.len());
        let guard = unsafe {
            stores
                .get_unchecked(self.0)
                .try_read::<D, _>(.., keys.len())
        };
        // TODO: Fix this...
        Some(unsafe { transmute(guard?) })
    }

    #[inline]
    fn get(&self, keys: &[Key], stores: &[Store]) -> Self::State {
        debug_assert!(self.0 < stores.len());
        let guard = unsafe { stores.get_unchecked(self.0).read::<D, _>(.., keys.len()) };
        // TODO: Fix this...
        unsafe { transmute(guard) }
    }

    #[inline]
    unsafe fn chunk(state: &mut Self::State) -> Self::Chunk {
        &*(state as *mut Self::State)
    }

    #[inline]
    unsafe fn item(state: &mut Self::State, index: usize) -> Self::Item {
        Self::chunk(state).get_unchecked(index)
    }
}

unsafe impl<D: Datum> Item for &mut D {
    type State = Write<D>;

    fn initialize(table: &Table) -> Option<Self::State> {
        Some(Write(table.store(TypeId::of::<D>())?, PhantomData))
    }

    fn validate(mut context: Context) -> Result<(), Error> {
        context.write::<D>()
    }
}

impl<'a, D: Datum> At<'a> for Write<D> {
    type State = MappedRwLockWriteGuard<'a, [D]>;
    type Chunk = &'a mut [D];
    type Item = &'a mut D;

    #[inline]
    fn try_get(&self, keys: &[Key], stores: &[Store]) -> Option<Self::State> {
        debug_assert!(self.0 < stores.len());
        let guard = unsafe {
            stores
                .get_unchecked(self.0)
                .try_write::<D, _>(.., keys.len())
        };
        // TODO: Fix this...
        Some(unsafe { transmute(guard?) })
    }

    #[inline]
    fn get(&self, keys: &[Key], stores: &[Store]) -> Self::State {
        debug_assert!(self.0 < stores.len());
        let guard = unsafe { stores.get_unchecked(self.0).write::<D, _>(.., keys.len()) };
        // TODO: Fix this...
        unsafe { transmute(guard) }
    }

    #[inline]
    unsafe fn chunk(state: &mut Self::State) -> Self::Chunk {
        &mut *(state as *mut Self::State)
    }

    #[inline]
    unsafe fn item(state: &mut Self::State, index: usize) -> Self::Item {
        Self::chunk(state).get_unchecked_mut(index)
    }
}

unsafe impl Item for () {
    type State = ();

    fn initialize(_: &Table) -> Option<Self::State> {
        Some(())
    }

    fn validate(_: Context) -> Result<(), Error> {
        Ok(())
    }
}

impl<'a> At<'a> for () {
    type State = ();
    type Chunk = ();
    type Item = ();

    #[inline]
    fn try_get(&self, _: &[Key], _: &[Store]) -> Option<Self::State> {
        Some(())
    }
    #[inline]
    fn get(&self, _: &[Key], _: &[Store]) -> Self::State {}
    #[inline]
    unsafe fn chunk(_: &mut Self::State) -> Self::Chunk {}
    #[inline]
    unsafe fn item(_: &mut Self::State, _: usize) -> Self::Item {}
}

unsafe impl<I1: Item, I2: Item> Item for (I1, I2) {
    type State = (I1::State, I2::State);

    fn initialize(table: &Table) -> Option<Self::State> {
        Some((I1::initialize(table)?, I2::initialize(table)?))
    }

    fn validate(mut context: Context) -> Result<(), Error> {
        I1::validate(context.own())?;
        I2::validate(context.own())?;
        Ok(())
    }
}

impl<'a, A1: At<'a>, A2: At<'a>> At<'a> for (A1, A2) {
    type State = (A1::State, A2::State);
    type Chunk = (A1::Chunk, A2::Chunk);
    type Item = (A1::Item, A2::Item);

    #[inline]
    fn try_get(&self, keys: &[Key], stores: &[Store]) -> Option<Self::State> {
        Some((self.0.try_get(keys, stores)?, self.1.try_get(keys, stores)?))
    }
    #[inline]
    fn get(&self, keys: &[Key], stores: &[Store]) -> Self::State {
        (self.0.get(keys, stores), self.1.get(keys, stores))
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
        self.query_with(())
    }

    pub fn query_with<I: Item, F: Filter>(&self, filter: F) -> Result<Query<I, F>, Error> {
        I::validate(Context {
            read: &mut HashSet::new(),
            write: &mut HashSet::new(),
        })?;
        Ok(Query {
            database: self,
            index: 0,
            indices: HashMap::new(),
            states: Vec::new(),
            queue: VecDeque::new(),
            filter,
            _marker: PhantomData,
        })
    }
}
