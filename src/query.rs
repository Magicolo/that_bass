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
    type State: for<'a> Lock<'a>;
    fn initialize(table: &Table) -> Option<Self::State>;
    fn validate(context: Context) -> Result<(), Error>;
}

pub trait Lock<'a> {
    type Guard;
    type Chunk;
    type Item;

    fn try_lock(&self, keys: &[Key], stores: &[Store]) -> Option<Self::Guard>;
    fn lock(&self, keys: &[Key], stores: &[Store]) -> Self::Guard;
    unsafe fn chunk(guard: &mut Self::Guard) -> Self::Chunk;
    unsafe fn item(guard: &mut Self::Guard, index: usize) -> Self::Item;
}

pub struct Read<T>(usize, PhantomData<T>);
pub struct Write<T>(usize, PhantomData<T>);

/// This trait exists solely for the purpose of helping the compiler reason about lifetimes.
trait With<'a, I: Item> {
    type Value;
    fn with(self, item: <I::State as Lock<'a>>::Item, table: TableRead<'a>) -> Self::Value;
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
    pub fn item(&mut self, key: Key) -> Option<Guard<<I::State as Lock<'a>>::Item>> {
        struct WithItem;
        impl<'a, I: Item> With<'a, I> for WithItem {
            type Value = Guard<'a, <I::State as Lock<'a>>::Item>;
            #[inline]
            fn with(
                self,
                item: <<I as Item>::State as Lock<'a>>::Item,
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
        with: impl FnOnce(<I::State as Lock>::Item) -> T,
    ) -> Option<T> {
        struct WithItem<'a, F, T>(&'a Database, F, PhantomData<T>);
        impl<'a, I: Item, T, F: FnOnce(<I::State as Lock<'a>>::Item) -> T> With<'a, I>
            for WithItem<'a, F, T>
        {
            type Value = T;
            #[inline]
            fn with(
                self,
                item: <<I as Item>::State as Lock<'a>>::Item,
                _: TableRead<'a>,
            ) -> Self::Value {
                self.1(item)
            }
        }
        self.with(key, WithItem(self.database, with, PhantomData))
    }

    #[inline]
    pub fn items(&mut self) -> impl Iterator<Item = <I::State as Lock<'a>>::Item> + '_ {
        from_generator(|| {
            for (mut guard, table_read) in self.iterate() {
                for i in 0..table_read.inner().count() {
                    yield unsafe { I::State::item(&mut guard, i as usize) };
                }
                drop(table_read);
            }
        })
    }

    #[inline]
    pub fn items_with(&mut self, mut each: impl FnMut(<I::State as Lock>::Item)) {
        for (mut guard, table_read) in self.iterate() {
            for i in 0..table_read.inner().count() {
                each(unsafe { I::State::item(&mut guard, i as usize) });
            }
            drop(table_read);
        }
    }

    #[inline]
    pub fn chunks(&mut self) -> impl Iterator<Item = <I::State as Lock<'a>>::Chunk> + '_ {
        from_generator(|| {
            for (mut guard, table_read) in self.iterate() {
                yield unsafe { I::State::chunk(&mut guard) };
                drop(table_read);
            }
        })
    }

    #[inline]
    pub fn chunks_with(&mut self, mut each: impl FnMut(<I::State as Lock>::Chunk)) {
        for (mut guard, table_read) in self.iterate() {
            each(unsafe { I::State::chunk(&mut guard) });
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

            if let Some(mut guard) =
                I::State::try_lock(item_state, table_read.keys(), table_read.stores())
            {
                let item = unsafe { I::State::item(&mut guard, store_index as _) };
                break with.with(item, table_read);
            }

            drop(table_read);
            // It is allowed that there be interleaving of other thread operations here as long as the
            // `slot.indices` are checked each time a table lock is acquired.
            let table_write = database.table_write(table);
            if slot.indices() != (table_index, store_index) {
                continue;
            }

            let mut guard = I::State::lock(item_state, table_write.keys(), table_write.stores());
            let table_read = table_write.downgrade();
            let item = unsafe { I::State::item(&mut guard, store_index as _) };
            break with.with(item, table_read);
        })
    }

    #[inline]
    fn iterate(&mut self) -> impl Iterator<Item = (<I::State as Lock<'a>>::Guard, TableRead)> {
        from_generator(|| {
            self.update();

            // Try to execute the query using only read locks on tables. This should succeed unless there is contention over
            // the store locks which would cause the `I::State::get` call to fail.
            for (state_index, (table, item_state)) in self.states.iter().enumerate() {
                if let Some(table_read) = self.database.table_try_read(table) {
                    match I::State::try_lock(item_state, table_read.keys(), table_read.stores()) {
                        Some(guard) => {
                            yield (guard, table_read);
                            continue;
                        }
                        None => drop(table_read),
                    }
                }
                self.queue.push_back(state_index);
            }

            // Try again to execute the tables that previously failed to take their store locks by still using only read
            // locks on tables hoping that there is no more contention.
            let mut count = self.queue.len();
            while let Some(state_index) = self.queue.pop_front() {
                debug_assert!(state_index < self.states.len());
                let (table, item_state) = unsafe { self.states.get_unchecked(state_index) };
                let table_read = self.database.table_read(table);
                match I::State::try_lock(item_state, table_read.keys() as _, table_read.stores()) {
                    Some(guard) => {
                        count = self.queue.len();
                        yield (guard, table_read);
                    }
                    None if count == 0 => {
                        drop(table_read);
                        count = self.queue.len();
                        // Since no table can make progress, escalate to a write lock.
                        let table_write = self.database.table_write(table);
                        let guard =
                            I::State::lock(item_state, table_write.keys(), table_write.stores());
                        let table_read = table_write.downgrade();
                        yield (guard, table_read);
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
    fn filter(&mut self, table: &Table) -> bool {
        true.filter(table)
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

impl<'a, D: Datum> Lock<'a> for Read<D> {
    type Guard = MappedRwLockReadGuard<'a, [D]>;
    type Chunk = &'a [D];
    type Item = &'a D;

    #[inline]
    fn try_lock(&self, keys: &[Key], stores: &[Store]) -> Option<Self::Guard> {
        debug_assert!(self.0 < stores.len());
        let store = unsafe { &*stores.as_ptr().add(self.0) };
        unsafe { store.try_read(.., keys.len()) }
    }

    #[inline]
    fn lock(&self, keys: &[Key], stores: &[Store]) -> Self::Guard {
        debug_assert!(self.0 < stores.len());
        let store = unsafe { &*stores.as_ptr().add(self.0) };
        unsafe { store.read(.., keys.len()) }
    }

    #[inline]
    unsafe fn chunk(guard: &mut Self::Guard) -> Self::Chunk {
        // TODO: Fix this...
        &*(guard as *mut Self::Guard)
    }

    #[inline]
    unsafe fn item(guard: &mut Self::Guard, index: usize) -> Self::Item {
        Self::chunk(guard).get_unchecked(index)
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

impl<'a, D: Datum> Lock<'a> for Write<D> {
    type Guard = MappedRwLockWriteGuard<'a, [D]>;
    type Chunk = &'a mut [D];
    type Item = &'a mut D;

    #[inline]
    fn try_lock(&self, keys: &[Key], stores: &[Store]) -> Option<Self::Guard> {
        debug_assert!(self.0 < stores.len());
        let store = unsafe { &*stores.as_ptr().add(self.0) };
        unsafe { store.try_write(.., keys.len()) }
    }

    #[inline]
    fn lock(&self, keys: &[Key], stores: &[Store]) -> Self::Guard {
        debug_assert!(self.0 < stores.len());
        let store = unsafe { &*stores.as_ptr().add(self.0) };
        unsafe { store.write(.., keys.len()) }
    }

    #[inline]
    unsafe fn chunk(guard: &mut Self::Guard) -> Self::Chunk {
        &mut *(guard as *mut Self::Guard)
    }

    #[inline]
    unsafe fn item(guard: &mut Self::Guard, index: usize) -> Self::Item {
        Self::chunk(guard).get_unchecked_mut(index)
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

impl<'a> Lock<'a> for () {
    type Guard = ();
    type Chunk = ();
    type Item = ();

    #[inline]
    fn try_lock(&self, _: &[Key], _: &[Store]) -> Option<Self::Guard> {
        Some(())
    }
    #[inline]
    fn lock(&self, _: &[Key], _: &[Store]) -> Self::Guard {}
    #[inline]
    unsafe fn chunk(_: &mut Self::Guard) -> Self::Chunk {}
    #[inline]
    unsafe fn item(_: &mut Self::Guard, _: usize) -> Self::Item {}
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

impl<'a, A1: Lock<'a>, A2: Lock<'a>> Lock<'a> for (A1, A2) {
    type Guard = (A1::Guard, A2::Guard);
    type Chunk = (A1::Chunk, A2::Chunk);
    type Item = (A1::Item, A2::Item);

    #[inline]
    fn try_lock(&self, keys: &[Key], stores: &[Store]) -> Option<Self::Guard> {
        Some((
            self.0.try_lock(keys, stores)?,
            self.1.try_lock(keys, stores)?,
        ))
    }
    #[inline]
    fn lock(&self, keys: &[Key], stores: &[Store]) -> Self::Guard {
        (self.0.lock(keys, stores), self.1.lock(keys, stores))
    }
    #[inline]
    unsafe fn chunk(guard: &mut Self::Guard) -> Self::Chunk {
        (A1::chunk(&mut guard.0), A2::chunk(&mut guard.1))
    }
    #[inline]
    unsafe fn item(guard: &mut Self::Guard, index: usize) -> Self::Item {
        (A1::item(&mut guard.0, index), A2::item(&mut guard.1, index))
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
