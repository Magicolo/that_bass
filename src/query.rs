use parking_lot::{MappedRwLockReadGuard, MappedRwLockWriteGuard};

use crate::{
    bits::Bits,
    database::{Database, TableRead},
    key::Key,
    resources::Local,
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
    states: Vec<(I::State, TableState)>,
    defer: VecDeque<usize>,
    skip: VecDeque<usize>,
    reads: HashSet<TypeId>,
    writes: HashSet<TypeId>,
    filter: F,
    _marker: PhantomData<I>,
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
pub struct Guard<'a, 'b, T>(T, TableGuard<'a, 'b>);

struct TableGuard<'a, 'b>(&'b TableState, TableRead<'a>);
#[derive(Hash, PartialEq, Eq)]
struct TableKey(u32);
struct TableLocks {
    readers: usize,
    locks: Bits,
    conflicts: Vec<(HashSet<TypeId>, HashSet<TypeId>, Bits)>,
}
struct TableState {
    table: Arc<Table>,
    locks: Local<TableLocks>,
    index: usize,
}

/// This trait exists solely for the purpose of helping the compiler reason about lifetimes.
trait With<'a, 'b, I: Item> {
    type Value;
    fn with(self, guard: Guard<'a, 'b, <I::State as Lock<'a>>::Item>) -> Self::Value;
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

impl<T> Deref for Guard<'_, '_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Guard<'_, '_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, I: Item, F: Filter> Query<'a, I, F> {
    #[inline]
    pub fn item(&mut self, key: Key) -> Result<Guard<<I::State as Lock<'a>>::Item>, Error> {
        struct WithItem;
        impl<'a, 'b, I: Item> With<'a, 'b, I> for WithItem {
            type Value = Guard<'a, 'b, <I::State as Lock<'a>>::Item>;
            #[inline]
            fn with(self, guard: Self::Value) -> Self::Value {
                guard
            }
        }
        self.with(key, WithItem)
    }

    #[inline]
    pub fn item_with<T>(
        &mut self,
        key: Key,
        with: impl FnOnce(<I::State as Lock>::Item) -> T,
    ) -> Result<T, Error> {
        struct WithItem<'a, F, T>(&'a Database, F, PhantomData<T>);
        impl<'a, 'b, I: Item, T, F: FnOnce(<I::State as Lock<'a>>::Item) -> T> With<'a, 'b, I>
            for WithItem<'a, F, T>
        {
            type Value = T;
            #[inline]
            fn with(self, guard: Guard<'a, 'b, <I::State as Lock<'a>>::Item>) -> Self::Value {
                self.1(guard.0)
            }
        }
        self.with(key, WithItem(self.database, with, PhantomData))
    }

    #[inline]
    pub fn items(&mut self) -> impl Iterator<Item = <I::State as Lock<'a>>::Item> + '_ {
        from_generator(|| {
            for mut guard in self.iterate() {
                for i in 0..guard.1 .1.inner().count() {
                    yield unsafe { I::State::item(&mut guard, i as usize) };
                }
                drop(guard);
            }
        })
    }

    #[inline]
    pub fn items_with(&mut self, mut each: impl FnMut(<I::State as Lock>::Item)) {
        for mut guard in self.iterate() {
            for i in 0..guard.1 .1.inner().count() {
                each(unsafe { I::State::item(&mut guard, i as usize) });
            }
            drop(guard);
        }
    }

    #[inline]
    pub fn chunks(&mut self) -> impl Iterator<Item = <I::State as Lock<'a>>::Chunk> + '_ {
        from_generator(|| {
            for mut guard in self.iterate() {
                yield unsafe { I::State::chunk(&mut guard) };
                drop(guard);
            }
        })
    }

    #[inline]
    pub fn chunks_with(&mut self, mut each: impl FnMut(<I::State as Lock>::Chunk)) {
        for mut guard in self.iterate() {
            each(unsafe { I::State::chunk(&mut guard) });
            drop(guard);
        }
    }

    /// Ensure that all tables have been filtered or initialized.
    fn update(&mut self) {
        while let Some(table) = self.database.tables.get_shared(self.index) {
            if self.filter.filter(&table) {
                if let Some(state) = I::initialize(&table) {
                    let locks = self
                        .database
                        .resources
                        .local_with(TableKey(table.index()), || TableLocks {
                            readers: 0,
                            conflicts: Vec::new(),
                            locks: Bits::new(),
                        });
                    let index = locks.write(|locks| {
                        let new_index = locks.conflicts.len();
                        let new_reads = self.reads.clone();
                        let new_writes = self.writes.clone();
                        let mut new_conflicts = Bits::new();
                        for (old_index, (old_reads, old_writes, old_conflicts)) in
                            locks.conflicts.iter_mut().enumerate()
                        {
                            if new_reads.is_disjoint(old_writes)
                                && new_writes.is_disjoint(old_reads)
                                && new_writes.is_disjoint(old_writes)
                            {
                                continue;
                            }
                            old_conflicts.set(new_index, true);
                            new_conflicts.set(old_index, true);
                        }

                        locks.conflicts.push((new_reads, new_writes, new_conflicts));
                        new_index
                    });
                    self.indices.insert(self.index, self.states.len());
                    self.states.push((
                        state,
                        TableState {
                            table,
                            index,
                            locks,
                        },
                    ));
                }
            }
            self.index += 1;
        }
    }

    fn with<'b, W: With<'a, 'b, I>>(&'b mut self, key: Key, with: W) -> Result<W::Value, Error> {
        self.update();
        Ok(loop {
            let Self {
                database,
                indices,
                states,
                ..
            } = &*self;
            let slot = self.database.keys.get(key)?;
            let (table_index, store_index) = slot.indices();
            let &state_index = indices
                .get(&(table_index as usize))
                .ok_or(Error::KeyNotInQuery)?;
            let state = unsafe { states.get_unchecked(state_index) };
            let table = unsafe { database.tables.get_unchecked(table_index as usize) };
            let table_read = database.table_read(table);
            if slot.indices() != (table_index, store_index) {
                drop(table_read);
                continue;
            }

            match state.0.try_lock(table_read.keys(), table_read.stores()) {
                Some(mut guard) => {
                    let item = unsafe { I::State::item(&mut guard, store_index as _) };
                    break with.with(Guard(item, TableGuard::new(&state.1, table_read)));
                }
                None if state.1.locks.read(|locks| locks.readers == 0) => {
                    drop(table_read);

                    // It is allowed that there be interleaving of other thread operations here as long as the
                    // `slot.indices` are checked each time a table lock is acquired.
                    let table_write = database.table_write(table);
                    if slot.indices() != (table_index, store_index) {
                        drop(table_write);
                        continue;
                    }

                    let mut guard = state.0.lock(table_write.keys(), table_write.stores());
                    let table_read = table_write.downgrade();
                    let item = unsafe { I::State::item(&mut guard, store_index as _) };
                    break with.with(Guard(item, TableGuard::new(&state.1, table_read)));
                }
                None if state.1.locks.read(|locks| locks.allows(state.1.index)) => {
                    let mut guard = state.0.lock(table_read.keys(), table_read.stores());
                    let item = unsafe { I::State::item(&mut guard, store_index as _) };
                    break with.with(Guard(item, TableGuard::new(&state.1, table_read)));
                }
                None => {
                    drop(table_read);
                    return Err(Error::WouldDeadlock);
                }
            }
        })
    }

    #[inline]
    fn iterate(&mut self) -> impl Iterator<Item = Guard<<I::State as Lock<'a>>::Guard>> {
        self.update();
        from_generator(|| {
            // Try to execute the query using only read locks on tables. This should succeed unless there is contention over
            // the store locks which would cause the `I::State::get` call to fail.
            for (index, (item_state, table_state)) in self.states.iter().enumerate() {
                if let Some(table_read) = self.database.table_try_read(&table_state.table) {
                    match item_state.try_lock(table_read.keys(), table_read.stores()) {
                        Some(guard) => {
                            yield Guard(guard, TableGuard::new(table_state, table_read));
                            continue;
                        }
                        None => drop(table_read),
                    }
                }
                self.defer.push_back(index);
            }

            // Try again to execute the tables that previously failed to take their store locks by still using only read
            // locks on tables hoping that there is no more contention.
            while let Some(state_index) = self.defer.pop_front() {
                debug_assert!(state_index < self.states.len());
                let state = unsafe { self.states.get_unchecked(state_index) };
                let table_read = self.database.table_read(&state.1.table);
                match state.0.try_lock(table_read.keys(), table_read.stores()) {
                    Some(guard) => yield Guard(guard, TableGuard::new(&state.1, table_read)),
                    None if state.1.locks.read(|locks| locks.readers == 0) => {
                        drop(table_read);

                        // Since the read lock failed to make progress, escalate to a write lock.
                        let table_write = self.database.table_write(&state.1.table);
                        let guard = state.0.lock(table_write.keys(), table_write.stores());
                        let table_read = table_write.downgrade();
                        yield Guard(guard, TableGuard::new(&state.1, table_read));
                    }
                    None if state.1.locks.read(|locks| locks.allows(state.1.index)) => {
                        let guard = state.0.lock(table_read.keys(), table_read.stores());
                        yield Guard(guard, TableGuard::new(&state.1, table_read));
                    }
                    None => {
                        drop(table_read);
                        self.skip.push_back(state_index);
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

impl<'a, 'b> TableGuard<'a, 'b> {
    #[inline]
    pub fn new(table_state: &'b TableState, table_read: TableRead<'a>) -> Self {
        table_state.locks.write(|locks| {
            locks.readers += 1;
            locks.locks.set(table_state.index, true);
        });
        Self(table_state, table_read)
    }
}

impl Drop for TableGuard<'_, '_> {
    #[inline]
    fn drop(&mut self) {
        self.0.locks.write(|locks| {
            debug_assert!(locks.readers > 0);
            debug_assert!(locks.locks.has(self.0.index));
            locks.readers -= 1;
            locks.locks.set(self.0.index, false);
        });
    }
}

impl TableLocks {
    pub fn allows(&self, index: usize) -> bool {
        match self.conflicts.get(index) {
            Some((_, _, conflicts)) => self.locks.has_none(conflicts),
            None => false,
        }
    }
}

impl Database {
    pub fn query<I: Item>(&self) -> Result<Query<I>, Error> {
        self.query_with(())
    }

    pub fn query_with<I: Item, F: Filter>(&self, filter: F) -> Result<Query<I, F>, Error> {
        let mut read = HashSet::new();
        let mut write = HashSet::new();
        I::validate(Context {
            read: &mut read,
            write: &mut write,
        })?;

        Ok(Query {
            database: self,
            index: 0,
            indices: HashMap::new(),
            states: Vec::new(),
            defer: VecDeque::new(),
            skip: VecDeque::new(),
            reads: read,
            writes: write,
            filter,
            _marker: PhantomData,
        })
    }
}
