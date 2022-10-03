use super::*;
use crate::{
    database::Database,
    key::Key,
    table::{Store, Table, TableRead},
    Datum, Error,
};
use parking_lot::{MappedRwLockReadGuard, MappedRwLockWriteGuard};
use std::{
    any::TypeId,
    collections::{HashMap, HashSet, VecDeque},
    marker::PhantomData,
    mem::swap,
    ops::{Deref, DerefMut},
    slice::from_raw_parts,
};

pub trait Row {
    type State: for<'a> Lock<'a>;
    type Read: Row;

    fn declare(context: Context) -> Result<(), Error>;
    fn initialize(table: &Table) -> Result<Self::State, Error>;
    fn read(state: Self::State) -> <Self::Read as Row>::State;
}

pub trait Lock<'a> {
    type Guard;
    type Item;
    type Chunk;

    fn try_lock(&self, keys: &[Key], stores: &[Store]) -> Option<Self::Guard>;
    fn lock(&self, keys: &[Key], stores: &[Store]) -> Self::Guard;
    fn item(guard: &mut Self::Guard, index: usize) -> Self::Item;
}

pub struct Context<'a> {
    reads: &'a mut HashSet<TypeId>,
    writes: &'a mut HashSet<TypeId>,
}

pub struct Read<T>(usize, PhantomData<T>);
pub struct Write<T>(usize, PhantomData<T>);
pub struct State;

pub struct Rows<'d, R: Row> {
    pub(crate) indices: HashMap<u32, u32>, // From table index to state index.
    pub(crate) states: Vec<(R::State, &'d Table)>,
    pub(crate) done: VecDeque<u32>,
    pub(crate) pending: VecDeque<u32>,
    _marker: PhantomData<fn(R)>,
}

pub(crate) struct Guards<'d, 'a, R: Row> {
    count: usize,
    database: &'d Database,
    done: &'a mut VecDeque<u32>,
    pending: &'a mut VecDeque<u32>,
    states: &'a [(R::State, &'d Table)],
}

impl<'a> Context<'a> {
    pub fn own(&mut self) -> Context {
        Context {
            reads: self.reads,
            writes: self.writes,
        }
    }

    pub fn read<T: 'static>(&mut self) -> Result<(), Error> {
        let identifier = TypeId::of::<T>();
        if self.writes.contains(&identifier) {
            Err(Error::ReadWriteConflict)
        } else {
            self.reads.insert(identifier);
            Ok(())
        }
    }

    pub fn write<T: 'static>(&mut self) -> Result<(), Error> {
        let identifier = TypeId::of::<T>();
        if self.reads.contains(&identifier) {
            Err(Error::ReadWriteConflict)
        } else if self.writes.insert(identifier) {
            Ok(())
        } else {
            Err(Error::WriteWriteConflict)
        }
    }
}

impl<'d, R: Row> Rows<'d, R> {
    pub fn new() -> Result<Self, Error> {
        // Detects violations of rust's invariants.
        R::declare(Context {
            reads: &mut HashSet::new(),
            writes: &mut HashSet::new(),
        })?;
        Ok(Self {
            indices: HashMap::new(),
            states: Vec::new(),
            done: VecDeque::new(),
            pending: VecDeque::new(),
            _marker: PhantomData,
        })
    }

    #[inline]
    pub(crate) fn guards(&mut self, database: &'d Database) -> Guards<'d, '_, R> {
        // It is assumed that all states to be visited are queued in `self.done`.
        swap(&mut self.done, &mut self.pending);
        Guards {
            count: self.done.len(),
            database: database,
            done: &mut self.done,
            pending: &mut self.pending,
            states: &self.states,
        }
    }

    fn try_guard<'a>(
        database: &'d Database,
        states: &'a [(R::State, &'d Table)],
        index: u32,
        mut valid: impl FnMut() -> bool,
    ) -> Option<Guard<<R::State as Lock<'d>>::Guard, TableRead<'d>>> {
        debug_assert!((index as usize) < states.len());
        let state = unsafe { states.get_unchecked(index as usize) };
        let table_read = database.table_try_read(state.1)?;
        if valid() {
            let guard = state.0.try_lock(table_read.keys(), table_read.stores())?;
            Some(Guard(guard, table_read))
        } else {
            None
        }
    }

    fn guard<'a>(
        database: &'d Database,
        states: &'a [(R::State, &'d Table)],
        index: u32,
        write: bool,
        mut valid: impl FnMut() -> bool,
    ) -> Option<Guard<<R::State as Lock<'d>>::Guard, TableRead<'d>>> {
        debug_assert!((index as usize) < states.len());
        let state = unsafe { states.get_unchecked(index as usize) };
        let table_read = database.table_read(state.1);
        if !valid() {
            return None;
        }

        match state.0.try_lock(table_read.keys(), table_read.stores()) {
            Some(guard) => Some(Guard(guard, table_read)),
            None if write => {
                drop(table_read);

                // Since the read lock failed to make progress, escalate to a write lock.
                let table_write = database.table_write(state.1);
                // Check `valid` again since the table lock was momentarily released.
                if valid() {
                    let guard = state.0.lock(table_write.keys(), table_write.stores());
                    Some(Guard(guard, table_write.downgrade()))
                } else {
                    None
                }
            }
            None => Some(Guard(
                state.0.lock(table_read.keys(), table_read.stores()),
                table_read,
            )),
        }
    }

    fn with<'a, T>(
        &'a mut self,
        key: Key,
        with: impl FnOnce(Guard<<R::State as Lock<'d>>::Item, TableRead<'d>>) -> T,
        database: &'d Database,
    ) -> Result<T, Error> {
        loop {
            let slot = database.keys().get(key)?;
            let (table_index, store_index) = slot.indices();
            let &state_index = self
                .indices
                .get(&table_index)
                .ok_or(Error::KeyNotInQuery(key))?;
            if let Some(mut guard) = Self::guard(database, &self.states, state_index, true, || {
                // If this is not the case, it means that the `key` has just been moved.
                slot.indices() == (table_index, store_index)
            }) {
                let item = R::State::item(&mut guard.0, store_index as _);
                break Ok(with(Guard(item, guard.1)));
            }
        }
    }
}

impl<'d, R: Row> Query<'d> for Rows<'d, R> {
    type Item = <R::State as Lock<'d>>::Item;
    type Items<'b> = impl Iterator<Item = Self::Item> where Self: 'b;
    type Guard = TableRead<'d>;
    type Read = Rows<'d, R::Read>;

    fn item<'a>(
        &'a mut self,
        key: Key,
        context: super::Context<'d>,
    ) -> Result<Guard<Self::Item, Self::Guard>, Error> {
        self.with(key, |guard| guard, context.database)
    }

    fn items<'a>(&'a mut self, context: super::Context<'d>) -> Self::Items<'a> {
        self.guards(context.database)
            .flat_map(|Guard((_, mut guard), table)| {
                (0..table.count()).map(move |index| {
                    let _ = table;
                    R::State::item(&mut guard, index as _)
                })
            })
    }

    fn read(self) -> Self::Read {
        Rows {
            indices: self.indices,
            states: self
                .states
                .into_iter()
                .map(|(state, table)| (R::read(state), table))
                .collect(),
            done: self.done,
            pending: self.pending,
            _marker: PhantomData,
        }
    }

    fn add(&mut self, table: &'d Table) -> bool {
        match R::initialize(&table) {
            Ok(state) => {
                let index = self.states.len() as _;
                self.done.push_back(index);
                self.indices.insert(table.index(), index);
                self.states.push((state, table));
                true
            }
            Err(_) => false,
        }
    }
}

impl Row for Key {
    type State = State;
    type Read = Self;

    fn declare(_: Context) -> Result<(), Error> {
        Ok(())
    }

    fn initialize(_: &Table) -> Result<Self::State, Error> {
        Ok(State)
    }

    fn read(state: Self::State) -> <Self::Read as Row>::State {
        state
    }
}

impl<'a> Lock<'a> for State {
    type Guard = &'a [Key];
    type Item = Key;
    type Chunk = &'a [Key];

    #[inline]
    fn try_lock(&self, keys: &[Key], stores: &[Store]) -> Option<Self::Guard> {
        Some(self.lock(keys, stores))
    }

    #[inline]
    fn lock(&self, keys: &[Key], _: &[Store]) -> Self::Guard {
        unsafe { from_raw_parts(keys.as_ptr(), keys.len()) }
    }

    #[inline]
    fn item(guard: &mut Self::Guard, index: usize) -> Self::Item {
        unsafe { *guard.get_unchecked(index) }
    }
}

impl<D: Datum> Row for &D {
    type State = Read<D>;
    type Read = Self;

    fn declare(mut context: Context) -> Result<(), Error> {
        context.read::<D>()
    }

    fn initialize(table: &Table) -> Result<Self::State, Error> {
        Ok(Read(table.store::<D>()?, PhantomData))
    }

    fn read(state: Self::State) -> <Self::Read as Row>::State {
        state
    }
}

impl<'a, D: Datum> Lock<'a> for Read<D> {
    type Guard = MappedRwLockReadGuard<'a, [D]>;
    type Item = &'a D;
    type Chunk = &'a [D];

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
    fn item(guard: &mut Self::Guard, index: usize) -> Self::Item {
        unsafe { &*guard.as_ptr().add(index) }
    }
}

impl<'a, D: Datum> Row for &'a mut D {
    type State = Write<D>;
    type Read = &'a D;

    fn declare(mut context: Context) -> Result<(), Error> {
        context.read::<D>()
    }

    fn initialize(table: &Table) -> Result<Self::State, Error> {
        Ok(Write(table.store::<D>()?, PhantomData))
    }

    fn read(state: Self::State) -> <Self::Read as Row>::State {
        Read(state.0, PhantomData)
    }
}

impl<'a, D: Datum> Lock<'a> for Write<D> {
    type Guard = MappedRwLockWriteGuard<'a, [D]>;
    type Item = &'a mut D;
    type Chunk = &'a mut [D];

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
    fn item(guard: &mut Self::Guard, index: usize) -> Self::Item {
        unsafe { &mut *guard.as_mut_ptr().add(index) }
    }
}

impl Row for () {
    type State = ();
    type Read = ();

    fn declare(_: Context) -> Result<(), Error> {
        Ok(())
    }

    fn initialize(_: &Table) -> Result<Self::State, Error> {
        Ok(())
    }

    fn read(_: Self::State) -> <Self::Read as Row>::State {}
}

impl<C1: Row> Row for (C1,) {
    type State = (C1::State,);
    type Read = (C1::Read,);

    fn declare(context: Context) -> Result<(), Error> {
        C1::declare(context)
    }

    fn initialize(table: &Table) -> Result<Self::State, Error> {
        Ok((C1::initialize(table)?,))
    }

    fn read(state: Self::State) -> <Self::Read as Row>::State {
        (C1::read(state.0),)
    }
}

impl<C1: Row, C2: Row> Row for (C1, C2) {
    type State = (C1::State, C2::State);
    type Read = (C1::Read, C2::Read);

    fn declare(mut context: Context) -> Result<(), Error> {
        C1::declare(context.own())?;
        C2::declare(context)
    }

    fn initialize(table: &Table) -> Result<Self::State, Error> {
        Ok((C1::initialize(table)?, C2::initialize(table)?))
    }

    fn read(state: Self::State) -> <Self::Read as Row>::State {
        (C1::read(state.0), C2::read(state.1))
    }
}

impl<'a> Lock<'a> for () {
    type Guard = ();
    type Item = ();
    type Chunk = ();

    #[inline]
    fn try_lock(&self, _: &[Key], _: &[Store]) -> Option<Self::Guard> {
        Some(())
    }
    #[inline]
    fn lock(&self, _: &[Key], _: &[Store]) -> Self::Guard {}
    #[inline]
    fn item(_: &mut Self::Guard, _: usize) -> Self::Item {}
}

impl<'a, L1: Lock<'a>> Lock<'a> for (L1,) {
    type Guard = (L1::Guard,);
    type Item = (L1::Item,);
    type Chunk = (L1::Chunk,);

    #[inline]
    fn try_lock(&self, keys: &[Key], stores: &[Store]) -> Option<Self::Guard> {
        Some((self.0.try_lock(keys, stores)?,))
    }

    #[inline]
    fn lock(&self, keys: &[Key], stores: &[Store]) -> Self::Guard {
        (self.0.lock(keys, stores),)
    }

    #[inline]
    fn item(guard: &mut Self::Guard, index: usize) -> Self::Item {
        (L1::item(&mut guard.0, index),)
    }
}

impl<'a, L1: Lock<'a>, L2: Lock<'a>> Lock<'a> for (L1, L2) {
    type Guard = (L1::Guard, L2::Guard);
    type Item = (L1::Item, L2::Item);
    type Chunk = (L1::Chunk, L2::Chunk);

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
    fn item(guard: &mut Self::Guard, index: usize) -> Self::Item {
        (L1::item(&mut guard.0, index), L2::item(&mut guard.1, index))
    }
}

impl<T, G> Deref for Guard<T, G> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, G> DerefMut for Guard<T, G> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, 'b, R: Row> Iterator for Guards<'a, 'b, R> {
    type Item = Guard<(u32, <R::State as Lock<'a>>::Guard), TableRead<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(state_index) = self.pending.pop_front() {
            let (state_index, guard, table) = if self.count == 0 {
                let Guard(guard, table) =
                    Rows::<R>::guard(self.database, self.states, state_index, true, || true)?;
                (state_index, guard, table)
            } else {
                match Rows::<R>::try_guard(self.database, self.states, state_index, || true) {
                    Some(Guard(guard, table)) => (state_index, guard, table),
                    None => {
                        self.count = self.count.saturating_sub(1);
                        self.pending.push_back(state_index);
                        continue;
                    }
                }
            };
            self.done.push_back(state_index);
            self.count = self.pending.len();
            return Some(Guard((state_index, guard), table));
        }

        None
    }
}

impl<'a, R: Row> ExactSizeIterator for Guards<'a, '_, R> {
    #[inline]
    fn len(&self) -> usize {
        self.pending.len()
    }
}

impl<R: Row> Drop for Guards<'_, '_, R> {
    #[inline]
    fn drop(&mut self) {
        if self.done.len() < self.pending.len() {
            swap(self.done, self.pending)
        }
        self.done.extend(self.pending.drain(..));
    }
}
