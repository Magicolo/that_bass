use super::*;
use crate::{
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
    slice::from_raw_parts,
};

pub trait Row {
    type State;
    type Read: Row;
    type Guard<'a>;
    type Item<'a>;

    fn declare(context: Context) -> Result<(), Error>;
    fn initialize(table: &Table) -> Result<Self::State, Error>;
    fn read(state: Self::State) -> <Self::Read as Row>::State;
    fn try_lock<'a>(
        state: &Self::State,
        keys: &'a [Key],
        stores: &'a [Store],
    ) -> Option<Self::Guard<'a>>;
    fn lock<'a>(state: &Self::State, keys: &'a [Key], stores: &'a [Store]) -> Self::Guard<'a>;
    fn item<'a: 'b, 'b>(guard: &'b mut Self::Guard<'a>, index: usize) -> Self::Item<'b>;
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
    pub(crate) fn guards<S, F: FnMut(S, u32, R::Guard<'_>, &TableRead<'d>) -> Result<S, S>>(
        &mut self,
        database: &'d Database,
        state: S,
        mut fold: F,
    ) -> S {
        let mut fold = |mut state: S| -> Result<S, S> {
            for _ in 0..self.pending.len() {
                let state_index = unsafe { self.pending.pop_front().unwrap_unchecked() };
                let (row_state, table) = unsafe { self.states.get_unchecked(state_index as usize) };
                if let Some(read) = database.table_try_read(table) {
                    if let Some(guard) = R::try_lock(row_state, read.keys(), read.stores()) {
                        self.done.push_back(state_index);
                        state = fold(state, state_index, guard, &read)?;
                        continue;
                    }
                }
                self.pending.push_back(state_index);
            }

            while let Some(state_index) = self.pending.pop_front() {
                self.done.push_back(state_index);
                let (row_state, table) = unsafe { self.states.get_unchecked(state_index as usize) };
                let read = database.table_read(table);
                // FIX: There could be a deadlock here.
                // - Would be fixed if store locks were always taken in the same order.
                let guard = R::lock(row_state, read.keys(), read.stores());
                state = fold(state, state_index, guard, &read)?;
            }
            Ok(state)
        };

        match fold(state) {
            Ok(state) => {
                swap(&mut self.done, &mut self.pending);
                state
            }
            Err(state) => {
                // Fold was interrupted, so move remaining indices in `pending` while preserving the order of the indices in `done`.
                if self.done.len() < self.pending.len() {
                    while let Some(index) = self.done.pop_back() {
                        self.pending.push_front(index);
                    }
                } else {
                    while let Some(index) = self.pending.pop_front() {
                        self.done.push_back(index);
                    }
                    swap(&mut self.done, &mut self.pending);
                }
                state
            }
        }
    }
}

impl<'d, R: Row> Query<'d> for Rows<'d, R> {
    type Item<'a> = R::Item<'a>;
    type Read = Rows<'d, R::Read>;

    fn initialize(&mut self, table: &'d Table) {
        if let Ok(state) = R::initialize(&table) {
            let index = self.states.len() as _;
            self.pending.push_back(index);
            self.indices.insert(table.index(), index);
            self.states.push((state, table));
        }
    }

    fn try_find<T, F: FnOnce(Result<Self::Item<'_>, Error>) -> T>(
        &mut self,
        key: Key,
        context: super::Context<'d>,
        find: F,
    ) -> T {
        loop {
            let slot = match context.database.keys().get(key) {
                Ok(slot) => slot,
                Err(error) => break find(Err(error)),
            };
            let (table_index, store_index) = slot.indices();
            let state_index = match self.indices.get(&table_index) {
                Some(&state_index) => state_index,
                None => break find(Err(Error::KeyNotInQuery(key))),
            };

            // If valid fails, it means that the `key` has just been moved.
            let valid = || slot.indices() == (table_index, store_index);
            let (row_state, table) = unsafe { self.states.get_unchecked(state_index as usize) };
            if let Some(read) = context.database.table_read_with(table, |_| valid()) {
                if let Some(mut guard) = R::try_lock(row_state, read.keys(), read.stores()) {
                    break find(Ok(R::item(&mut guard, store_index as _)));
                }

                drop(read);
                if let Some(write) = context.database.table_write_with(table, |_| valid()) {
                    let mut guard = R::lock(row_state, write.keys(), write.stores());
                    break find(Ok(R::item(&mut guard, store_index as _)));
                }
            }
        }
    }

    #[inline]
    fn try_fold<S, F: FnMut(S, Self::Item<'_>) -> Result<S, S>>(
        &mut self,
        context: super::Context<'d>,
        state: S,
        mut fold: F,
    ) -> S {
        self.guards(context.database, state, |mut state, _, mut guard, table| {
            for i in 0..table.count() {
                state = fold(state, R::item(&mut guard, i as _))?;
            }
            Ok(state)
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
}

impl Row for Key {
    type State = State;
    type Read = Self;
    type Guard<'a> = &'a [Key];
    type Item<'a> = Key;

    fn declare(_: Context) -> Result<(), Error> {
        Ok(())
    }

    fn initialize(_: &Table) -> Result<Self::State, Error> {
        Ok(State)
    }

    fn read(state: Self::State) -> <Self::Read as Row>::State {
        state
    }

    #[inline]
    fn try_lock<'a>(
        state: &Self::State,
        keys: &'a [Key],
        stores: &'a [Store],
    ) -> Option<Self::Guard<'a>> {
        Some(Self::lock(state, keys, stores))
    }

    #[inline]
    fn lock<'a>(_: &Self::State, keys: &'a [Key], _: &'a [Store]) -> Self::Guard<'a> {
        unsafe { from_raw_parts(keys.as_ptr(), keys.len()) }
    }

    #[inline]
    fn item<'a: 'b, 'b>(guard: &'b mut Self::Guard<'a>, index: usize) -> Self::Item<'b> {
        unsafe { *guard.get_unchecked(index) }
    }
}

impl<D: Datum> Row for &D {
    type State = Read<D>;
    type Read = Self;
    type Guard<'a> = MappedRwLockReadGuard<'a, [D]>;
    type Item<'a> = &'a D;

    fn declare(mut context: Context) -> Result<(), Error> {
        context.read::<D>()
    }

    fn initialize(table: &Table) -> Result<Self::State, Error> {
        Ok(Read(table.store::<D>()?, PhantomData))
    }

    fn read(state: Self::State) -> <Self::Read as Row>::State {
        state
    }

    #[inline]
    fn try_lock<'a>(
        state: &Self::State,
        keys: &'a [Key],
        stores: &'a [Store],
    ) -> Option<Self::Guard<'a>> {
        debug_assert!(state.0 < stores.len());
        let store = unsafe { &*stores.as_ptr().add(state.0) };
        unsafe { store.try_read(.., keys.len()) }
    }

    #[inline]
    fn lock<'a>(state: &Self::State, keys: &'a [Key], stores: &'a [Store]) -> Self::Guard<'a> {
        debug_assert!(state.0 < stores.len());
        let store = unsafe { &*stores.as_ptr().add(state.0) };
        unsafe { store.read(.., keys.len()) }
    }

    #[inline]
    fn item<'a: 'b, 'b>(guard: &'b mut Self::Guard<'a>, index: usize) -> Self::Item<'b> {
        unsafe { &*guard.as_ptr().add(index) }
    }
}

impl<'c, D: Datum> Row for &'c mut D {
    type State = Write<D>;
    type Read = &'c D;
    type Guard<'a> = MappedRwLockWriteGuard<'a, [D]>;
    type Item<'a> = &'a mut D;

    fn declare(mut context: Context) -> Result<(), Error> {
        context.read::<D>()
    }

    fn initialize(table: &Table) -> Result<Self::State, Error> {
        Ok(Write(table.store::<D>()?, PhantomData))
    }

    fn read(state: Self::State) -> <Self::Read as Row>::State {
        Read(state.0, PhantomData)
    }

    #[inline]
    fn try_lock<'a>(
        state: &Self::State,
        keys: &'a [Key],
        stores: &'a [Store],
    ) -> Option<Self::Guard<'a>> {
        debug_assert!(state.0 < stores.len());
        let store = unsafe { &*stores.as_ptr().add(state.0) };
        unsafe { store.try_write(.., keys.len()) }
    }

    #[inline]
    fn lock<'a>(state: &Self::State, keys: &'a [Key], stores: &'a [Store]) -> Self::Guard<'a> {
        debug_assert!(state.0 < stores.len());
        let store = unsafe { &*stores.as_ptr().add(state.0) };
        unsafe { store.write(.., keys.len()) }
    }

    #[inline]
    fn item<'a: 'b, 'b>(guard: &'b mut Self::Guard<'a>, index: usize) -> Self::Item<'b> {
        unsafe { &mut *guard.as_mut_ptr().add(index) }
    }
}

impl Row for () {
    type State = ();
    type Read = ();
    type Guard<'a> = ();
    type Item<'a> = ();

    fn declare(_: Context) -> Result<(), Error> {
        Ok(())
    }

    fn initialize(_: &Table) -> Result<Self::State, Error> {
        Ok(())
    }

    fn read(_: Self::State) -> <Self::Read as Row>::State {}

    #[inline]
    fn try_lock<'a>(_: &Self::State, _: &'a [Key], _: &'a [Store]) -> Option<Self::Guard<'a>> {
        Some(())
    }

    #[inline]
    fn lock<'a>(_: &Self::State, _: &'a [Key], _: &'a [Store]) -> Self::Guard<'a> {}

    #[inline]
    fn item<'a: 'b, 'b>(_: &'b mut Self::Guard<'a>, _: usize) -> Self::Item<'b> {}
}

impl<C1: Row> Row for (C1,) {
    type State = (C1::State,);
    type Read = (C1::Read,);
    type Guard<'a> = (C1::Guard<'a>,);
    type Item<'a> = (C1::Item<'a>,);

    fn declare(context: Context) -> Result<(), Error> {
        C1::declare(context)
    }

    fn initialize(table: &Table) -> Result<Self::State, Error> {
        Ok((C1::initialize(table)?,))
    }

    fn read(state: Self::State) -> <Self::Read as Row>::State {
        (C1::read(state.0),)
    }

    #[inline]
    fn try_lock<'a>(
        state: &Self::State,
        keys: &'a [Key],
        stores: &'a [Store],
    ) -> Option<Self::Guard<'a>> {
        Some((C1::try_lock(&state.0, keys, stores)?,))
    }

    #[inline]
    fn lock<'a>(state: &Self::State, keys: &'a [Key], stores: &'a [Store]) -> Self::Guard<'a> {
        (C1::lock(&state.0, keys, stores),)
    }

    #[inline]
    fn item<'a: 'b, 'b>(guard: &'b mut Self::Guard<'a>, index: usize) -> Self::Item<'b> {
        (C1::item(&mut guard.0, index),)
    }
}

impl<C1: Row, C2: Row> Row for (C1, C2) {
    type State = (C1::State, C2::State);
    type Read = (C1::Read, C2::Read);
    type Guard<'a> = (C1::Guard<'a>, C2::Guard<'a>);
    type Item<'a> = (C1::Item<'a>, C2::Item<'a>);

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

    #[inline]
    fn try_lock<'a>(
        state: &Self::State,
        keys: &'a [Key],
        stores: &'a [Store],
    ) -> Option<Self::Guard<'a>> {
        Some((
            C1::try_lock(&state.0, keys, stores)?,
            C2::try_lock(&state.1, keys, stores)?,
        ))
    }

    #[inline]
    fn lock<'a>(state: &Self::State, keys: &'a [Key], stores: &'a [Store]) -> Self::Guard<'a> {
        (
            C1::lock(&state.0, keys, stores),
            C2::lock(&state.1, keys, stores),
        )
    }

    #[inline]
    fn item<'a: 'b, 'b>(guard: &'b mut Self::Guard<'a>, index: usize) -> Self::Item<'b> {
        (C1::item(&mut guard.0, index), C2::item(&mut guard.1, index))
    }
}
