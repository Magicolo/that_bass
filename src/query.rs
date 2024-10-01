use crate::{
    Database, Error,
    core::{
        iterate::FullIterator,
        utility::{fold_swap, get_unchecked, get_unchecked_mut, try_fold_swap},
    },
    filter::Filter,
    key::{Key, Keys},
    row::{Access, ChunkContext, InitializeContext, ItemContext, Row, ShareAccess},
    table::{Table, Tables},
};
use std::{
    any::TypeId,
    marker::PhantomData,
    num::NonZeroUsize,
    ops::ControlFlow::{self, *},
    sync::Arc,
};

pub struct Query<'d, R: Row, F = (), I = Item> {
    database: &'d Database,
    keys: Keys<'d>,
    tables: Tables<'d>,
    index: usize,
    indices: Vec<u32>,     // May be reordered (ex: by `fold_swap`).
    states: Vec<State<R>>, /* Must remain sorted by `state.table.index()` for `binary_search` to
                            * work. */
    filter: F,
    _marker: PhantomData<fn(I)>,
}

pub struct Split<'d, 'a, R: Row, I = Item> {
    keys: &'a Keys<'d>,
    state: &'a State<R>,
    _marker: PhantomData<fn(I)>,
}

pub struct Item;
pub struct Chunk;

pub struct By<V = ()> {
    pairs: Vec<(Key, V)>,
    pending: Vec<(Key, V, u32)>,
    sorted: Vec<Vec<(Key, V)>>,
    errors: Vec<(V, Error)>,
    indices: Vec<u32>,
}

struct State<R: Row> {
    state: R::State,
    table: Arc<Table>,
    locks: Box<[(usize, Access)]>,
}

impl Database {
    pub fn query<R: Row>(&self) -> Result<Query<'_, R>, Error> {
        ShareAccess::<R>::from(self.resources()).map(|_| Query {
            database: self,
            keys: self.keys(),
            tables: self.tables(),
            indices: Vec::new(),
            states: Vec::new(),
            index: 0,
            filter: (),
            _marker: PhantomData,
        })
    }
}

impl<'d, R: Row, F: Filter, I> Query<'d, R, F, I> {
    pub fn keys<K: Default + Extend<Key>>(&mut self) -> K {
        let mut keys = K::default();
        self.keys_in(&mut keys);
        keys
    }

    pub fn keys_in<K: Extend<Key>>(&mut self, keys: &mut K) {
        self.update();
        Self::try_guards(
            keys,
            &mut self.indices,
            &self.states,
            |state, _, _, _, keys, count| {
                state.extend(keys.iter().take(count.get()).copied());
                Continue(state)
            },
        );
    }

    pub fn tables(&mut self) -> impl FullIterator<Item = &Table> {
        self.update();
        self.states.iter().map(|state| &*state.table)
    }

    pub fn split(&mut self) -> impl FullIterator<Item = Split<'d, '_, R, I>> {
        self.update();
        self.states.iter().map(|state| Split {
            keys: &self.keys,
            state,
            _marker: PhantomData,
        })
    }

    pub fn read(self) -> Query<'d, R::Read, F, I> {
        Query {
            database: self.database,
            keys: self.keys,
            tables: self.tables,
            indices: self.indices.clone(),
            states: self
                .states
                .into_iter()
                .map(|mut state| {
                    for (_, access) in state.locks.iter_mut() {
                        *access = access.read();
                    }
                    State {
                        state: R::read(&state.state),
                        table: state.table,
                        locks: state.locks,
                    }
                })
                .collect(),
            filter: self.filter,
            index: self.index,
            _marker: PhantomData,
        }
    }

    pub fn filter<G: Filter + Default>(self) -> Query<'d, R, (F, G), I> {
        self.filter_with(G::default())
    }

    pub fn filter_with<G: Filter>(mut self, filter: G) -> Query<'d, R, (F, G), I> {
        self.states
            .retain(|state| filter.filter(&state.table, self.database));
        self.indices.clear();
        self.indices.extend(0..self.states.len() as u32);
        Query {
            database: self.database,
            keys: self.keys,
            tables: self.tables,
            indices: self.indices,
            states: self.states,
            index: self.index,
            filter: self.filter.and(filter),
            _marker: PhantomData,
        }
    }

    pub(crate) fn update(&mut self) {
        while let Ok(table) = self.tables.get_shared(self.index) {
            self.index += 1;

            if self.filter.filter(&table, self.database) {
                // Initialize first to save some work if it fails.
                let Ok(state) = R::initialize(InitializeContext::new(&table)) else {
                    continue;
                };
                let mut locks = Vec::new();
                let Ok(accesses) = ShareAccess::<R>::from(&self.database.resources) else {
                    continue;
                };
                for &access in accesses.iter() {
                    if let Ok((index, column)) = table.column_with(access.identifier()) {
                        // No need to lock columns of size 0.
                        if column.meta().size() > 0 {
                            locks.push((index, access));
                        }
                    }
                }
                // The sorting of indices ensures that there cannot be a deadlock between `Rows`
                // when locking multiple columns as long as this happens while
                // holding at most 1 table lock.
                locks.sort_unstable_by_key(|&(index, _)| index);

                let index = self.states.len() as u32;
                self.indices.push(index);
                self.states.push(State {
                    state,
                    table,
                    locks: locks.into_boxed_slice(),
                });
            }
        }
    }

    #[inline]
    fn try_guards<S>(
        state: S,
        indices: &mut [u32],
        states: &[State<R>],
        fold: impl FnMut(S, usize, &R::State, &Table, &[Key], NonZeroUsize) -> ControlFlow<S, S>,
    ) -> ControlFlow<S, S> {
        try_fold_swap(
            indices,
            state,
            (states, fold),
            |state, (states, fold), index| {
                let index = *index as usize;
                let State {
                    state: row,
                    table,
                    locks,
                } = unsafe { get_unchecked(states, index) };
                let Some(keys) = table.keys.try_read() else {
                    return Err(state);
                };
                let Some(count) = NonZeroUsize::new(table.count()) else {
                    return Ok(Continue(state));
                };

                try_lock(state, locks, table, |state, table| {
                    fold(state, index, row, table, &keys, count)
                })
            },
            |state, (states, fold), index| {
                let index = *index as usize;
                let State {
                    state: row,
                    table,
                    locks,
                } = unsafe { get_unchecked(states, index) };
                let keys = table.keys.read();
                let Some(count) = NonZeroUsize::new(table.count()) else {
                    return Continue(state);
                };
                lock(locks, table, |table| {
                    fold(state, index, row, table, &keys, count)
                })
            },
        )
    }

    #[inline]
    fn guards<S>(
        state: S,
        indices: &mut [u32],
        states: &[State<R>],
        fold: impl FnMut(S, usize, &R::State, &Table, &[Key], NonZeroUsize) -> S,
    ) -> S {
        fold_swap(
            indices,
            state,
            (states, fold),
            |state, (states, fold), index| {
                let index = *index as usize;
                let State {
                    state: row,
                    table,
                    locks,
                } = unsafe { get_unchecked(states, index) };
                let Some(keys) = table.keys.try_read() else {
                    return Err(state);
                };
                let Some(count) = NonZeroUsize::new(table.count()) else {
                    return Ok(state);
                };
                try_lock(state, locks, table, |state, table| {
                    fold(state, index, row, table, &keys, count)
                })
            },
            |state, (states, fold), index| {
                let index = *index as usize;
                let State {
                    state: row,
                    table,
                    locks,
                } = unsafe { get_unchecked(states, index) };
                let keys = table.keys.read();
                let Some(count) = NonZeroUsize::new(table.count()) else {
                    return state;
                };
                lock(locks, table, |table| {
                    fold(state, index, row, table, &keys, count)
                })
            },
        )
    }
}

impl<'d, R: Row, F: Filter> Query<'d, R, F, Item> {
    pub fn chunk(self) -> Query<'d, R, F, Chunk> {
        Query {
            database: self.database,
            tables: self.tables,
            keys: self.keys,
            indices: self.indices,
            states: self.states,
            index: self.index,
            filter: self.filter,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn count(&mut self) -> usize {
        self.tables().map(|table| table.count()).sum()
    }

    #[inline]
    pub fn count_by<V>(&mut self, by: &By<V>) -> usize {
        self.update();
        by.pairs
            .iter()
            .filter(|&&(key, ..)| match self.keys.get(key) {
                Ok((_, table)) => find_state(&self.states, table).is_some(),
                Err(_) => false,
            })
            .count()
    }

    #[inline]
    pub fn try_fold<S, G: FnMut(S, R::Item<'_>) -> ControlFlow<S, S>>(
        &mut self,
        state: S,
        mut fold: G,
    ) -> S {
        self.update();
        let flow = Self::try_guards(
            state,
            &mut self.indices,
            &self.states,
            |mut state, _, row, table, keys, count| {
                let context = ItemContext::new(table, keys);
                for i in 0..count.get() {
                    let item = unsafe { R::item(row, context.with(i)) };
                    state = fold(state, item)?;
                }
                Continue(state)
            },
        );
        match flow {
            Continue(state) => state,
            Break(state) => state,
        }
    }

    #[inline]
    pub fn try_fold_by<V, S, G: FnMut(S, V, Result<R::Item<'_>, Error>) -> ControlFlow<S, S>>(
        &mut self,
        by: &mut By<V>,
        mut state: S,
        mut fold: G,
    ) -> S {
        self.update();
        while by.sorted.len() < self.states.len() {
            by.sorted.push(Vec::new());
        }
        loop {
            state = match self.try_fold_by_sorted(by, state, &mut fold) {
                Continue(state) => {
                    by.indices.clear();
                    state
                }
                Break(state) => {
                    // Fold was interrupted. Clean up.
                    by.pending.clear();
                    for index in by.indices.drain(..) {
                        unsafe { get_unchecked_mut(&mut by.sorted, index as usize) }.clear();
                    }
                    break state;
                }
            };
            if by.pending.is_empty() {
                break state;
            }
        }
    }

    #[inline]
    pub fn fold<S, G: FnMut(S, R::Item<'_>) -> S>(&mut self, state: S, mut fold: G) -> S {
        self.update();
        Self::guards(
            state,
            &mut self.indices,
            &self.states,
            |mut state, _, row, table, keys, count| {
                let context = ItemContext::new(table, keys);
                for i in 0..count.get() {
                    let item = unsafe { R::item(row, context.with(i)) };
                    state = fold(state, item);
                }
                state
            },
        )
    }

    #[inline]
    pub fn fold_by<V, S, G: FnMut(S, V, Result<R::Item<'_>, Error>) -> S>(
        &mut self,
        by: &mut By<V>,
        mut state: S,
        mut fold: G,
    ) -> S {
        self.update();
        while by.sorted.len() < self.states.len() {
            by.sorted.push(Vec::new());
        }
        loop {
            state = self.fold_by_sorted(by, state, &mut fold);
            by.indices.clear();
            if by.pending.is_empty() {
                break state;
            }
        }
    }

    #[inline]
    pub fn fold_by_ok<V, S, G: FnMut(S, V, R::Item<'_>) -> S>(
        &mut self,
        by: &mut By<V>,
        state: S,
        mut fold: G,
    ) -> S {
        self.fold_by(by, state, |state, value, item| match item {
            Ok(item) => fold(state, value, item),
            Err(_) => state,
        })
    }

    #[inline]
    pub fn try_each<G: FnMut(R::Item<'_>) -> bool>(&mut self, mut each: G) {
        self.try_fold(
            (),
            |_, item| if each(item) { Continue(()) } else { Break(()) },
        )
    }

    #[inline]
    pub fn each<G: FnMut(R::Item<'_>)>(&mut self, mut each: G) {
        self.fold((), |_, item| each(item))
    }

    #[inline]
    pub fn each_by<V, G: FnMut(V, Result<R::Item<'_>, Error>)>(
        &mut self,
        by: &mut By<V>,
        mut each: G,
    ) {
        self.fold_by(by, (), |_, value, item| each(value, item))
    }

    #[inline]
    pub fn each_by_ok<V, G: FnMut(V, R::Item<'_>)>(&mut self, by: &mut By<V>, mut each: G) {
        self.fold_by(by, (), |_, value, item| {
            if let Ok(item) = item {
                each(value, item);
            }
        })
    }

    #[inline]
    pub fn has(&mut self, key: Key) -> bool {
        self.update();
        match self.keys.get(key) {
            Ok((_, table)) => find_state(&self.states, table).is_some(),
            Err(_) => false,
        }
    }

    pub fn try_find<T, G: FnOnce(Result<R::Item<'_>, Error>) -> T>(
        &mut self,
        key: Key,
        find: G,
    ) -> T {
        self.update();
        let (slot, mut old_table) = match self.keys.get(key) {
            Ok(pair) => pair,
            Err(error) => return find(Err(error)),
        };
        loop {
            let State {
                state,
                table,
                locks,
            } = match find_state(&self.states, old_table) {
                Some((_, state)) => state,
                None => break find(Err(Error::KeyNotInQuery(key))),
            };

            let keys = table.keys.read();
            // The key must be checked again while holding the table lock to be sure is has
            // not been moved/destroyed since last read.
            let new_table = match slot.table(key) {
                Ok(new_table) => new_table,
                // The `key` has just been destroyed.
                // - Do not call `find` in here since it would hold locks for longer.
                Err(error) => {
                    drop(keys);
                    break find(Err(error));
                }
            };
            if new_table == table.index() {
                debug_assert_eq!(old_table, table.index());
                let Some(count) = NonZeroUsize::new(table.count()) else {
                    drop(keys);
                    break find(Err(Error::InvalidKey(key)));
                };
                let row = slot.row();
                if row < count.get() && keys.get(row) == Some(&key) {
                    break lock(locks, table, |table| {
                        let context = ItemContext::new(table, &keys).with(row);
                        find(Ok(unsafe { R::item(state, context) }))
                    });
                } else {
                    // This is an edge case where a `Create::resolve` operation from another thread
                    // has initialized its slots but hasn't yet commited the
                    // table count. This should be reported as if `Database::keys().get()` had
                    // failed.
                    drop(keys);
                    break find(Err(Error::InvalidKey(key)));
                }
            } else {
                // The `key` has just been moved; try again with the new table.
                old_table = new_table;
            }
        }
    }

    #[inline]
    pub fn find<T, G: FnOnce(R::Item<'_>) -> T>(&mut self, key: Key, find: G) -> Result<T, Error> {
        self.try_find(key, |item| item.map(find))
    }

    fn fold_by_sorted<V, S, G: FnMut(S, V, Result<R::Item<'_>, Error>) -> S>(
        &mut self,
        by: &mut By<V>,
        mut state: S,
        fold: &mut G,
    ) -> S {
        for (key, value, result) in self.keys.get_all_with(by.pairs.drain(..)) {
            match result {
                Ok((_, table)) => {
                    if let Some((value, error)) = Self::sort(
                        &self.states,
                        &mut by.sorted,
                        &mut by.indices,
                        key,
                        value,
                        table,
                    ) {
                        state = fold(state, value, Err(error))
                    }
                }
                Err(error) => state = fold(state, value, Err(dbg!(error))),
            }
        }

        for (key, value, table) in by.pending.drain(..) {
            if let Some((value, error)) = Self::sort(
                &self.states,
                &mut by.sorted,
                &mut by.indices,
                key,
                value,
                table,
            ) {
                state = fold(state, value, Err(error))
            }
        }

        let mut state = Self::guards(
            state,
            &mut by.indices,
            &self.states,
            |mut state, index, row_state, table, keys, count| {
                let context = ItemContext::new(table, keys);
                let pairs = unsafe { get_unchecked_mut(&mut by.sorted, index) };
                for (key, value) in pairs.drain(..) {
                    let slot = unsafe { self.keys.get_unchecked(key) };
                    // The key is allowed to move within its table (such as with a swap as part of a
                    // remove).
                    match slot.table(key) {
                        Ok(table_index) if table.index() == table_index => {
                            let row = slot.row();
                            if row < count.get() && keys.get(row) == Some(&key) {
                                let item = unsafe { R::item(row_state, context.with(row)) };
                                state = fold(state, value, Ok(item));
                            } else {
                                // This is an edge case where a `Create::resolve` operation from
                                // another thread has initialized its slots but hasn't
                                // yet commited the table count. This should be reported as if
                                // `Database::keys().get()` had failed.
                                by.errors.push((value, Error::InvalidKey(key)));
                            }
                        }
                        // The key has moved to another table between the last moment the slot
                        // indices were read and now.
                        Ok(table_index) => by.pending.push((key, value, table_index)),
                        Err(error) => by.errors.push((value, error)),
                    }
                }
                state
            },
        );
        // Resolve errors outside of `guards` to release locks sooner.
        for (value, error) in by.errors.drain(..) {
            state = fold(state, value, Err(error));
        }
        state
    }

    fn try_fold_by_sorted<V, S, G: FnMut(S, V, Result<R::Item<'_>, Error>) -> ControlFlow<S, S>>(
        &mut self,
        by: &mut By<V>,
        mut state: S,
        fold: &mut G,
    ) -> ControlFlow<S, S> {
        for (key, value) in by.pairs.drain(..) {
            match self.keys.get(key) {
                Ok((_, table)) => {
                    if let Some((value, error)) = Self::sort(
                        &self.states,
                        &mut by.sorted,
                        &mut by.indices,
                        key,
                        value,
                        table,
                    ) {
                        state = fold(state, value, Err(error))?
                    }
                }
                Err(error) => state = fold(state, value, Err(error))?,
            }
        }

        for (key, value, table) in by.pending.drain(..) {
            if let Some((value, error)) = Self::sort(
                &self.states,
                &mut by.sorted,
                &mut by.indices,
                key,
                value,
                table,
            ) {
                state = fold(state, value, Err(error))?
            }
        }

        let state = Self::try_guards(
            state,
            &mut by.indices,
            &self.states,
            |mut state, index, row_state, table, keys, count| {
                let context = ItemContext::new(table, keys);
                let slots = unsafe { get_unchecked_mut(&mut by.sorted, index) };
                for (key, value) in slots.drain(..) {
                    let slot = unsafe { self.keys.get_unchecked(key) };
                    // The key is allowed to move within its table (such as with a swap as part of a
                    // remove).
                    match slot.table(key) {
                        Ok(table_index) if table.index() == table_index => {
                            let row = slot.row();
                            if row < count.get() && keys.get(row) == Some(&key) {
                                let item = unsafe { R::item(row_state, context.with(row)) };
                                state = fold(state, value, Ok(item))?;
                            } else {
                                by.errors.push((value, Error::InvalidKey(key)));
                            }
                        }
                        // The key has moved to another table between the last moment the slot
                        // indices were read and now.
                        Ok(table_index) => by.pending.push((key, value, table_index)),
                        Err(error) => by.errors.push((value, error)),
                    }
                }
                Continue(state)
            },
        );
        let mut state = state?;
        // Resolve errors outside of `guards` to release locks sooner.
        for (value, error) in by.errors.drain(..) {
            state = fold(state, value, Err(error))?;
        }
        Continue(state)
    }

    /// Sorts keys by state index such that table locks can be used for
    /// (hopefully) more than one key at a time.
    #[inline]
    fn sort<V>(
        states: &[State<R>],
        slots: &mut [Vec<(Key, V)>],
        indices: &mut Vec<u32>,
        key: Key,
        value: V,
        table: u32,
    ) -> Option<(V, Error)> {
        match find_state(states, table) {
            Some((index, _)) => {
                let slots = unsafe { get_unchecked_mut(slots, index) };
                if slots.is_empty() {
                    indices.push(index as _);
                }
                slots.push((key, value));
                None
            }
            None => Some((value, Error::KeyNotInQuery(key))),
        }
    }
}

impl<'d, R: Row, F: Filter> Query<'d, R, F, Chunk> {
    pub fn item(self) -> Query<'d, R, F, Item> {
        Query {
            database: self.database,
            tables: self.tables,
            keys: self.keys,
            indices: self.indices,
            states: self.states,
            index: self.index,
            filter: self.filter,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn count(&mut self) -> usize {
        self.tables().filter(|table| table.count() > 0).count()
    }

    #[inline]
    pub fn try_fold<S, G: FnMut(S, R::Chunk<'_>) -> ControlFlow<S, S>>(
        &mut self,
        state: S,
        mut fold: G,
    ) -> S {
        self.update();
        let flow = Self::try_guards(
            state,
            &mut self.indices,
            &self.states,
            |state, _, row, table, keys, count| {
                let chunk = unsafe { R::chunk(row, ChunkContext::new(table, keys, count)) };
                fold(state, chunk)
            },
        );
        match flow {
            Continue(state) => state,
            Break(state) => state,
        }
    }

    #[inline]
    pub fn fold<S, G: FnMut(S, R::Chunk<'_>) -> S>(&mut self, state: S, mut fold: G) -> S {
        self.update();
        Self::guards(
            state,
            &mut self.indices,
            &self.states,
            |state, _, row, table, keys, count| {
                let chunk = unsafe { R::chunk(row, ChunkContext::new(table, keys, count)) };
                fold(state, chunk)
            },
        )
    }

    #[inline]
    pub fn try_each<G: FnMut(R::Chunk<'_>) -> bool>(&mut self, mut each: G) {
        self.try_fold(
            (),
            |_, item| if each(item) { Continue(()) } else { Break(()) },
        )
    }

    #[inline]
    pub fn each<G: FnMut(R::Chunk<'_>)>(&mut self, mut each: G) {
        self.fold((), |_, item| each(item))
    }
}

impl<'d, R: Row, I> Split<'d, '_, R, I> {
    #[inline]
    pub fn table(&self) -> &Table {
        &self.state.table
    }
}

impl<'d, R: Row> Split<'d, '_, R, Item> {
    #[inline]
    pub fn count(&self) -> usize {
        self.state.table.count()
    }

    #[inline]
    pub fn has(&self, key: Key) -> bool {
        match unsafe { self.keys.get_semi_checked(key) } {
            Ok((_, table)) => self.state.table.index() == table,
            Err(_) => false,
        }
    }

    #[inline]
    pub fn try_fold<S, F: FnMut(S, R::Item<'_>) -> ControlFlow<S, S>>(
        &self,
        mut state: S,
        mut fold: F,
    ) -> Result<S, S> {
        let State {
            state: row,
            table,
            locks,
        } = self.state;
        let keys = table.keys.read();
        let Some(count) = NonZeroUsize::new(table.count()) else {
            return Ok(state);
        };
        Ok(lock(locks, table, |table| {
            let context = ItemContext::new(table, &keys);
            for i in 0..count.get() {
                let item = unsafe { R::item(row, context.with(i)) };
                state = match fold(state, item) {
                    Continue(state) => state,
                    Break(state) => return state,
                };
            }
            state
        }))
    }

    #[inline]
    pub fn fold<S, F: FnMut(S, R::Item<'_>) -> S>(
        &self,
        mut state: S,
        mut fold: F,
    ) -> Result<S, S> {
        let State {
            state: row,
            table,
            locks,
        } = self.state;
        let keys = table.keys.read();
        let Some(count) = NonZeroUsize::new(table.count()) else {
            return Ok(state);
        };
        Ok(lock(locks, table, |table| {
            let context = ItemContext::new(table, &keys);
            for i in 0..count.get() {
                let item = unsafe { R::item(row, context.with(i)) };
                state = fold(state, item);
            }
            state
        }))
    }

    #[inline]
    pub fn try_each<F: FnMut(R::Item<'_>) -> bool>(&self, mut each: F) -> bool {
        self.try_fold(
            (),
            |_, item| if each(item) { Continue(()) } else { Break(()) },
        )
        .is_ok()
    }

    #[inline]
    pub fn each<F: FnMut(R::Item<'_>)>(self, mut each: F) -> bool {
        self.fold((), |_, item| each(item)).is_ok()
    }

    pub fn try_find<T, G: FnOnce(Result<R::Item<'_>, Error>) -> T>(&self, key: Key, find: G) -> T {
        let State {
            state,
            table,
            locks,
        } = self.state;

        let keys = table.keys.read();
        // Check the slot while under the table lock to ensure that it doesn't move.
        let slot = match unsafe { self.keys.get_semi_checked(key) } {
            Ok(pair) if pair.1 == table.index() => pair.0,
            Ok(_) => {
                drop(keys);
                return find(Err(Error::KeyNotInSplit(key)));
            }
            Err(error) => {
                drop(keys);
                return find(Err(error));
            }
        };
        let Some(count) = NonZeroUsize::new(table.count()) else {
            drop(keys);
            return find(Err(Error::InvalidKey(key)));
        };
        let row = slot.row();
        if row < count.get() && keys.get(row) == Some(&key) {
            lock(locks, table, |table| {
                let context = ItemContext::new(table, &keys).with(row);
                find(Ok(unsafe { R::item(state, context) }))
            })
        } else {
            drop(keys);
            find(Err(Error::InvalidKey(key)))
        }
    }

    #[inline]
    pub fn find<T, G: FnOnce(R::Item<'_>) -> T>(&self, key: Key, find: G) -> Result<T, Error> {
        self.try_find(key, |item| item.map(find))
    }
}

impl<'d, R: Row> Split<'d, '_, R, Chunk> {
    #[inline]
    pub fn map<T, F: FnOnce(R::Chunk<'_>) -> T>(&self, map: F) -> Option<T> {
        let State {
            state: row,
            table,
            locks,
        } = self.state;
        let keys = table.keys.read();
        let Some(count) = NonZeroUsize::new(table.count()) else {
            return None;
        };
        Some(lock(locks, table, |table| {
            let context = ChunkContext::new(table, &keys, count);
            map(unsafe { R::chunk(row, context) })
        }))
    }
}

impl Access {
    #[inline]
    pub const fn identifier(&self) -> TypeId {
        match *self {
            Access::Read(identifier) => identifier,
            Access::Write(identifier) => identifier,
        }
    }

    #[inline]
    pub const fn read(&self) -> Self {
        Self::Read(self.identifier())
    }
}

impl By<()> {
    #[inline]
    pub fn key(&mut self, key: Key) {
        self.pair(key, ())
    }

    #[inline]
    pub fn keys<I: IntoIterator<Item = Key>>(&mut self, keys: I) {
        self.pairs(keys.into_iter().map(|key| (key, ())))
    }
}

impl<V> By<V> {
    #[inline]
    pub const fn new() -> Self {
        Self {
            pairs: Vec::new(),
            pending: Vec::new(),
            sorted: Vec::new(),
            errors: Vec::new(),
            indices: Vec::new(),
        }
    }

    #[inline]
    pub fn pair(&mut self, key: Key, value: V) {
        self.pairs.push((key, value));
    }

    #[inline]
    pub fn pairs<I: IntoIterator<Item = (Key, V)>>(&mut self, pairs: I) {
        self.pairs.extend(pairs);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn iter(&self) -> impl FullIterator<Item = (Key, &V)> {
        self.pairs.iter().map(|(key, value)| (*key, value))
    }

    #[inline]
    pub fn iter_mut(&mut self) -> impl FullIterator<Item = (Key, &mut V)> {
        self.pairs.iter_mut().map(|(key, value)| (*key, value))
    }

    #[inline]
    pub fn drain(&mut self) -> impl FullIterator<Item = (Key, V)> + '_ {
        self.pairs.drain(..)
    }

    #[inline]
    pub fn clear(&mut self) {
        self.pairs.clear();
    }
}

impl<V> Default for By<V> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn find_state<R: Row>(states: &[State<R>], table: u32) -> Option<(usize, &State<R>)> {
    match states.binary_search_by_key(&table, |state| state.table.index()) {
        Ok(index) => Some((index, unsafe { get_unchecked(states, index) })),
        Err(_) => None,
    }
}

#[inline]
fn try_lock<T, S, F: FnOnce(S, &Table) -> T>(
    state: S,
    locks: &[(usize, Access)],
    table: &Table,
    with: F,
) -> Result<T, S> {
    match locks.split_first() {
        Some((&(index, access), rest)) => {
            let column = unsafe { get_unchecked(table.columns(), index) };
            debug_assert_eq!(access.identifier(), column.meta().identifier());
            debug_assert!(column.meta().size() > 0);
            match access {
                Access::Read(_) => match column.data().try_read() {
                    Some(_guard) => try_lock(state, rest, table, with),
                    None => Err(state),
                },
                Access::Write(_) => match column.data().try_write() {
                    Some(_guard) => try_lock(state, rest, table, with),
                    None => Err(state),
                },
            }
        }
        None => Ok(with(state, table)),
    }
}

#[inline]
fn lock<T, F: FnOnce(&Table) -> T>(locks: &[(usize, Access)], table: &Table, with: F) -> T {
    match locks.split_first() {
        Some((&(index, access), rest)) => {
            let column = unsafe { get_unchecked(table.columns(), index) };
            debug_assert_eq!(access.identifier(), column.meta().identifier());
            debug_assert!(column.meta().size() > 0);

            match access {
                Access::Read(_) => {
                    let _guard = column.data().read();
                    lock(rest, table, with)
                }
                Access::Write(_) => {
                    let _guard = column.data().write();
                    lock(rest, table, with)
                }
            }
        }
        None => with(table),
    }
}
