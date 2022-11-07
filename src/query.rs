use crate::{
    core::{
        utility::{fold_swap, get_unchecked, get_unchecked_mut, try_fold_swap},
        FullIterator,
    },
    filter::Filter,
    key::{Key, Slot},
    row::{Access, ChunkContext, DeclareContext, InitializeContext, ItemContext, Row},
    table::{self, Column, Table},
    Database, Error,
};
use std::{
    any::TypeId,
    collections::{HashSet, VecDeque},
    marker::PhantomData,
    mem::swap,
    ops::ControlFlow::{self, *},
};

// TODO: Share some state... But how to `update` without locking when accessing `states`?
pub struct Query<'d, R: Row, F = (), I = Item> {
    database: &'d Database,
    index: usize,
    indices: Vec<u32>,         // May be reordered.
    states: Vec<State<'d, R>>, // Must remain sorted by `state.table.index()`.
    accesses: HashSet<Access>,
    _marker: PhantomData<fn(F, I)>,
}

pub struct Item;
pub struct Chunk;

pub struct By<'d, V> {
    database: &'d Database,
    pairs: VecDeque<(Key, V, &'d Slot, u32)>,
    slots: Vec<Vec<(Key, V, &'d Slot)>>,
    indices: Vec<u32>,
}

struct State<'d, R: Row> {
    state: R::State,
    table: &'d Table,
    indices: Vec<(usize, Access)>,
}

struct Errors<'d, 'a, R: Row, V> {
    states: &'a mut [State<'d, R>],
    by: &'a mut By<'d, V>,
}

impl Database {
    pub fn query<R: Row>(&self) -> Result<Query<'_, R>, Error> {
        Ok(Query {
            database: self,
            indices: Vec::new(),
            states: Vec::new(),
            accesses: DeclareContext::accesses::<R>()?,
            index: 0,
            _marker: PhantomData,
        })
    }

    pub fn by<V>(&self) -> By<V> {
        By {
            database: self,
            pairs: VecDeque::new(),
            slots: Vec::new(),
            indices: Vec::new(),
        }
    }
}

impl<'d, R: Row, F: Filter, I> Query<'d, R, F, I> {
    pub fn tables(&mut self) -> impl FullIterator<Item = &'d Table> + '_ {
        self.update();
        self.states.iter().map(|state| state.table)
    }

    pub fn chunk(self) -> Query<'d, R, F, Chunk> {
        Query {
            database: self.database,
            indices: self.indices,
            states: self.states,
            accesses: self.accesses,
            index: self.index,
            _marker: PhantomData,
        }
    }

    pub fn read(self) -> Query<'d, R::Read, F, I> {
        let states = self
            .states
            .into_iter()
            .map(|mut state| {
                for (_, access) in state.indices.iter_mut() {
                    *access = access.read();
                }
                State {
                    state: R::read(state.state),
                    table: state.table,
                    indices: state.indices,
                }
            })
            .collect();
        let accesses = self
            .accesses
            .into_iter()
            .map(|access| access.read())
            .collect();
        Query {
            database: self.database,
            indices: self.indices,
            states: states,
            accesses,
            index: self.index,
            _marker: PhantomData,
        }
    }

    pub fn filter<G: Filter>(mut self) -> Query<'d, R, (F, G), I> {
        self.states
            .retain(|state| G::filter(state.table, self.database));
        self.indices.clear();
        self.indices.extend(0..self.states.len() as u32);
        Query {
            database: self.database,
            indices: self.indices,
            states: self.states,
            accesses: self.accesses,
            index: self.index,
            _marker: PhantomData,
        }
    }

    pub(crate) fn update(&mut self) {
        let tables = self.database.tables();
        while let Ok(table) = tables.get(self.index) {
            self.index += 1;
            let _ = self.try_add(table);
        }
    }

    #[inline]
    pub(crate) fn try_guards<S>(
        &mut self,
        state: S,
        fold: impl FnMut(S, u32, &mut R::State, &Table, &[Key], &[Column]) -> ControlFlow<S, S>,
    ) -> ControlFlow<S, S> {
        try_fold_swap(
            &mut self.indices,
            state,
            (&mut self.states, fold),
            |state, (states, fold), index| {
                let State {
                    state: row,
                    table,
                    indices,
                } = unsafe { get_unchecked_mut(states, *index as usize) };
                let Some(inner) = table.inner.try_read() else {
                    return Err(state);
                };
                let keys = inner.keys();
                if keys.len() == 0 {
                    return Ok(Continue(state));
                }
                try_lock(state, indices, &inner, |state| {
                    fold(state, *index, row, table, keys, inner.columns())
                })
            },
            |state, (states, fold), index| {
                let State {
                    state: row,
                    table,
                    indices,
                } = unsafe { get_unchecked_mut(states, *index as usize) };
                let inner = table.inner.read();
                let keys = inner.keys();
                if keys.len() == 0 {
                    return Continue(state);
                }
                lock(indices, &inner, || {
                    fold(state, *index, row, table, keys, inner.columns())
                })
            },
        )
    }

    #[inline]
    pub(crate) fn guards<S>(
        &mut self,
        state: S,
        fold: impl FnMut(S, u32, &mut R::State, &Table, &[Key], &[Column]) -> S,
    ) -> S {
        fold_swap(
            &mut self.indices,
            state,
            (&mut self.states, fold),
            |state, (states, fold), index| {
                let State {
                    state: row,
                    table,
                    indices,
                } = unsafe { get_unchecked_mut(states, *index as usize) };
                let Some(inner) = table.inner.try_read() else {
                    return Err(state);
                };
                let keys = inner.keys();
                if keys.len() == 0 {
                    return Ok(state);
                }
                try_lock(state, indices, &inner, |state| {
                    fold(state, *index, row, table, keys, inner.columns())
                })
            },
            |state, (states, fold), index| {
                let State {
                    state: row,
                    table,
                    indices,
                } = unsafe { get_unchecked_mut(states, *index as usize) };
                let inner = table.inner.read();
                let keys = inner.keys();
                if keys.len() == 0 {
                    return state;
                }
                lock(indices, &inner, || {
                    fold(state, *index, row, table, keys, inner.columns())
                })
            },
        )
    }

    fn try_add(&mut self, table: &'d Table) -> Result<(), Error> {
        if F::filter(table, self.database) {
            // Initialize first to save some work if it fails.
            let state = R::initialize(InitializeContext::new(table))?;
            let mut indices = Vec::new();
            for &access in self.accesses.iter() {
                if let Ok(index) = table.column_with(access.identifier()) {
                    indices.push((index, access));
                }
            }
            // The sorting of indices ensures that there cannot be a deadlock between `Rows` when locking multiple columns as long as this
            // happens while holding at most 1 table lock.
            indices.sort_unstable_by_key(|&(index, _)| index);
            indices.shrink_to_fit();

            let index = self.states.len() as _;
            self.indices.push(index);
            self.states.push(State {
                state,
                table,
                indices,
            });
            Ok(())
        } else {
            Err(Error::InvalidTable)
        }
    }
}

impl<'d, R: Row, F: Filter> Query<'d, R, F, Item> {
    #[inline]
    pub fn count(&mut self) -> usize {
        self.tables()
            .fold(0, |sum, table| sum + table.inner.read().count())
    }

    #[inline]
    pub fn count_by<V>(&mut self, by: &By<V>) -> usize {
        self.update();
        by.pairs
            .iter()
            .filter(|&&(.., table)| find_state(&mut self.states, table).is_some())
            .count()
    }

    #[inline]
    pub fn try_fold<S, G: FnMut(S, R::Item<'_>) -> ControlFlow<S, S>>(
        &mut self,
        state: S,
        mut fold: G,
    ) -> S {
        self.update();
        let flow = self.try_guards(state, |mut state, _, row, _, keys, columns| {
            debug_assert!(keys.len() > 0);
            let context = ItemContext::new(keys, columns);
            for i in 0..keys.len() {
                state = fold(state, R::item(row, context.with(i as _)))?;
            }
            Continue(state)
        });
        match flow {
            Continue(state) => state,
            Break(state) => state,
        }
    }

    #[inline]
    pub fn try_fold_by<V, S, G: FnMut(S, V, Result<R::Item<'_>, Error>) -> ControlFlow<S, S>>(
        &mut self,
        by: &mut By<'d, V>,
        mut state: S,
        mut fold: G,
    ) -> S {
        self.update();
        while by.pairs.len() > 0 {
            state = match self.try_fold_by_sorted(by, state, &mut fold) {
                Continue(state) => {
                    by.indices.clear();
                    state
                }
                Break(state) => {
                    // Fold was interrupted. Clean up.
                    by.pairs.clear();
                    for index in by.indices.drain(..) {
                        unsafe { get_unchecked_mut(&mut by.slots, index as usize) }.clear();
                    }
                    return state;
                }
            }
        }
        state
    }

    #[inline]
    pub fn fold<S, G: FnMut(S, R::Item<'_>) -> S>(&mut self, state: S, mut fold: G) -> S {
        self.update();
        self.guards(state, |mut state, _, row, _, keys, columns| {
            debug_assert!(keys.len() > 0);
            let context = ItemContext::new(keys, columns);
            for i in 0..keys.len() {
                state = fold(state, R::item(row, context.with(i as _)));
            }
            state
        })
    }

    #[inline]
    pub fn fold_by<V, S, G: FnMut(S, V, Result<R::Item<'_>, Error>) -> S>(
        &mut self,
        by: &mut By<'d, V>,
        mut state: S,
        mut fold: G,
    ) -> S {
        self.update();
        while by.pairs.len() > 0 {
            state = self.fold_by_sorted(by, state, &mut fold);
            by.indices.clear();
        }
        state
    }

    #[inline]
    pub fn fold_by_ok<V, S, G: FnMut(S, V, R::Item<'_>) -> S>(
        &mut self,
        by: &mut By<'d, V>,
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
        by: &mut By<'d, V>,
        mut each: G,
    ) {
        self.fold_by(by, (), |_, value, item| each(value, item))
    }

    #[inline]
    pub fn each_by_ok<V, G: FnMut(V, R::Item<'_>)>(&mut self, by: &mut By<'d, V>, mut each: G) {
        self.fold_by(by, (), |_, value, item| {
            if let Ok(item) = item {
                each(value, item);
            }
        })
    }

    #[inline]
    pub fn has(&mut self, key: Key) -> bool {
        self.update();
        match self.database.keys().get(key) {
            Ok((_, table)) => find_state(&mut self.states, table).is_some(),
            Err(_) => false,
        }
    }

    pub fn try_find<T, G: FnOnce(Result<R::Item<'_>, Error>) -> T>(
        &mut self,
        key: Key,
        find: G,
    ) -> T {
        self.update();
        let (slot, mut old_table) = match self.database.keys().get(key) {
            Ok(slot) => slot,
            Err(error) => return find(Err(error)),
        };
        loop {
            let State {
                state,
                table,
                indices,
            } = match find_state(&mut self.states, old_table) {
                Some((_, state)) => state,
                None => break find(Err(Error::KeyNotInQuery(key))),
            };

            let inner = table.inner.read();
            // The key must be checked again while holding the table lock to be sure is has not been moved/destroyed since last read.
            let new_table = match slot.table(key.generation()) {
                Ok(new_table) => new_table,
                // The `key` has just been destroyed.
                // - Do not call `find` in here since it would hold locks for longer.
                Err(error) => break find(Err(error)),
            };
            if new_table == table.index() {
                debug_assert_eq!(old_table, table.index());
                break lock(indices, &inner, || {
                    let row = slot.row() as usize;
                    let keys = inner.keys();
                    debug_assert_eq!(keys.get(row).copied(), Some(key));
                    let context = ItemContext::new(keys, inner.columns());
                    find(Ok(R::item(state, context.with(row))))
                });
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
        by: &mut By<'d, V>,
        mut state: S,
        fold: &mut G,
    ) -> S {
        for (value, error) in self.sort(by) {
            state = fold(state, value, Err(error));
        }

        swap(&mut self.indices, &mut by.indices);
        let state = self.guards(state, |mut state, index, row, table, keys, columns| {
            let context = ItemContext::new(keys, columns);
            let slots = unsafe { get_unchecked_mut(&mut by.slots, index as usize) };
            for (key, value, slot) in slots.drain(..) {
                // The key is allowed to move within its table (such as with a swap as part of a remove).
                match slot.table(key.generation()) {
                    Ok(table_index) if table.index() == table_index => {
                        let item = R::item(row, context.with(slot.row() as _));
                        state = fold(state, value, Ok(item));
                    }
                    // The key has moved to another table between the last moment the slot indices were read and now.
                    Ok(table_index) => by.pairs.push_back((key, value, slot, table_index)),
                    Err(error) => state = fold(state, value, Err(error)),
                }
            }
            state
        });
        swap(&mut self.indices, &mut by.indices);
        state
    }

    fn try_fold_by_sorted<V, S, G: FnMut(S, V, Result<R::Item<'_>, Error>) -> ControlFlow<S, S>>(
        &mut self,
        by: &mut By<'d, V>,
        mut state: S,
        fold: &mut G,
    ) -> ControlFlow<S, S> {
        for (value, error) in self.sort(by) {
            state = fold(state, value, Err(error))?;
        }

        swap(&mut self.indices, &mut by.indices);
        // TODO: No need to lock the columns if `by.slots[index].is_empty()` after being filtered.
        let flow = self.try_guards(state, |mut state, index, row, table, keys, columns| {
            let context = ItemContext::new(keys, columns);
            let slots = unsafe { get_unchecked_mut(&mut by.slots, index as usize) };
            for (key, value, slot) in slots.drain(..) {
                // The key is allowed to move within its table (such as with a swap as part of a remove).
                match slot.table(key.generation()) {
                    Ok(table_index) if table.index() == table_index => {
                        let item = R::item(row, context.with(slot.row() as _));
                        state = fold(state, value, Ok(item))?;
                    }
                    // The key has moved to another table between the last moment the slot indices were read and now.
                    Ok(table_index) => by.pairs.push_back((key, value, slot, table_index)),
                    Err(error) => state = fold(state, value, Err(error))?,
                }
            }
            Continue(state)
        });
        swap(&mut self.indices, &mut by.indices);
        flow
    }

    /// Sorts keys by state index such that table locks can be used for (hopefully) more than one key at a time.
    #[inline]
    fn sort<'a, V>(&'a mut self, by: &'a mut By<'d, V>) -> Errors<'d, 'a, R, V> {
        while by.slots.len() < self.states.len() {
            by.slots.push(Vec::new());
        }
        // Sort keys by state such that table locks can be used for (hopefully) more than one key at a time.
        Errors {
            states: &mut self.states,
            by: by,
        }
    }
}

impl<'d, R: Row, F: Filter> Query<'d, R, F, Chunk> {
    #[inline]
    pub fn count(&mut self) -> usize {
        self.tables()
            .filter(|table| table.inner.read().count() > 0)
            .count()
    }

    #[inline]
    pub fn try_fold<S, G: FnMut(S, R::Chunk<'_>) -> ControlFlow<S, S>>(
        &mut self,
        state: S,
        mut fold: G,
    ) -> S {
        self.update();
        let flow = self.try_guards(state, |state, _, row, _, keys, columns| {
            fold(state, R::chunk(row, ChunkContext::new(keys, columns)))
        });
        match flow {
            Continue(state) => state,
            Break(state) => state,
        }
    }

    #[inline]
    pub fn fold<S, G: FnMut(S, R::Chunk<'_>) -> S>(&mut self, state: S, mut fold: G) -> S {
        self.update();
        self.guards(state, |state, _, row, _, keys, columns| {
            fold(state, R::chunk(row, ChunkContext::new(keys, columns)))
        })
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

impl By<'_, ()> {
    #[inline]
    pub fn key(&mut self, key: Key) -> bool {
        self.pair(key, ())
    }

    #[inline]
    pub fn keys<I: IntoIterator<Item = Key>>(&mut self, keys: I) -> usize {
        self.pairs(keys.into_iter().map(|key| (key, ())))
    }
}

impl<V> By<'_, V> {
    #[inline]
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    #[inline]
    pub fn pair(&mut self, key: Key, value: V) -> bool {
        match self.database.keys().get(key) {
            Ok((slot, table)) => {
                self.pairs.push_back((key, value, slot, table));
                true
            }
            Err(_) => false,
        }
    }

    #[inline]
    pub fn pairs<I: IntoIterator<Item = (Key, V)>>(&mut self, pairs: I) -> usize {
        pairs
            .into_iter()
            .filter_map(|(key, value)| self.pair(key, value).then_some(()))
            .count()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.pairs.clear();
    }
}

impl<R: Row, V> Iterator for Errors<'_, '_, R, V> {
    type Item = (V, Error);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((key, value, slot, table)) = self.by.pairs.pop_front() {
            match find_state(self.states, table) {
                Some((index, _)) => {
                    let slots = unsafe { get_unchecked_mut(&mut self.by.slots, index) };
                    if slots.len() == 0 {
                        self.by.indices.push(index as _);
                    }
                    slots.push((key, value, slot));
                }
                None => return Some((value, Error::KeyNotInQuery(key))),
            }
        }
        None
    }
}

#[inline]
fn find_state<'d, 'a, R: Row>(
    states: &'a mut [State<'d, R>],
    table: u32,
) -> Option<(usize, &'a mut State<'d, R>)> {
    match states.binary_search_by_key(&table, |state| state.table.index()) {
        Ok(index) => Some((index, unsafe { get_unchecked_mut(states, index) })),
        Err(_) => None,
    }
}

fn try_lock<T, S>(
    state: S,
    indices: &[(usize, Access)],
    inner: &table::Inner,
    with: impl FnOnce(S) -> T,
) -> Result<T, S> {
    match indices.split_first() {
        Some((&(index, access), rest)) => {
            let column = unsafe { get_unchecked(inner.columns(), index) };
            debug_assert_eq!(access.identifier(), column.meta().identifier());
            if column.meta().size == 0 {
                // TODO: No need to recurse here.
                try_lock(state, rest, inner, with)
            } else {
                match access {
                    Access::Read(_) => {
                        let Some(_guard) = column.data().try_read() else {
                            return Err(state);
                        };
                        try_lock(state, rest, inner, with)
                    }
                    Access::Write(_) => {
                        let Some(_guard) = column.data().try_write() else {
                            return Err(state);
                        };
                        try_lock(state, rest, inner, with)
                    }
                }
            }
        }
        None => Ok(with(state)),
    }
}

fn lock<T>(indices: &[(usize, Access)], inner: &table::Inner, with: impl FnOnce() -> T) -> T {
    match indices.split_first() {
        Some((&(index, access), rest)) => {
            let column = unsafe { get_unchecked(inner.columns(), index) };
            debug_assert_eq!(access.identifier(), column.meta().identifier());
            if column.meta().size == 0 {
                // TODO: No need to recurse here.
                lock(rest, inner, with)
            } else {
                match access {
                    Access::Read(_) => {
                        let _guard = column.data().read();
                        lock(rest, inner, with)
                    }
                    Access::Write(_) => {
                        let _guard = column.data().write();
                        lock(rest, inner, with)
                    }
                }
            }
        }
        None => with(),
    }
}
