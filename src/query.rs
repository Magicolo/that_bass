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
    collections::HashSet,
    marker::PhantomData,
    mem::swap,
    ops::ControlFlow::{self, *},
    vec::Drain,
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
    pairs: Vec<(Key, V)>,
    pending: Vec<(Key, V, &'d Slot, u32)>,
    slots: Vec<Vec<(Key, V, &'d Slot)>>,
    indices: Vec<u32>,
}

struct State<'d, R: Row> {
    state: R::State,
    table: &'d Table,
    indices: Box<[(usize, Access)]>,
}

struct Errors<'d, 'a, R: Row, V> {
    database: &'d Database,
    states: &'a mut [State<'d, R>],
    pairs: Drain<'a, (Key, V)>,
    pending: Drain<'a, (Key, V, &'d Slot, u32)>,
    slots: &'a mut Vec<Vec<(Key, V, &'d Slot)>>,
    indices: &'a mut Vec<u32>,
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

            let index = self.states.len() as _;
            self.indices.push(index);
            self.states.push(State {
                state,
                table,
                indices: indices.into_boxed_slice(),
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
        by.pairs.iter().filter(|&&(key, ..)| self.has(key)).count()
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
                let item = unsafe { R::item(row, context.with(i as _)) };
                state = fold(state, item)?;
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
                        unsafe { get_unchecked_mut(&mut by.slots, index as usize) }.clear();
                    }
                    break state;
                }
            };
            if by.pending.len() == 0 {
                break state;
            }
        }
    }

    #[inline]
    pub fn fold<S, G: FnMut(S, R::Item<'_>) -> S>(&mut self, state: S, mut fold: G) -> S {
        self.update();
        self.guards(state, |mut state, _, row, _, keys, columns| {
            debug_assert!(keys.len() > 0);
            let context = ItemContext::new(keys, columns);
            for i in 0..keys.len() {
                let item = unsafe { R::item(row, context.with(i as _)) };
                state = fold(state, item);
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
        loop {
            state = self.fold_by_sorted(by, state, &mut fold);
            by.indices.clear();
            if by.pending.len() == 0 {
                break state;
            }
        }
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
                    let item = unsafe { R::item(state, context.with(row)) };
                    find(Ok(item))
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
        for (key, value) in by.pairs.drain(..) {
            match self.database.keys().get(key) {
                Ok((slot, table)) => match self.sort(&mut by.slots, key, value, slot, table) {
                    Some((value, error)) => state = fold(state, value, Err(error)),
                    None => {}
                },
                Err(error) => state = fold(state, value, Err(error)),
            }
        }

        for (key, value, slot, table) in by.pending.drain(..) {
            match self.sort(&mut by.slots, key, value, slot, table) {
                Some((value, error)) => state = fold(state, value, Err(error)),
                None => {}
            }
        }

        swap(&mut self.indices, &mut by.indices);
        let state = self.guards(state, |mut state, index, row, table, keys, columns| {
            let context = ItemContext::new(keys, columns);
            let slots = unsafe { get_unchecked_mut(&mut by.slots, index as usize) };
            for (key, value, slot) in slots.drain(..) {
                // The key is allowed to move within its table (such as with a swap as part of a remove).
                match slot.table(key.generation()) {
                    Ok(table_index) if table.index() == table_index => {
                        let item = unsafe { R::item(row, context.with(slot.row() as _)) };
                        state = fold(state, value, Ok(item));
                    }
                    // The key has moved to another table between the last moment the slot indices were read and now.
                    Ok(table_index) => by.pending.push((key, value, slot, table_index)),
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
        for (key, value) in by.pairs.drain(..) {
            match self.database.keys().get(key) {
                Ok((slot, table)) => match self.sort(&mut by.slots, key, value, slot, table) {
                    Some((value, error)) => state = fold(state, value, Err(error))?,
                    None => {}
                },
                Err(error) => state = fold(state, value, Err(error))?,
            }
        }

        for (key, value, slot, table) in by.pending.drain(..) {
            match self.sort(&mut by.slots, key, value, slot, table) {
                Some((value, error)) => state = fold(state, value, Err(error))?,
                None => {}
            }
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
                        let item = unsafe { R::item(row, context.with(slot.row() as _)) };
                        state = fold(state, value, Ok(item))?;
                    }
                    // The key has moved to another table between the last moment the slot indices were read and now.
                    Ok(table_index) => by.pending.push((key, value, slot, table_index)),
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
    fn sort<V>(
        &mut self,
        slots: &mut Vec<Vec<(Key, V, &'d Slot)>>,
        key: Key,
        value: V,
        slot: &'d Slot,
        table: u32,
    ) -> Option<(V, Error)> {
        match find_state(&mut self.states, table) {
            Some((index, _)) => {
                let slots = unsafe { get_unchecked_mut(slots, index) };
                if slots.len() == 0 {
                    self.indices.push(index as _);
                }
                slots.push((key, value, slot));
                None
            }
            None => Some((value, Error::KeyNotInQuery(key))),
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
            let chunk = unsafe { R::chunk(row, ChunkContext::new(keys, columns)) };
            fold(state, chunk)
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
            let chunk = unsafe { R::chunk(row, ChunkContext::new(keys, columns)) };
            fold(state, chunk)
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
    pub fn key(&mut self, key: Key) {
        self.pair(key, ())
    }

    #[inline]
    pub fn keys<I: IntoIterator<Item = Key>>(&mut self, keys: I) {
        self.pairs(keys.into_iter().map(|key| (key, ())))
    }
}

impl<V> By<'_, V> {
    #[inline]
    pub const fn new() -> Self {
        By {
            pairs: Vec::new(),
            pending: Vec::new(),
            slots: Vec::new(),
            indices: Vec::new(),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.pairs.len()
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
    pub fn clear(&mut self) {
        self.pairs.clear();
    }
}

impl<V> Default for By<'_, V> {
    fn default() -> Self {
        Self {
            pairs: Default::default(),
            pending: Default::default(),
            slots: Default::default(),
            indices: Default::default(),
        }
    }
}

impl<'d, R: Row, V> Errors<'d, '_, R, V> {
    fn find(&mut self, key: Key, value: V, slot: &'d Slot, table: u32) -> Option<(V, Error)> {
        match find_state(self.states, table) {
            Some((index, _)) => {
                let slots = unsafe { get_unchecked_mut(&mut self.slots, index) };
                if slots.len() == 0 {
                    self.indices.push(index as _);
                }
                slots.push((key, value, slot));
                None
            }
            None => Some((value, Error::KeyNotInQuery(key))),
        }
    }
}

impl<R: Row, V> Iterator for Errors<'_, '_, R, V> {
    type Item = (V, Error);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((key, value)) = self.pairs.next() {
            match self.database.keys().get(key) {
                Ok((slot, table)) => match self.find(key, value, slot, table) {
                    Some(pair) => return Some(pair),
                    None => {}
                },
                Err(error) => return Some((value, error)),
            }
        }

        while let Some((key, value, slot, table)) = self.pending.next() {
            match self.find(key, value, slot, table) {
                Some(pair) => return Some(pair),
                None => {}
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

fn try_lock<T, S, F: FnOnce(S) -> T>(
    state: S,
    mut indices: &[(usize, Access)],
    inner: &table::Inner,
    with: F,
) -> Result<T, S> {
    while let Some((&(index, access), rest)) = indices.split_first() {
        let column = unsafe { get_unchecked(inner.columns(), index) };
        debug_assert_eq!(access.identifier(), column.meta().identifier());
        if column.meta().size == 0 {
            // No need to recurse since no lock needs to be accumulated.
            indices = rest;
            continue;
        }

        match access {
            Access::Read(_) => match column.data().try_read() {
                Some(_guard) => return try_lock(state, rest, inner, with),
                None => return Err(state),
            },
            Access::Write(_) => match column.data().try_write() {
                Some(_guard) => return try_lock(state, rest, inner, with),
                None => return Err(state),
            },
        }
    }
    Ok(with(state))
}

fn lock<T, F: FnOnce() -> T>(mut indices: &[(usize, Access)], inner: &table::Inner, with: F) -> T {
    while let Some((&(index, access), rest)) = indices.split_first() {
        let column = unsafe { get_unchecked(inner.columns(), index) };
        debug_assert_eq!(access.identifier(), column.meta().identifier());
        if column.meta().size == 0 {
            // No need to recurse since no lock needs to be accumulated.
            indices = rest;
            continue;
        }

        match access {
            Access::Read(_) => {
                let _guard = column.data().read();
                return lock(rest, inner, with);
            }
            Access::Write(_) => {
                let _guard = column.data().write();
                return lock(rest, inner, with);
            }
        }
    }
    with()
}
