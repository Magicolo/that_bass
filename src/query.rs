use crate::{
    core::{
        utility::{fold_swap, try_fold_swap},
        FullIterator,
    },
    filter::Filter,
    key::{Key, Slot},
    row::{Access, ChunkContext, DeclareContext, InitializeContext, ItemContext, Row},
    table::{self, Table},
    Database, Error,
};
use std::{
    any::TypeId,
    collections::{HashMap, HashSet, VecDeque},
    marker::PhantomData,
    mem::swap,
    ops::ControlFlow::{self, *},
    ptr::NonNull,
};

// TODO: Share some state... But how to `update` without locking when accessing `states`?
pub struct Query<'d, R: Row, F = (), I = Item> {
    database: &'d Database,
    index: usize,
    indices: HashMap<u32, u32>, // From table index to state index.
    states: (Vec<u32>, Vec<State<'d, R>>),
    accesses: HashSet<Access>,
    pointers: Vec<NonNull<()>>,
    _marker: PhantomData<fn(F, I)>,
}

pub struct Item;
pub struct Chunk;

pub struct By<'d, V> {
    database: &'d Database,
    pairs: VecDeque<(Key, V, &'d Slot, u32)>,
    slots: Vec<Vec<(Key, V, &'d Slot)>>,
    states: Vec<u32>,
}

struct State<'d, R: Row> {
    state: R::State,
    table: &'d Table,
    indices: Vec<(usize, Access)>,
}

struct Errors<'d, 'a, V> {
    indices: &'a HashMap<u32, u32>,
    by: &'a mut By<'d, V>,
}

impl Database {
    pub fn query<R: Row>(&self) -> Result<Query<'_, R>, Error> {
        // Detects violations of rust's invariants.
        let mut accesses = HashSet::new();
        let context = DeclareContext(&mut accesses);
        R::declare(context)?;
        Ok(Query {
            database: self,
            indices: HashMap::new(),
            states: (Vec::new(), Vec::new()),
            accesses,
            pointers: Vec::new(),
            index: 0,
            _marker: PhantomData,
        })
    }

    pub fn by<V>(&self) -> By<V> {
        By {
            database: self,
            pairs: VecDeque::new(),
            slots: Vec::new(),
            states: Vec::new(),
        }
    }
}

impl<'d, R: Row, F: Filter, I> Query<'d, R, F, I> {
    pub fn tables(&mut self) -> impl FullIterator<Item = &'d Table> + '_ {
        self.update();
        self.states.1.iter().map(|state| state.table)
    }

    pub fn chunk(self) -> Query<'d, R, F, Chunk> {
        Query {
            database: self.database,
            indices: self.indices,
            states: self.states,
            accesses: self.accesses,
            pointers: self.pointers,
            index: self.index,
            _marker: PhantomData,
        }
    }

    pub fn read(self) -> Query<'d, R::Read, F, I> {
        debug_assert!(self.pointers.is_empty());

        let states = self
            .states
            .1
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
            states: (self.states.0, states),
            accesses,
            pointers: self.pointers,
            index: self.index,
            _marker: PhantomData,
        }
    }

    pub fn filter<G: Filter>(mut self) -> Query<'d, R, (F, G), I> {
        debug_assert!(self.pointers.is_empty());

        for state in self.states.1.iter() {
            if G::filter(state.table, self.database) {
                continue;
            } else {
                self.indices.remove(&state.table.index());
            }
        }
        self.states.0.clear();
        self.states.0.extend(self.indices.values());

        Query {
            database: self.database,
            indices: self.indices,
            states: self.states,
            accesses: self.accesses,
            pointers: self.pointers,
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
        fold: impl FnMut(
            S,
            u32,
            &mut R::State,
            &[NonNull<()>],
            &Table,
            &table::Inner,
        ) -> ControlFlow<S, S>,
    ) -> ControlFlow<S, S> {
        try_fold_swap(
            &mut self.states.0,
            state,
            (&mut self.states.1, &mut self.pointers, fold),
            |state, (states, pointers, fold), index| {
                unsafe { states.get_unchecked_mut(*index as usize) }.try_lock(
                    state,
                    pointers,
                    |state, row, pointers, table, inner| {
                        fold(state, *index, row, pointers, table, inner)
                    },
                )
            },
            |state, (states, pointers, fold), index| {
                unsafe { states.get_unchecked_mut(*index as usize) }
                    .lock(pointers, |row, pointers, table, inner| {
                        fold(state, *index, row, pointers, table, inner)
                    })
            },
        )
    }

    #[inline]
    pub(crate) fn guards<S>(
        &mut self,
        state: S,
        fold: impl FnMut(S, u32, &mut R::State, &[NonNull<()>], &Table, &table::Inner) -> S,
    ) -> S {
        fold_swap(
            &mut self.states.0,
            state,
            (&mut self.states.1, &mut self.pointers, fold),
            |state, (states, pointers, fold), index| {
                unsafe { states.get_unchecked_mut(*index as usize) }.try_lock(
                    state,
                    pointers,
                    |state, row, pointers, table, inner| {
                        fold(state, *index, row, pointers, table, inner)
                    },
                )
            },
            |state, (states, pointers, fold), index| {
                unsafe { states.get_unchecked_mut(*index as usize) }
                    .lock(pointers, |row, pointers, table, inner| {
                        fold(state, *index, row, pointers, table, inner)
                    })
            },
        )
    }

    fn try_add(&mut self, table: &'d Table) -> Result<(), Error> {
        if F::filter(table, self.database) {
            let mut indices = Vec::with_capacity(self.accesses.len());
            for &access in self.accesses.iter() {
                if let Ok(index) = table.store_with(access.identifier()) {
                    indices.push((index, access));
                }
            }

            // The sorting of indices ensures that there cannot be a deadlock between `Rows` when locking multiple stores as long as this
            // happens while holding at most 1 table lock.
            indices.sort_unstable_by_key(|&(index, _)| index);
            let map = indices
                .iter()
                .enumerate()
                .map(|(i, &(_, access))| (access, i))
                .collect();
            let state = R::initialize(InitializeContext(&map))?;
            let index = self.states.1.len() as _;
            self.indices.insert(table.index(), index);
            self.states.0.push(index);
            self.states.1.push(State {
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

impl<'d, R: Row> State<'d, R> {
    #[inline]
    fn try_lock<S, T>(
        &mut self,
        state: S,
        pointers: &mut Vec<NonNull<()>>,
        with: impl FnOnce(S, &mut R::State, &[NonNull<()>], &Table, &table::Inner) -> T,
    ) -> Result<T, S> {
        fn next<S, T>(
            state: S,
            indices: &[(usize, Access)],
            pointers: &mut Vec<NonNull<()>>,
            inner: &table::Inner,
            with: impl FnOnce(S, &[NonNull<()>], &table::Inner) -> T,
        ) -> Result<T, S> {
            match indices.split_first() {
                Some((&(index, access), rest)) => {
                    let store = unsafe { inner.stores.get_unchecked(index) };
                    debug_assert_eq!(access.identifier(), store.meta().identifier());
                    if store.meta().size == 0 {
                        pointers.push(unsafe { *store.data().data_ptr() });
                        next(state, rest, pointers, inner, with)
                    } else {
                        match access {
                            Access::Read(_) => {
                                let guard = match store.data().try_read() {
                                    Some(guard) => guard,
                                    None => return Err(state),
                                };
                                pointers.push(*guard);
                                let result = next(state, rest, pointers, inner, with);
                                drop(guard);
                                result
                            }
                            Access::Write(_) => {
                                let guard = match store.data().try_write() {
                                    Some(guard) => guard,
                                    None => return Err(state),
                                };
                                pointers.push(*guard);
                                let result = next(state, rest, pointers, inner, with);
                                drop(guard);
                                result
                            }
                        }
                    }
                }
                None => {
                    let value = with(state, pointers, inner);
                    pointers.clear();
                    Ok(value)
                }
            }
        }

        let Some(inner) = self.table.inner.try_read() else {
            return Err(state);
        };
        next(
            state,
            &self.indices,
            pointers,
            &inner,
            |state, pointers, inner| with(state, &mut self.state, pointers, self.table, inner),
        )
    }

    #[inline]
    fn lock<T>(
        &mut self,
        pointers: &mut Vec<NonNull<()>>,
        with: impl FnOnce(&mut R::State, &[NonNull<()>], &Table, &table::Inner) -> T,
    ) -> T {
        #[inline]
        fn next<T>(
            indices: &[(usize, Access)],
            pointers: &mut Vec<NonNull<()>>,
            inner: &table::Inner,
            with: impl FnOnce(&[NonNull<()>], &table::Inner) -> T,
        ) -> T {
            match indices.split_first() {
                Some((&(index, access), rest)) => {
                    let store = unsafe { inner.stores.get_unchecked(index) };
                    debug_assert_eq!(access.identifier(), store.meta().identifier());
                    if store.meta().size == 0 {
                        pointers.push(unsafe { *store.data().data_ptr() });
                        next(rest, pointers, inner, with)
                    } else {
                        match access {
                            Access::Read(_) => {
                                let guard = store.data().read();
                                pointers.push(*guard);
                                let state = next(rest, pointers, inner, with);
                                drop(guard);
                                state
                            }
                            Access::Write(_) => {
                                let guard = store.data().write();
                                pointers.push(*guard);
                                let state = next(rest, pointers, inner, with);
                                drop(guard);
                                state
                            }
                        }
                    }
                }
                None => {
                    let state = with(pointers, inner);
                    pointers.clear();
                    state
                }
            }
        }

        let inner = self.table.inner.read();
        next(&self.indices, pointers, &inner, |pointers, inner| {
            with(&mut self.state, pointers, self.table, inner)
        })
    }
}

impl<'d, R: Row, F: Filter> Query<'d, R, F, Item> {
    #[inline]
    pub fn count(&mut self) -> usize {
        self.tables()
            .fold(0, |sum, table| sum + table.inner.read().count() as usize)
    }

    #[inline]
    pub fn count_by<V>(&mut self, by: &By<V>) -> usize {
        self.update();
        by.pairs
            .iter()
            .filter(|&(.., table)| self.indices.contains_key(table))
            .count()
    }

    #[inline]
    pub fn try_fold<S, G: FnMut(S, R::Item<'_>) -> ControlFlow<S, S>>(
        &mut self,
        state: S,
        mut fold: G,
    ) -> S {
        self.update();
        let flow = self.try_guards(state, |mut state, _, row, pointers, _, inner| {
            let context = ItemContext::new(inner.keys(), pointers);
            for i in 0..inner.count() {
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
                    by.states.clear();
                    state
                }
                Break(state) => {
                    // Fold was interrupted. Clean up.
                    by.pairs.clear();
                    for index in by.states.drain(..) {
                        unsafe { by.slots.get_unchecked_mut(index as usize) }.clear();
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
        self.guards(state, |mut state, _, row, pointers, _, inner| {
            let context = ItemContext::new(inner.keys(), pointers);
            for i in 0..inner.count() {
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
            by.states.clear();
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
            Ok((_, table)) => self.indices.contains_key(&table),
            Err(_) => false,
        }
    }

    pub fn try_find<T, G: FnOnce(Result<R::Item<'_>, Error>) -> T>(
        &mut self,
        key: Key,
        mut find: G,
    ) -> T {
        self.update();
        let (slot, mut old_table) = match self.database.keys().get(key) {
            Ok(slot) => slot,
            Err(error) => return find(Err(error)),
        };
        loop {
            let state = match self.indices.get(&old_table) {
                Some(&index) => unsafe { self.states.1.get_unchecked_mut(index as usize) },
                None => break find(Err(Error::KeyNotInQuery(key))),
            };

            find = match state.lock(&mut self.pointers, |row, pointers, table, inner| {
                debug_assert_eq!(old_table, table.index());
                match slot.table(key.generation()) {
                    Ok(new_table) if old_table == new_table => {
                        let context = ItemContext::new(inner.keys(), pointers);
                        Ok(find(Ok(R::item(row, context.with(slot.row() as _)))))
                    }
                    // The `key` has just been moved to another table.
                    Ok(new_table) => {
                        old_table = new_table;
                        Err((find, None))
                    }
                    // The `key` has just been destroyed.
                    // - Do not call `find` in here since it would hold locks for longer.
                    Err(error) => Err((find, Some(error))),
                }
            }) {
                Ok(value) => break value,
                Err((find, Some(error))) => break find(Err(error)),
                Err((find, None)) => find,
            };
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

        swap(&mut self.states.0, &mut by.states);
        let state = self.guards(state, |mut state, index, row, pointers, table, inner| {
            let context = ItemContext::new(inner.keys(), pointers);
            let slots = unsafe { by.slots.get_unchecked_mut(index as usize) };
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
        swap(&mut self.states.0, &mut by.states);
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

        swap(&mut self.states.0, &mut by.states);
        let flow = self.try_guards(state, |mut state, index, row, pointers, table, inner| {
            let context = ItemContext::new(inner.keys(), pointers);
            let slots = unsafe { by.slots.get_unchecked_mut(index as usize) };
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
        swap(&mut self.states.0, &mut by.states);
        flow
    }

    /// Sorts keys by state index such that table locks can be used for (hopefully) more than one key at a time.
    #[inline]
    fn sort<'a, V>(&'a mut self, by: &'a mut By<'d, V>) -> Errors<'d, 'a, V> {
        while by.slots.len() < self.states.1.len() {
            by.slots.push(Vec::new());
        }
        // Sort keys by state such that table locks can be used for (hopefully) more than one key at a time.
        Errors {
            indices: &self.indices,
            by: by,
        }
    }
}

impl<'d, R: Row, F: Filter> Query<'d, R, F, Chunk> {
    #[inline]
    pub fn count(&mut self) -> usize {
        self.tables().len()
    }

    #[inline]
    pub fn try_fold<S, G: FnMut(S, R::Chunk<'_>) -> ControlFlow<S, S>>(
        &mut self,
        state: S,
        mut fold: G,
    ) -> S {
        self.update();
        let flow = self.try_guards(state, |state, _, row, pointers, _, inner| {
            fold(state, R::chunk(row, ChunkContext(inner.keys(), pointers)))
        });
        match flow {
            Continue(state) => state,
            Break(state) => state,
        }
    }

    #[inline]
    pub fn fold<S, G: FnMut(S, R::Chunk<'_>) -> S>(&mut self, state: S, mut fold: G) -> S {
        self.update();
        self.guards(state, |state, _, row, pointers, _, inner| {
            fold(state, R::chunk(row, ChunkContext(inner.keys(), pointers)))
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

impl<V> Iterator for Errors<'_, '_, V> {
    type Item = (V, Error);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((key, value, slot, table)) = self.by.pairs.pop_front() {
            match self.indices.get(&table) {
                Some(&index) => {
                    let slots = unsafe { self.by.slots.get_unchecked_mut(index as usize) };
                    if slots.len() == 0 {
                        self.by.states.push(index);
                    }
                    slots.push((key, value, slot));
                }
                None => return Some((value, Error::KeyNotInQuery(key))),
            }
        }
        None
    }
}
