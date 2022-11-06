use crate::{
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
    states: Vec<State<'d, R>>,
    accesses: HashSet<Access>,
    done: VecDeque<u32>,
    pending: VecDeque<u32>,
    pointers: Vec<NonNull<()>>,
    _marker: PhantomData<fn(F, I)>,
}

pub struct Item;
pub struct Chunk;

pub struct By<'d, V> {
    database: &'d Database,
    pairs: VecDeque<(Key, V, &'d Slot, u32)>,
    slots: Vec<Vec<(Key, V, &'d Slot)>>,
}

struct State<'d, R: Row> {
    state: R::State,
    table: &'d Table,
    indices: Vec<(usize, Access)>,
}

struct Errors<'d, 'a, V> {
    indices: &'a HashMap<u32, u32>,
    pending: &'a mut VecDeque<u32>,
    pairs: &'a mut VecDeque<(Key, V, &'d Slot, u32)>,
    slots: &'a mut [Vec<(Key, V, &'d Slot)>],
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
            states: Vec::new(),
            done: VecDeque::new(),
            pending: VecDeque::new(),
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
        }
    }
}

impl<'d, R: Row, F: Filter, I> Query<'d, R, F, I> {
    pub fn chunk(self) -> Query<'d, R, F, Chunk> {
        Query {
            database: self.database,
            indices: self.indices,
            done: self.done,
            pending: self.pending,
            states: self.states,
            accesses: self.accesses,
            pointers: self.pointers,
            index: self.index,
            _marker: PhantomData,
        }
    }

    pub fn read(self) -> Query<'d, R::Read, F, I> {
        debug_assert!(self.done.is_empty());
        debug_assert!(self.pointers.is_empty());

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
            states,
            done: self.done,
            pending: self.pending,
            accesses,
            pointers: self.pointers,
            index: self.index,
            _marker: PhantomData,
        }
    }

    pub fn filter<G: Filter>(mut self) -> Query<'d, R, (F, G), I> {
        debug_assert!(self.done.is_empty());
        debug_assert!(self.pointers.is_empty());

        for state in self.states.iter() {
            if G::filter(state.table, self.database) {
                continue;
            } else {
                self.indices.remove(&state.table.index());
            }
        }
        self.pending.clear();
        self.pending.extend(self.indices.values());

        Query {
            database: self.database,
            indices: self.indices,
            states: self.states,
            done: self.done,
            pending: self.pending,
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

    pub(crate) fn try_guards<S>(
        &mut self,
        state: S,
        mut fold: impl FnMut(
            S,
            u32,
            &mut R::State,
            &[NonNull<()>],
            &Table,
            &table::Inner,
        ) -> ControlFlow<S, S>,
    ) -> ControlFlow<S, S> {
        let mut fold = |mut state: S| -> ControlFlow<S, S> {
            for _ in 0..self.pending.len() {
                let index = unsafe { self.pending.pop_front().unwrap_unchecked() };
                let result = self.try_lock(state, index, |state, row, pointers, table, inner| {
                    fold(state, index, row, pointers, table, inner)
                });
                state = match result {
                    Ok(result) => {
                        self.done.push_back(index);
                        result?
                    }
                    Err(state) => {
                        self.pending.push_back(index);
                        state
                    }
                };
            }

            while let Some(index) = self.pending.pop_front() {
                self.done.push_back(index);
                state = self.lock(index, |row, pointers, table, inner| {
                    fold(state, index, row, pointers, table, inner)
                })?;
            }
            Continue(state)
        };

        match fold(state) {
            Continue(state) => {
                swap(&mut self.done, &mut self.pending);
                Continue(state)
            }
            Break(state) => {
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
                Break(state)
            }
        }
    }

    pub(crate) fn guards<S>(
        &mut self,
        mut state: S,
        mut fold: impl FnMut(S, u32, &mut R::State, &[NonNull<()>], &Table, &table::Inner) -> S,
    ) -> S {
        for _ in 0..self.pending.len() {
            let index = unsafe { self.pending.pop_front().unwrap_unchecked() };
            state = match self.try_lock(state, index, |state, row, pointers, table, inner| {
                fold(state, index, row, pointers, table, inner)
            }) {
                Ok(state) => {
                    self.done.push_back(index);
                    state
                }
                Err(state) => {
                    self.pending.push_back(index);
                    state
                }
            };
        }

        while let Some(index) = self.pending.pop_front() {
            self.done.push_back(index);
            state = self.lock(index, |row, pointers, table, inner| {
                fold(state, index, row, pointers, table, inner)
            });
        }

        swap(&mut self.done, &mut self.pending);
        state
    }

    #[inline]
    fn try_lock<S, T>(
        &mut self,
        state: S,
        index: u32,
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

        let State {
            state: row,
            table,
            indices,
        } = unsafe { self.states.get_unchecked_mut(index as usize) };
        let Some(inner) = table.inner.try_read() else {
            return Err(state);
        };
        next(
            state,
            indices,
            &mut self.pointers,
            &inner,
            |state, pointers, inner| with(state, row, pointers, table, inner),
        )
    }

    #[inline]
    fn lock<T>(
        &mut self,
        index: u32,
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

        let State {
            state: row,
            table,
            indices,
        } = unsafe { self.states.get_unchecked_mut(index as usize) };
        let inner = table.inner.read();
        next(indices, &mut self.pointers, &inner, |pointers, inner| {
            with(row, pointers, table, inner)
        })
    }

    fn try_add(&mut self, table: &'d Table) -> Result<(), Error> {
        if F::filter(table, self.database) {
            let mut indices = Vec::with_capacity(self.accesses.len());
            for &access in self.accesses.iter() {
                indices.push((table.store_with(access.identifier())?, access));
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
            let index = self.states.len() as _;
            self.pending.push_back(index);
            self.indices.insert(table.index(), index);
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
        self.update();
        self.indices
            .values()
            .map(|&index| {
                unsafe { self.states.get_unchecked(index as usize) }
                    .table
                    .inner
                    .read()
                    .count() as usize
            })
            .sum()
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
        let pending = self.pending.len();
        while by.pairs.len() > 0 {
            state = match self.try_fold_by_sorted(by, state, &mut fold) {
                Continue(state) => state,
                Break(state) => {
                    // Fold was interrupted. Clean up.
                    by.pairs.clear();
                    for index in self.pending.drain(pending..) {
                        unsafe { by.slots.get_unchecked_mut(index as usize) }.clear();
                    }
                    return state;
                }
            }
        }
        self.pending.drain(pending..);
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
        let pending = self.pending.len();
        while by.pairs.len() > 0 {
            state = self.fold_by_sorted(by, state, &mut fold);
        }
        self.pending.drain(pending..);
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
            let state_index = match self.indices.get(&old_table) {
                Some(&state_index) => state_index,
                None => break find(Err(Error::KeyNotInQuery(key))),
            };

            find = match self.lock(state_index, |row, pointers, table, inner| {
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

        self.guards(state, |mut state, index, row, pointers, table, inner| {
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
        })
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

        self.try_guards(state, |mut state, index, row, pointers, table, inner| {
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
        })
    }

    /// Sorts keys by state index such that table locks can be used for (hopefully) more than one key at a time.
    #[inline]
    fn sort<'a, V>(&'a mut self, by: &'a mut By<'d, V>) -> Errors<'d, 'a, V> {
        self.update();
        by.ensure(self.states.len());
        swap(&mut self.done, &mut self.pending);

        // Sort keys by state such that table locks can be used for (hopefully) more than one key at a time.
        Errors {
            indices: &self.indices,
            pending: &mut self.pending,
            pairs: &mut by.pairs,
            slots: &mut by.slots,
        }
    }
}

impl<'d, R: Row, F: Filter> Query<'d, R, F, Chunk> {
    #[inline]
    pub fn count(&mut self) -> usize {
        self.update();
        self.indices.len()
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

    #[inline]
    fn ensure(&mut self, capacity: usize) {
        while self.slots.len() < capacity {
            self.slots.push(Vec::new());
        }
    }
}

impl<V> Iterator for Errors<'_, '_, V> {
    type Item = (V, Error);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((key, value, slot, table)) = self.pairs.pop_front() {
            match self.indices.get(&table) {
                Some(&index) => {
                    let slots = unsafe { self.slots.get_unchecked_mut(index as usize) };
                    if slots.len() == 0 {
                        self.pending.push_back(index);
                    }
                    slots.push((key, value, slot));
                }
                None => return Some((value, Error::KeyNotInQuery(key))),
            }
        }
        None
    }
}
