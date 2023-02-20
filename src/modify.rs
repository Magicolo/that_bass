use parking_lot::{RwLockReadGuard, RwLockUpgradableReadGuard, RwLockWriteGuard};

use crate::{
    core::utility::{fold_swap, get_unchecked, get_unchecked_mut, ONE},
    event::Events,
    filter::Filter,
    key::{Key, Keys},
    table::{Column, Table, Tables},
    template::{ApplyContext, InitializeContext, ShareMeta, Template},
    Database, Error,
};
use std::{
    collections::HashMap,
    marker::PhantomData,
    mem::MaybeUninit,
    num::NonZeroUsize,
    sync::{atomic::Ordering, Arc},
};

/// Adds template `A` and removes template `R` to accumulated keys that satisfy the filter `F`.
pub struct Modify<'d, A: Template, R: Template, F> {
    database: &'d Database,
    keys: Keys<'d>,
    events: Events<'d>,
    pairs: HashMap<Key, MaybeUninit<A>>, // A `HashMap` is used because the move algorithm assumes that rows will be unique.
    indices: Vec<usize>,                 // May be reordered (ex: by `fold_swap`).
    states: Vec<Result<State<A>, u32>>, // Must remain sorted by `state.source.index()` for `binary_search` to work.
    pending: Vec<(Key, A, u32)>,
    moves: Vec<(usize, usize, NonZeroUsize)>,
    copies: Vec<(usize, usize, NonZeroUsize)>,
    filter: F,
    _marker: PhantomData<fn(R)>,
}

/// Adds template `A` and removes template `R` to all keys in tables that satisfy the filter `F`.
pub struct ModifyAll<'d, A: Template, R: Template, F> {
    database: &'d Database,
    tables: Tables<'d>,
    keys: Keys<'d>,
    events: Events<'d>,
    index: usize,
    states: Vec<StateAll<A>>,
    filter: F,
    _marker: PhantomData<fn(R)>,
}

struct State<T: Template> {
    source: Arc<Table>,
    target: Arc<Table>,
    inner: Arc<Inner<T>>,
    rows: Vec<(Key, u32)>,
    templates: Vec<T>,
}

struct StateAll<T: Template> {
    source: Arc<Table>,
    target: Arc<Table>,
    inner: Arc<Inner<T>>,
}

struct Inner<T: Template> {
    state: T::State,
    apply: Box<[usize]>,
}

struct ShareTable<A: Template, R: Template> {
    source: Arc<Table>,
    target: Arc<Table>,
    inner: Arc<Inner<A>>,
    _marker: PhantomData<fn(R)>,
}

impl Database {
    pub fn modify<A: Template, R: Template>(&self) -> Result<Modify<A, R, ()>, Error> {
        // Validate metas here (no duplicates allowed), but there is no need to store them.
        ShareMeta::<(A, R)>::from(self).map(|_| Modify {
            database: self,
            keys: self.keys(),
            events: self.events(),
            pairs: HashMap::new(),
            pending: Vec::new(),
            indices: Vec::new(),
            states: Vec::new(),
            copies: Vec::new(),
            moves: Vec::new(),
            filter: (),
            _marker: PhantomData,
        })
    }

    pub fn modify_all<A: Template, R: Template>(&self) -> Result<ModifyAll<A, R, ()>, Error> {
        // Validate metas here (no duplicates allowed), but there is no need to store them.
        ShareMeta::<(A, R)>::from(self).map(|_| ModifyAll {
            database: self,
            tables: self.tables(),
            keys: self.keys(),
            events: self.events(),
            index: 0,
            states: Vec::new(),
            filter: (),
            _marker: PhantomData,
        })
    }

    pub fn add<T: Template>(&self) -> Result<Modify<T, (), ()>, Error> {
        self.modify()
    }

    pub fn add_all<T: Template>(&self) -> Result<ModifyAll<T, (), ()>, Error> {
        self.modify_all()
    }

    pub fn remove<T: Template>(&self) -> Result<Modify<(), T, ()>, Error> {
        self.modify()
    }

    pub fn remove_all<T: Template>(&self) -> Result<ModifyAll<(), T, ()>, Error> {
        self.modify_all()
    }
}

impl<'d, A: Template, R: Template, F> Modify<'d, A, R, F> {
    #[inline]
    pub fn one(&mut self, key: Key)
    where
        A: Default,
    {
        self.one_with(key, A::default())
    }

    #[inline]
    pub fn one_with(&mut self, key: Key, template: A) {
        self.pairs.insert(key, MaybeUninit::new(template));
    }

    #[inline]
    pub fn all<I: IntoIterator<Item = Key>>(&mut self, keys: I)
    where
        A: Default,
    {
        self.all_with(keys.into_iter().map(|key| (key, A::default())))
    }

    #[inline]
    pub fn all_with<I: IntoIterator<Item = (Key, A)>>(&mut self, pairs: I) {
        self.pairs.extend(
            pairs
                .into_iter()
                .map(|pair| (pair.0, MaybeUninit::new(pair.1))),
        );
    }

    pub fn filter<G: Filter + Default>(self) -> Modify<'d, A, R, (F, G)> {
        self.filter_with(G::default())
    }

    pub fn filter_with<G: Filter>(mut self, filter: G) -> Modify<'d, A, R, (F, G)> {
        for state in self.states.iter_mut() {
            let index = match state {
                Ok(state) if filter.filter(&state.source, self.database) => None,
                Ok(state) => Some(state.source.index()),
                Err(_) => None,
            };
            if let Some(index) = index {
                *state = Err(index);
            }
        }
        Modify {
            database: self.database,
            keys: self.keys,
            events: self.events,
            pairs: self.pairs,
            pending: self.pending,
            states: self.states,
            indices: self.indices,
            copies: self.copies,
            moves: self.moves,
            filter: (self.filter, filter),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (Key, &A)> {
        self.pairs
            .iter()
            .map(|pair| (*pair.0, unsafe { pair.1.assume_init_ref() }))
    }

    pub fn drain(&mut self) -> impl ExactSizeIterator<Item = (Key, A)> + '_ {
        debug_assert_eq!(self.pending.len(), 0);
        debug_assert_eq!(self.indices.len(), 0);
        self.pairs
            .drain()
            .map(|pair| (pair.0, unsafe { pair.1.assume_init() }))
    }

    pub fn clear(&mut self) {
        debug_assert_eq!(self.pending.len(), 0);
        debug_assert_eq!(self.indices.len(), 0);
        self.pairs.clear();
    }
}

impl<'d, A: Template, R: Template, F: Filter> Modify<'d, A, R, F> {
    pub fn resolve(&mut self) -> usize {
        for (key, template, result) in self
            .keys
            .get_all_with(self.pairs.iter().map(|pair| (*pair.0, pair.1)))
        {
            let template = unsafe { template.assume_init_read() };
            if let Ok((_, table)) = result {
                Self::sort(
                    key,
                    template,
                    table,
                    &mut self.indices,
                    &mut self.states,
                    &self.filter,
                    self.database,
                );
            }
        }

        let mut sum = 0;
        loop {
            sum += self.resolve_sorted();
            self.indices.clear();
            if self.pending.len() == 0 {
                break;
            }

            for (key, template, table) in self.pending.drain(..) {
                Self::sort(
                    key,
                    template,
                    table,
                    &mut self.indices,
                    &mut self.states,
                    &self.filter,
                    self.database,
                );
            }
        }

        self.pairs.clear();
        debug_assert_eq!(self.pairs.len(), 0);
        debug_assert_eq!(self.moves.len(), 0);
        debug_assert_eq!(self.copies.len(), 0);
        debug_assert_eq!(self.indices.len(), 0);
        sum
    }

    fn resolve_sorted(&mut self) -> usize {
        fold_swap(
            &mut self.indices,
            0,
            (
                &mut self.states,
                &mut self.pending,
                &mut self.moves,
                &mut self.copies,
            ),
            |sum, (states, pending, moves, copies), index| {
                let result = unsafe { get_unchecked_mut(states, *index) };
                let state = unsafe { result.as_mut().unwrap_unchecked() };
                debug_assert!(state.rows.len() > 0);
                if state.source.index() == state.target.index() {
                    let keys = state.source.keys.try_read().ok_or(sum)?;
                    let count = Self::resolve_set(
                        &self.keys,
                        &state.source,
                        keys,
                        &state.inner.state,
                        &mut state.rows,
                        &mut state.templates,
                        pending,
                        &state.inner.apply,
                    );
                    return Ok(sum + count);
                }
                let source = state.source.keys.try_upgradable_read().ok_or(sum)?;
                let target = state.target.keys.try_upgradable_read().ok_or(sum)?;
                let (low, high) = Self::retain(
                    &self.keys,
                    &state.source,
                    &mut state.rows,
                    &mut state.templates,
                    pending,
                );
                let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                    // Happens if all keys from this table have been moved or destroyed between here and the sorting.
                    return Ok(sum);
                };
                move_to(
                    &self.keys,
                    &self.events,
                    &self.pairs,
                    &mut state.templates,
                    moves,
                    copies,
                    (&state.source, source),
                    (&state.target, target),
                    (low, high, count),
                    &mut state.rows,
                    &state.inner,
                );
                Ok(sum + count.get())
            },
            |sum, (states, pending, moves, copies), index| {
                let result = unsafe { get_unchecked_mut(states, *index) };
                let state = unsafe { result.as_mut().unwrap_unchecked() };
                debug_assert!(state.rows.len() > 0);
                if state.source.index() == state.target.index() {
                    let keys = state.source.keys.read();
                    let count = Self::resolve_set(
                        &self.keys,
                        &state.source,
                        keys,
                        &state.inner.state,
                        &mut state.rows,
                        &mut state.templates,
                        pending,
                        &state.inner.apply,
                    );
                    return sum + count;
                }

                let (source, target, low, high, count) =
                    // If locks are always taken in order (lower index first), there can not be a deadlock between move operations.
                    if state.source.index() < state.target.index() {
                        let source = state.source.keys.upgradable_read();
                        let (low, high) = Self::retain(&self.keys,&state.source, &mut state.rows, &mut state.templates, pending);
                        let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                            // Happens if all keys from this table have been moved or destroyed between here and the sorting.
                            return sum;
                        };
                        let target = state.target.keys.upgradable_read();
                        (source, target, low, high, count)
                    } else  {
                        let target = state.target.keys.upgradable_read();
                        let source = state.source.keys.upgradable_read();
                        let (low, high) = Self::retain(&self.keys,&state.source, &mut state.rows, &mut state.templates, pending);
                        let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                            // Happens if all keys from this table have been moved or destroyed between here and the sorting.
                            return sum;
                        };
                        (source, target, low, high, count)
                    };
                move_to(
                    &self.keys,
                    &self.events,
                    &self.pairs,
                    &mut state.templates,
                    moves,
                    copies,
                    (&state.source, source),
                    (&state.target, target),
                    (low, high, count),
                    &mut state.rows,
                    &state.inner,
                );
                sum + count.get()
            },
        )
    }

    fn resolve_set(
        keys: &Keys,
        table: &Table,
        table_keys: RwLockReadGuard<Vec<Key>>,
        state: &A::State,
        rows: &mut Vec<(Key, u32)>,
        templates: &mut Vec<A>,
        pending: &mut Vec<(Key, A, u32)>,
        apply: &[usize],
    ) -> usize {
        Self::retain(keys, table, rows, templates, pending);
        let Some(count) = NonZeroUsize::new(rows.len()) else {
            return 0;
        };
        // The keys do not need to be moved, simply write the row data.
        lock(apply, table, |table| {
            let context = ApplyContext::new(table, &table_keys);
            for ((.., row), template) in rows.drain(..).zip(templates.drain(..)) {
                debug_assert!(row < u32::MAX);
                unsafe { template.apply(state, context.with(row as _)) };
            }
        });
        count.get()
    }

    fn sort(
        key: Key,
        template: A,
        table: u32,
        indices: &mut Vec<usize>,
        states: &mut Vec<Result<State<A>, u32>>,
        filter: &F,
        database: &'d Database,
    ) {
        let index = match states.binary_search_by_key(&table, |result| match result {
            Ok(state) => state.source.index(),
            Err(index) => *index,
        }) {
            Ok(index) => index,
            Err(index) => {
                let result = match ShareTable::<A, R>::from(table, database) {
                    Ok((source, target, inner)) if filter.filter(&source, database) => Ok(State {
                        source,
                        target,
                        inner,
                        rows: Vec::new(),
                        templates: Vec::new(),
                    }),
                    _ => Err(table),
                };
                for i in indices.iter_mut().filter(|i| **i >= index) {
                    *i += 1;
                }
                states.insert(index, result);
                index
            }
        };
        if let Ok(state) = unsafe { get_unchecked_mut(states, index) } {
            if state.rows.len() == 0 {
                indices.push(index);
            }
            state.rows.push((key, u32::MAX));
            state.templates.push(template);
        }
    }

    /// Call this while holding a lock on `table`.
    fn retain(
        keys: &Keys,
        table: &Table,
        rows: &mut Vec<(Key, u32)>,
        templates: &mut Vec<A>,
        pending: &mut Vec<(Key, A, u32)>,
    ) -> (u32, u32) {
        let mut low = u32::MAX;
        let mut high = 0;
        for i in (0..rows.len()).rev() {
            let (key, row) = unsafe { get_unchecked_mut(rows, i) };
            let slot = unsafe { keys.get_unchecked(*key) };
            match slot.table(*key) {
                Ok(table_index) if table_index == table.index() => {
                    *row = slot.row();
                    low = low.min(*row);
                    high = high.max(*row);
                }
                Ok(table_index) => {
                    let (key, _) = rows.swap_remove(i);
                    let template = templates.swap_remove(i);
                    pending.push((key, template, table_index));
                }
                Err(_) => {
                    rows.swap_remove(i);
                    templates.swap_remove(i);
                }
            }
        }
        debug_assert_eq!(low <= high, rows.len() > 0);
        (low, high)
    }
}

impl<'d, A: Template, R: Template, F> ModifyAll<'d, A, R, F> {
    pub fn filter<G: Filter + Default>(self) -> ModifyAll<'d, A, R, (F, G)> {
        self.filter_with(G::default())
    }

    pub fn filter_with<G: Filter>(mut self, filter: G) -> ModifyAll<'d, A, R, (F, G)> {
        self.states
            .retain(|state| filter.filter(&state.source, self.database));
        ModifyAll {
            database: self.database,
            tables: self.tables,
            keys: self.keys,
            events: self.events,
            index: self.index,
            states: self.states,
            filter: (self.filter, filter),
            _marker: PhantomData,
        }
    }
}

impl<'d, A: Template, R: Template, F: Filter> ModifyAll<'d, A, R, F> {
    #[inline]
    pub fn resolve(&mut self) -> usize
    where
        A: Default,
    {
        self.resolve_with(true, A::default)
    }

    pub fn resolve_with<G: FnMut() -> A>(&mut self, set: bool, with: G) -> usize {
        while let Ok(table) = self.tables.get(self.index) {
            self.index += 1;
            match ShareTable::<A, R>::from(table.index(), self.database) {
                Ok((source, target, inner)) if self.filter.filter(table, self.database) => {
                    self.states.push(StateAll {
                        source,
                        target,
                        inner,
                    })
                }
                _ => (),
            }
        }

        fold_swap(
            &mut self.states,
            0,
            with,
            |sum, with, state| {
                if state.source.index() != state.target.index() {
                    let source = state.source.keys.try_write().ok_or(sum)?;
                    let target = state.target.keys.try_upgradable_read().ok_or(sum)?;
                    Ok(sum
                        + Self::resolve_tables(
                            source,
                            target,
                            state,
                            &self.keys,
                            &self.events,
                            with,
                        ))
                } else if set {
                    let keys = state.source.keys.try_read().ok_or(sum)?;
                    Ok(sum + Self::resolve_table(keys, state, with))
                } else {
                    Ok(sum)
                }
            },
            |sum, with, state| {
                if state.source.index() < state.target.index() {
                    let source = state.source.keys.write();
                    let target = state.target.keys.upgradable_read();
                    sum + Self::resolve_tables(
                        source,
                        target,
                        state,
                        &self.keys,
                        &self.events,
                        with,
                    )
                } else if state.source.index() > state.target.index() {
                    let target = state.target.keys.upgradable_read();
                    let source = state.source.keys.write();
                    sum + Self::resolve_tables(
                        source,
                        target,
                        state,
                        &self.keys,
                        &self.events,
                        with,
                    )
                } else if set {
                    let keys = state.source.keys.read();
                    sum + Self::resolve_table(keys, state, with)
                } else {
                    sum
                }
            },
        )
    }

    fn resolve_tables(
        mut source: RwLockWriteGuard<Vec<Key>>,
        target: RwLockUpgradableReadGuard<Vec<Key>>,
        state: &StateAll<A>,
        keys: &Keys,
        events: &Events,
        mut with: impl FnMut() -> A,
    ) -> usize {
        let count = state.source.count.swap(0, Ordering::AcqRel);
        let Some(count) = NonZeroUsize::new(count) else {
            return 0;
        };
        let (start, target) = state.target.reserve(target, count);
        let target_keys = unsafe { &mut *RwLockUpgradableReadGuard::rwlock(&target).data_ptr() };
        target_keys[start..start + count.get()].copy_from_slice(&source[..count.get()]);
        resolve_copy_move(
            (&mut source, &state.source),
            &state.target,
            &[(0, start, count)],
            &[],
            keys,
        );

        // SAFETY: Since this row is not yet observable by any thread but this one, bypass locks.
        let context = ApplyContext::new(&state.target, &target);
        for i in 0..count.get() {
            unsafe { with().apply(&state.inner.state, context.with(start + i)) };
        }

        state
            .target
            .count
            .fetch_add(count.get() as _, Ordering::Release);
        // Slots must be updated after the table `fetch_add` to prevent a `query::find` to be able to observe a row which
        // has an index greater than the `table.count()`. As long as the slots remain in the source table, all accesses
        // to these keys will block at the table access and will correct their table index after they acquire the source
        // table lock.
        keys.initialize_all(&target, state.target.index(), start..start + count.get());
        drop(source);
        // Although `source` has been dropped, coherence with be maintained since the `target` lock prevent the keys
        // moving again before `on_remove` is done.
        events.emit_modify(
            &target[start..start + count.get()],
            (&state.source, &state.target),
        );
        count.get()
    }

    fn resolve_table(
        keys: RwLockReadGuard<Vec<Key>>,
        state: &StateAll<A>,
        with: &mut impl FnMut() -> A,
    ) -> usize {
        debug_assert_eq!(state.source.index(), state.target.index());
        debug_assert!(state.inner.apply.len() > 0); // `apply.len() == 0` should've been filtered by `ShareTable`
        let Some(count) = NonZeroUsize::new(state.source.count()) else {
            return 0;
        };
        lock(&state.inner.apply, &state.source, |table| {
            let context = ApplyContext::new(table, &keys);
            for i in 0..count.get() {
                unsafe { with().apply(&state.inner.state, context.with(i)) };
            }
        });
        count.get()
    }
}

impl<A: Template, R: Template> ShareTable<A, R> {
    pub fn from(
        table: u32,
        database: &Database,
    ) -> Result<(Arc<Table>, Arc<Table>, Arc<Inner<A>>), Error> {
        let adds = ShareMeta::<A>::from(database)?;
        let removes = ShareMeta::<R>::from(database)?;
        let share = database.resources().try_global_with(table, || {
            let mut tables = database.tables();
            let source = tables.get_shared(table as usize)?;
            let target = {
                let mut metas = adds.to_vec();
                for meta in source.metas() {
                    match (
                        metas.binary_search_by_key(&meta.identifier(), |meta| meta.identifier()),
                        removes.binary_search_by_key(&meta.identifier(), |meta| meta.identifier()),
                    ) {
                        (Ok(_), Ok(_)) => return Err(Error::DuplicateMeta),
                        (Err(index), Err(_)) => metas.insert(index, meta),
                        (Ok(_), Err(_)) | (Err(_), Ok(_)) => {}
                    }
                }
                tables.find_or_add(&metas)
            };
            let state = A::initialize(InitializeContext::new(&target))?;

            let mut apply = Vec::new();
            for meta in adds.iter() {
                let (index, column) = target.column_with(meta.identifier())?;
                if column.meta().layout().size() > 0 {
                    apply.push(index);
                }
            }

            if apply.len() == 0 && source.index() == target.index() {
                return Err(Error::TablesMustDiffer(source.index() as _));
            }

            Ok(Self {
                source,
                target,
                inner: Arc::new(Inner {
                    state,
                    apply: apply.into_boxed_slice(),
                }),
                _marker: PhantomData,
            })
        })?;
        Ok((
            share.source.clone(),
            share.target.clone(),
            share.inner.clone(),
        ))
    }
}

fn move_to<'d, 'a, V, A: Template>(
    keys: &Keys,
    events: &Events,
    set: &HashMap<Key, V>,
    templates: &mut Vec<A>,
    moves: &mut Vec<(usize, usize, NonZeroUsize)>,
    copies: &mut Vec<(usize, usize, NonZeroUsize)>,
    (source_table, source_keys): (&Table, RwLockUpgradableReadGuard<'a, Vec<Key>>),
    (target_table, target_keys): (&Table, RwLockUpgradableReadGuard<'a, Vec<Key>>),
    (low, high, count): (u32, u32, NonZeroUsize),
    rows: &mut Vec<(Key, u32)>,
    inner: &Inner<A>,
) {
    let (start, target_keys) = target_table.reserve(target_keys, count);
    // Move data from source to target.
    let range = low..high + 1;
    let head = source_table.count.load(Ordering::Acquire) - count.get();
    let (low, high) = (range.start as usize, range.end as usize);

    if range.len() == count.get() {
        // Fast path. The move range is contiguous. Copy everything from source to target at once.
        copies.push((low, start, count));

        let over = high.saturating_sub(head);
        if let Some(end) = NonZeroUsize::new(count.get() - over) {
            moves.push((head + over, low, end));
        }
    } else {
        // Range is not contiguous; use the slow path.
        let mut cursor = head;
        for (i, &(.., row)) in rows.iter().enumerate() {
            let row = row as usize;
            copies.push((row, start + i, ONE));

            if row < head {
                // Find the next valid row to move.
                while set.contains_key(unsafe { get_unchecked(&source_keys, cursor) }) {
                    cursor += 1;
                }
                debug_assert!(cursor < head + count.get());
                moves.push((cursor, row, ONE));
                cursor += 1;
            }
        }
    }

    {
        // Target keys can be copied over without requiring the `source_keys` write lock since the range `start..start + count` is
        // reserved to this operation.
        let target_keys =
            unsafe { &mut *RwLockUpgradableReadGuard::rwlock(&target_keys).data_ptr() };
        for &(source, target, count) in copies.iter() {
            target_keys[target..target + count.get()]
                .copy_from_slice(&source_keys[source..source + count.get()]);
        }
    }

    let mut source_keys = RwLockUpgradableReadGuard::upgrade(source_keys);
    resolve_copy_move(
        (&mut source_keys, source_table),
        target_table,
        copies,
        moves,
        keys,
    );

    // Initialize missing data `T` in target.
    // SAFETY: Since this row is not yet observable by any thread but this one, bypass locks.
    let context = ApplyContext::new(target_table, &target_keys);
    for (i, template) in templates.drain(..).enumerate() {
        unsafe { template.apply(&inner.state, context.with(start + i)) };
    }
    source_table.count.fetch_sub(count.get(), Ordering::Release);
    target_table.count.fetch_add(count.get(), Ordering::Release);
    // Slots must be updated after the table `fetch_add` to prevent a `query::find` to be able to observe a row which
    // has an index greater than the `table.count()`. As long as the slots remain in the source table, all accesses
    // to these keys will block at the table access and will correct their table index after they acquire the source
    // table lock.
    keys.initialize_all(
        &target_keys,
        target_table.index(),
        start..start + count.get(),
    );
    drop(source_keys);
    // Although `source_keys` has been dropped, coherence will be maintained since the `target` lock prevents the keys from
    // moving again before `emit` is done.
    events.emit_modify(
        &target_keys[start..start + count.get()],
        (source_table, target_table),
    );
    drop(target_keys);
    rows.clear();
    copies.clear();
    moves.clear();
}

/// SAFETY: Since a write lock is held over the `source_keys`, it is guaranteed that there is no reader/writer in
/// the columns, so there is no need to take column locks.
fn resolve_copy_move(
    (source_keys, source_table): (&mut [Key], &Table),
    target_table: &Table,
    copies: &[(usize, usize, NonZeroUsize)],
    moves: &[(usize, usize, NonZeroUsize)],
    keys: &Keys,
) {
    for &(source, target, count) in moves {
        source_keys.copy_within(source..source + count.get(), target);
        keys.update_all(&source_keys, target..target + count.get());
    }

    let mut index = 0;
    for source_column in source_table.columns() {
        let copy = source_column.meta().layout().size() > 0;
        let mut drop = source_column.meta().drop.0();
        while let Some(target_column) = target_table.columns().get(index) {
            if source_column.meta().identifier() == target_column.meta().identifier() {
                index += 1;
                drop = false;
                if copy {
                    for &(source, target, count) in copies {
                        unsafe {
                            Column::copy_to(
                                (&source_column, source),
                                (&target_column, target),
                                count,
                            )
                        };
                    }
                }
                break;
            } else if source_column.meta().identifier() < target_column.meta().identifier() {
                break;
            } else {
                index += 1;
            }
        }
        if drop {
            for &(index, _, count) in copies {
                unsafe { source_column.drop(index, count) }
            }
        }
        if copy {
            for &(source, target, count) in moves {
                unsafe { source_column.copy(source, target, count) };
            }
        }
    }
}

#[inline]
fn lock<T>(indices: &[usize], table: &Table, with: impl FnOnce(&Table) -> T) -> T {
    match indices.split_first() {
        Some((&index, rest)) => {
            let column = unsafe { get_unchecked(table.columns(), index) };
            debug_assert!(column.meta().layout().size() > 0);
            let _guard = column.data().write();
            lock(rest, table, with)
        }
        None => with(table),
    }
}
