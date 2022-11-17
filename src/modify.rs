use parking_lot::{RwLockReadGuard, RwLockUpgradableReadGuard, RwLockWriteGuard};

use crate::{
    core::utility::{fold_swap, get_unchecked, get_unchecked_mut, sorted_difference, ONE},
    filter::Filter,
    key::{Key, Slot},
    listen::Listen,
    resources::Resources,
    table::{Column, Table, Tables},
    template::{ApplyContext, InitializeContext, ShareMeta, Template},
    Database, Error,
};
use std::{
    any::TypeId,
    collections::HashMap,
    marker::PhantomData,
    mem::MaybeUninit,
    num::NonZeroUsize,
    sync::{atomic::Ordering, Arc},
};

/// Adds template `A` and removes template `R` to accumulated keys that satisfy the filter `F`.
pub struct Modify<'d, A: Template, R: Template, F, L> {
    database: &'d Database<L>,
    pairs: HashMap<Key, MaybeUninit<A>>, // A `HashMap` is used because the move algorithm assumes that rows will be unique.
    indices: Vec<usize>,                 // May be reordered (ex: by `fold_swap`).
    states: Vec<Result<State<'d, A>, u32>>, // Must remain sorted by `state.source.index()` for `binary_search` to work.
    pending: Vec<(Key, &'d Slot, A, u32)>,
    moves: Vec<(usize, usize, NonZeroUsize)>,
    copies: Vec<(usize, usize, NonZeroUsize)>,
    filter: F,
    _marker: PhantomData<fn(R)>,
}

/// Adds template `A` and removes template `R` to all keys in tables that satisfy the filter `F`.
pub struct ModifyAll<'d, A: Template, R: Template, F, L> {
    database: &'d Database<L>,
    index: usize,
    states: Vec<StateAll<A>>,
    filter: F,
    _marker: PhantomData<fn(R)>,
}

type Rows<'d> = Vec<(Key, &'d Slot, u32)>;

struct State<'d, T: Template> {
    source: Arc<Table>,
    target: Arc<Table>,
    inner: Arc<Inner<T>>,
    rows: Rows<'d>,
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
    added: Box<[TypeId]>,
    removed: Box<[TypeId]>,
}

struct ShareTable<A: Template, R: Template> {
    source: Arc<Table>,
    target: Arc<Table>,
    inner: Arc<Inner<A>>,
    _marker: PhantomData<fn(R)>,
}

impl<L> Database<L> {
    pub fn modify<A: Template, R: Template>(&self) -> Result<Modify<A, R, (), L>, Error> {
        // Validate metas here (no duplicates allowed), but there is no need to store them.
        ShareMeta::<(A, R)>::from(self.resources()).map(|_| Modify {
            database: self,
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

    pub fn modify_all<A: Template, R: Template>(&self) -> Result<ModifyAll<A, R, (), L>, Error> {
        // Validate metas here (no duplicates allowed), but there is no need to store them.
        ShareMeta::<(A, R)>::from(self.resources()).map(|_| ModifyAll {
            database: self,
            index: 0,
            states: Vec::new(),
            filter: (),
            _marker: PhantomData,
        })
    }

    pub fn add<T: Template>(&self) -> Result<Modify<T, (), (), L>, Error> {
        self.modify()
    }

    pub fn add_all<T: Template>(&self) -> Result<ModifyAll<T, (), (), L>, Error> {
        self.modify_all()
    }

    pub fn remove<T: Template>(&self) -> Result<Modify<(), T, (), L>, Error> {
        self.modify()
    }

    pub fn remove_all<T: Template>(&self) -> Result<ModifyAll<(), T, (), L>, Error> {
        self.modify_all()
    }
}

impl<'d, A: Template, R: Template, F, L> Modify<'d, A, R, F, L> {
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

    pub fn filter<G: Filter + Default>(self) -> Modify<'d, A, R, (F, G), L> {
        self.filter_with(G::default())
    }

    pub fn filter_with<G: Filter>(mut self, filter: G) -> Modify<'d, A, R, (F, G), L> {
        for state in self.states.iter_mut() {
            let index = match state {
                Ok(state) if filter.filter(&state.source, self.database.into()) => None,
                Ok(state) => Some(state.source.index()),
                Err(_) => None,
            };
            if let Some(index) = index {
                *state = Err(index);
            }
        }
        Modify {
            database: self.database,
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

impl<'d, A: Template, R: Template, F: Filter, L: Listen> Modify<'d, A, R, F, L> {
    pub fn resolve(&mut self) -> usize {
        for (&key, template) in self.pairs.iter() {
            if let Ok((slot, table)) = self.database.keys().get(key) {
                Self::sort(
                    key,
                    slot,
                    unsafe { template.assume_init_read() },
                    table,
                    &mut self.indices,
                    &mut self.states,
                    &self.filter,
                    &self.database.inner,
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

            for (key, slot, template, table) in self.pending.drain(..) {
                Self::sort(
                    key,
                    slot,
                    template,
                    table,
                    &mut self.indices,
                    &mut self.states,
                    &self.filter,
                    &self.database.inner,
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
                    &self.database,
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
                        let (low, high) = Self::retain(&state.source, &mut state.rows, &mut state.templates, pending);
                        let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                            // Happens if all keys from this table have been moved or destroyed between here and the sorting.
                            return sum;
                        };
                        let target = state.target.keys.upgradable_read();
                        (source, target, low, high, count)
                    } else  {
                        let target = state.target.keys.upgradable_read();
                        let source = state.source.keys.upgradable_read();
                        let (low, high) = Self::retain(&state.source, &mut state.rows, &mut state.templates, pending);
                        let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                            // Happens if all keys from this table have been moved or destroyed between here and the sorting.
                            return sum;
                        };
                        (source, target, low, high, count)
                    };
                move_to(
                    &self.database,
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
        table: &Table,
        keys: RwLockReadGuard<Vec<Key>>,
        state: &A::State,
        rows: &mut Rows<'d>,
        templates: &mut Vec<A>,
        pending: &mut Vec<(Key, &'d Slot, A, u32)>,
        apply: &[usize],
    ) -> usize
    where
        L: 'd, // Why does the compiler requires this?
    {
        Self::retain(table, rows, templates, pending);
        let Some(count) = NonZeroUsize::new(rows.len()) else {
            return 0;
        };
        // The keys do not need to be moved, simply write the row data.
        lock(apply, table, |table| {
            let context = ApplyContext::new(table, &keys);
            for ((.., row), template) in rows.drain(..).zip(templates.drain(..)) {
                debug_assert!(row < u32::MAX);
                unsafe { template.apply(state, context.with(row as _)) };
            }
        });
        count.get()
    }

    fn sort(
        key: Key,
        slot: &'d Slot,
        template: A,
        table: u32,
        indices: &mut Vec<usize>,
        states: &mut Vec<Result<State<'d, A>, u32>>,
        filter: &F,
        database: &'d crate::Inner,
    ) {
        let index = match states.binary_search_by_key(&table, |result| match result {
            Ok(state) => state.source.index(),
            Err(index) => *index,
        }) {
            Ok(index) => index,
            Err(index) => {
                let result =
                    match ShareTable::<A, R>::from(table, &database.tables, &database.resources) {
                        Ok((source, target, inner)) if filter.filter(&source, database.into()) => {
                            Ok(State {
                                source,
                                target,
                                inner,
                                rows: Vec::new(),
                                templates: Vec::new(),
                            })
                        }
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
            state.rows.push((key, slot, u32::MAX));
            state.templates.push(template);
        }
    }

    /// Call this while holding a lock on `table`.
    fn retain(
        table: &Table,
        rows: &mut Rows<'d>,
        templates: &mut Vec<A>,
        pending: &mut Vec<(Key, &'d Slot, A, u32)>,
    ) -> (u32, u32) {
        let mut low = u32::MAX;
        let mut high = 0;
        for i in (0..rows.len()).rev() {
            let (key, slot, row) = unsafe { get_unchecked_mut(rows, i) };
            if let Ok(table_index) = slot.table(*key) {
                if table_index == table.index() {
                    *row = slot.row();
                    low = low.min(*row);
                    high = high.max(*row);
                } else {
                    let (key, slot, _) = rows.swap_remove(i);
                    let template = templates.swap_remove(i);
                    pending.push((key, slot, template, table_index));
                }
            } else {
                rows.swap_remove(i);
                templates.swap_remove(i);
            }
        }
        debug_assert_eq!(low <= high, rows.len() > 0);
        (low, high)
    }
}

impl<'d, A: Template, R: Template, F, L> ModifyAll<'d, A, R, F, L> {
    pub fn filter<G: Filter + Default>(self) -> ModifyAll<'d, A, R, (F, G), L> {
        self.filter_with(G::default())
    }

    pub fn filter_with<G: Filter>(mut self, filter: G) -> ModifyAll<'d, A, R, (F, G), L> {
        self.states
            .retain(|state| filter.filter(&state.source, self.database.into()));
        ModifyAll {
            database: self.database,
            index: self.index,
            states: self.states,
            filter: (self.filter, filter),
            _marker: PhantomData,
        }
    }
}

impl<'d, A: Template, R: Template, F: Filter, L: Listen> ModifyAll<'d, A, R, F, L> {
    #[inline]
    pub fn resolve(&mut self) -> usize
    where
        A: Default,
    {
        self.resolve_with(true, A::default)
    }

    pub fn resolve_with<G: FnMut() -> A>(&mut self, set: bool, with: G) -> usize {
        while let Ok(table) = self.database.tables().get(self.index) {
            self.index += 1;
            match ShareTable::<A, R>::from(
                table.index(),
                self.database.tables(),
                self.database.resources(),
            ) {
                Ok((source, target, inner)) if self.filter.filter(table, self.database.into()) => {
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
                    Ok(sum + Self::resolve_tables(source, target, state, self.database, with))
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
                    sum + Self::resolve_tables(source, target, state, self.database, with)
                } else if state.source.index() > state.target.index() {
                    let target = state.target.keys.upgradable_read();
                    let source = state.source.keys.write();
                    sum + Self::resolve_tables(source, target, state, self.database, with)
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
        database: &Database<impl Listen>,
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
            &database.inner,
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
        database
            .keys()
            .initialize(&target, state.target.index(), start..start + count.get());
        drop(source);
        // Although `source` has been dropped, coherence with be maintained since the `target` lock prevent the keys
        // moving again before `on_remove` is done.
        database.listen.on_modify(
            &target[start..start + count.get()],
            &state.inner.added,
            &state.inner.removed,
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
        tables: &Tables,
        resources: &Resources,
    ) -> Result<(Arc<Table>, Arc<Table>, Arc<Inner<A>>), Error> {
        let adds = ShareMeta::<A>::from(resources)?;
        let removes = ShareMeta::<R>::from(resources)?;
        let share = resources.try_global_with(table, || {
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
                if column.meta().size > 0 {
                    apply.push(index);
                }
            }

            if apply.len() == 0 && source.index() == target.index() {
                return Err(Error::TablesMustDiffer(source.index() as _));
            }

            let added = sorted_difference(target.types(), source.types()).collect();
            let removed = sorted_difference(source.types(), target.types()).collect();
            Ok(Self {
                source,
                target,
                inner: Arc::new(Inner {
                    state,
                    apply: apply.into_boxed_slice(),
                    added,
                    removed,
                }),
                _marker: PhantomData,
            })
        })?;
        let share = share.read();
        Ok((
            share.source.clone(),
            share.target.clone(),
            share.inner.clone(),
        ))
    }
}

fn move_to<'d, 'a, V, A: Template>(
    database: &Database<impl Listen>,
    set: &HashMap<Key, V>,
    templates: &mut Vec<A>,
    moves: &mut Vec<(usize, usize, NonZeroUsize)>,
    copies: &mut Vec<(usize, usize, NonZeroUsize)>,
    (source_table, source_keys): (&Table, RwLockUpgradableReadGuard<'a, Vec<Key>>),
    (target_table, target_keys): (&Table, RwLockUpgradableReadGuard<'a, Vec<Key>>),
    (low, high, count): (u32, u32, NonZeroUsize),
    rows: &mut Rows<'d>,
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
        &database.inner,
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
    for (i, (key, slot, _)) in rows.drain(..).enumerate() {
        slot.initialize(key.generation(), target_table.index(), (start + i) as u32);
    }

    drop(source_keys);
    // Although `source_keys` has been dropped, coherence will be maintained since the `target` lock prevents the keys from
    // moving again before `emit` is done.
    database.listen.on_modify(
        &target_keys[start..start + count.get()],
        &inner.added,
        &inner.removed,
    );
    drop(target_keys);
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
    database: &crate::Inner,
) {
    for &(source, target, count) in moves {
        source_keys.copy_within(source..source + count.get(), target);
        database
            .keys
            .update(&source_keys, target..target + count.get());
    }

    let mut index = 0;
    for source_column in source_table.columns() {
        let copy = source_column.meta().size > 0;
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
            debug_assert!(column.meta().size > 0);
            let _guard = column.data().write();
            lock(rest, table, with)
        }
        None => with(table),
    }
}
