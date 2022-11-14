use crate::{
    core::utility::{fold_swap, get_unchecked, get_unchecked_mut, unreachable, ONE},
    event::Listen,
    filter::Filter,
    key::{Key, Slot},
    resources::Resources,
    table::{self, Column, Table, Tables},
    template::{ApplyContext, InitializeContext, ShareMeta, Template},
    Database, Error,
};
use parking_lot::{RwLockReadGuard, RwLockUpgradableReadGuard, RwLockWriteGuard};
use std::{collections::HashMap, mem, num::NonZeroUsize, sync::Arc};

/// Adds template `T` to accumulated add operations.
pub struct Add<'d, T: Template, F, L> {
    database: &'d Database<L>,
    pairs: HashMap<Key, T>, // A `HashMap` is used because the move algorithm assumes that rows will be unique.
    indices: Vec<usize>,    // May be reordered (ex: by `fold_swap`).
    states: Vec<Result<State<'d, T>, u32>>, // Must remain sorted by `state.source.index()` for `binary_search` to work.
    pending: Vec<(Key, &'d Slot, T, u32)>,
    filter: F,
}

/// Adds template `T` to all keys in tables that satisfy the filter `F`.
pub struct AddAll<'d, T: Template, F, L> {
    database: &'d Database<L>,
    index: usize,
    states: Vec<StateAll<T>>,
    filter: F,
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
    copy: Box<[(usize, usize)]>,
}

struct ShareTable<T: Template> {
    source: Arc<Table>,
    target: Arc<Table>,
    inner: Arc<Inner<T>>,
}

impl<L> Database<L> {
    pub fn add<T: Template>(&self) -> Result<Add<T, (), L>, Error> {
        // Validate metas here, but there is no need to store them.
        ShareMeta::<T>::from(self.resources()).map(|_| Add {
            database: self,
            pairs: HashMap::new(),
            pending: Vec::new(),
            indices: Vec::new(),
            states: Vec::new(),
            filter: (),
        })
    }

    pub fn add_all<T: Template>(&self) -> Result<AddAll<T, (), L>, Error> {
        // Validate metas here, but there is no need to store them.
        ShareMeta::<T>::from(self.resources()).map(|_| AddAll {
            database: self,
            index: 0,
            states: Vec::new(),
            filter: (),
        })
    }
}

impl<'d, T: Template, F, L> Add<'d, T, F, L> {
    #[inline]
    pub fn one(&mut self, key: Key, template: T) {
        self.pairs.insert(key, template);
    }

    #[inline]
    pub fn all<I: IntoIterator<Item = (Key, T)>>(&mut self, templates: I) {
        self.pairs.extend(templates);
    }

    pub fn filter<G: Filter + Default>(self) -> Add<'d, T, (F, G), L> {
        self.filter_with(G::default())
    }

    pub fn filter_with<G: Filter>(mut self, filter: G) -> Add<'d, T, (F, G), L> {
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
        Add {
            database: self.database,
            pairs: self.pairs,
            pending: self.pending,
            states: self.states,
            indices: self.indices,
            filter: (self.filter, filter),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (Key, &T)> {
        self.pairs.iter().map(|pair| (*pair.0, pair.1))
    }

    pub fn drain(&mut self) -> impl ExactSizeIterator<Item = (Key, T)> + '_ {
        debug_assert_eq!(self.pending.len(), 0);
        debug_assert_eq!(self.indices.len(), 0);
        self.pairs.drain()
    }

    pub fn clear(&mut self) {
        debug_assert_eq!(self.pending.len(), 0);
        debug_assert_eq!(self.indices.len(), 0);
        self.pairs.clear();
    }
}

impl<'d, T: Template, F: Filter, L: Listen> Add<'d, T, F, L> {
    pub fn resolve(&mut self) -> usize {
        for (key, template) in self.pairs.drain() {
            if let Ok((slot, table)) = self.database.keys().get(key) {
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
        sum
    }

    fn resolve_sorted(&mut self) -> usize {
        fold_swap(
            &mut self.indices,
            0,
            (&mut self.states, &mut self.pending),
            |sum, (states, pending), index| {
                let Some(Ok(state)) = states.get_mut(*index) else {
                    unsafe { unreachable() };
                };
                if state.rows.len() == 0 {
                    return Ok(sum);
                }
                if state.source.index() == state.target.index() {
                    let inner = state.source.inner.try_read().ok_or(sum)?;
                    Self::retain(
                        &state.source,
                        &mut state.rows,
                        &mut state.templates,
                        pending,
                    );
                    let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                        return Ok(sum);
                    };
                    Self::resolve_set(
                        inner,
                        &state.inner.state,
                        &mut state.rows,
                        &mut state.templates,
                        &state.inner.apply,
                    );
                    return Ok(sum + count.get());
                }
                let source = state.source.inner.try_write().ok_or(sum)?;
                let target = state.target.inner.try_upgradable_read().ok_or(sum)?;
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
                    &self.database.inner,
                    source,
                    (state.target.index(), target),
                    (low, high, count),
                    &mut state.rows,
                    &state.inner.copy,
                    &[],
                    |keys, columns, index| {
                        // Initialize missing data `T` in target.
                        // SAFETY: Since this row is not yet observable by any thread but this one, bypass locks.
                        let context = ApplyContext::new(keys, columns);
                        for (i, template) in state.templates.drain(..).enumerate() {
                            unsafe { template.apply(&state.inner.state, context.with(index + i)) };
                        }
                    },
                    |keys| {
                        self.database
                            .listen
                            .on_add(keys, &state.source, &state.target)
                    },
                );
                Ok(sum + count.get())
            },
            |sum, (states, pending), index| {
                let Some(Ok(state)) = states.get_mut(*index) else {
                    unsafe { unreachable() };
                };
                if state.rows.len() == 0 {
                    return sum;
                }
                if state.source.index() == state.target.index() {
                    let inner = state.source.inner.read();
                    Self::retain(
                        &state.source,
                        &mut state.rows,
                        &mut state.templates,
                        pending,
                    );
                    let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                        return sum;
                    };
                    Self::resolve_set(
                        inner,
                        &state.inner.state,
                        &mut state.rows,
                        &mut state.templates,
                        &state.inner.apply,
                    );
                    return sum + count.get();
                }

                let (source, target, low, high, count) =
                    // If locks are always taken in order (lower index first), there can not be a deadlock between move operations.
                    if state.source.index() < state.target.index() {
                        let source = state.source.inner.write();
                        let (low, high) = Self::retain(&state.source, &mut state.rows, &mut state.templates, pending);
                        let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                            // Happens if all keys from this table have been moved or destroyed between here and the sorting.
                            return sum;
                        };
                        let target = state.target.inner.upgradable_read();
                        (source, target, low, high, count)
                    } else  {
                        let target = state.target.inner.upgradable_read();
                        let source = state.source.inner.write();
                        let (low, high) = Self::retain(&state.source, &mut state.rows, &mut state.templates, pending);
                        let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                            // Happens if all keys from this table have been moved or destroyed between here and the sorting.
                            return sum;
                        };
                        (source, target, low, high, count)
                    };
                move_to(
                    &self.database.inner,
                    source,
                    (state.target.index(), target),
                    (low, high, count),
                    &mut state.rows,
                    &state.inner.copy,
                    &[],
                    |keys, columns, index| {
                        // Initialize missing data `T` in target.
                        // SAFETY: Since this row is not yet observable by any thread but this one, bypass locks.
                        let context = ApplyContext::new(keys, columns);
                        for (i, template) in state.templates.drain(..).enumerate() {
                            unsafe { template.apply(&state.inner.state, context.with(index + i)) };
                        }
                    },
                    |keys| {
                        self.database
                            .listen
                            .on_add(keys, &state.source, &state.target)
                    },
                );
                sum + count.get()
            },
        )
    }

    fn resolve_set<'a>(
        inner: RwLockReadGuard<'a, table::Inner>,
        state: &T::State,
        rows: &mut Rows<'d>,
        templates: &mut Vec<T>,
        apply: &[usize],
    ) {
        // The keys do not need to be moved, simply write the row data.
        let table_keys = inner.keys();
        debug_assert!(rows.len() <= table_keys.len());
        lock(apply, &inner, || {
            let context = ApplyContext::new(table_keys, inner.columns());
            for ((.., row), template) in rows.drain(..).zip(templates.drain(..)) {
                debug_assert!(row < u32::MAX);
                unsafe { template.apply(state, context.with(row as _)) };
            }
        });
    }

    fn sort(
        key: Key,
        slot: &'d Slot,
        template: T,
        table: u32,
        indices: &mut Vec<usize>,
        states: &mut Vec<Result<State<'d, T>, u32>>,
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
                    match ShareTable::<T>::from(table, &database.tables, &database.resources) {
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
    fn retain<'a>(
        table: &'a Table,
        rows: &mut Rows<'d>,
        templates: &mut Vec<T>,
        pending: &mut Vec<(Key, &'d Slot, T, u32)>,
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

impl<'d, T: Template, F, L> AddAll<'d, T, F, L> {
    pub fn filter<G: Filter + Default>(self) -> AddAll<'d, T, (F, G), L> {
        self.filter_with(G::default())
    }

    pub fn filter_with<G: Filter>(mut self, filter: G) -> AddAll<'d, T, (F, G), L> {
        self.states
            .retain(|state| filter.filter(&state.source, self.database.into()));
        AddAll {
            database: self.database,
            index: self.index,
            states: self.states,
            filter: (self.filter, filter),
        }
    }
}

impl<'d, T: Template, F: Filter, L: Listen> AddAll<'d, T, F, L> {
    #[inline]
    pub fn resolve(&mut self) -> usize
    where
        T: Default,
    {
        self.resolve_with(true, T::default)
    }

    pub fn resolve_with<G: FnMut() -> T>(&mut self, set: bool, with: G) -> usize {
        while let Ok(table) = self.database.tables().get(self.index) {
            self.index += 1;
            match ShareTable::from(
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
                    let mut source = state.source.inner.try_write().ok_or(sum)?;
                    let Some(count) = NonZeroUsize::new(*source.count.get_mut() as _) else {
                        return Ok(sum);
                    };
                    let target = state.target.inner.try_upgradable_read().ok_or(sum)?;
                    Self::resolve_tables(source, target, state, count, self.database, with);
                    Ok(sum + count.get())
                } else if set {
                    let inner = state.source.inner.try_read().ok_or(sum)?;
                    Ok(sum + Self::resolve_table(inner, state, with))
                } else {
                    Ok(sum)
                }
            },
            |sum, with, state| {
                if state.source.index() < state.target.index() {
                    let mut source = state.source.inner.write();
                    let Some(count) = NonZeroUsize::new(*source.count.get_mut() as _) else {
                        return sum;
                    };
                    let target = state.target.inner.upgradable_read();
                    Self::resolve_tables(source, target, state, count, self.database, with);
                    sum + count.get()
                } else if state.source.index() > state.target.index() {
                    let target = state.target.inner.upgradable_read();
                    let mut source = state.source.inner.write();
                    let Some(count) = NonZeroUsize::new(*source.count.get_mut() as _) else {
                        return sum;
                    };
                    Self::resolve_tables(source, target, state, count, self.database, with);
                    sum + count.get()
                } else if set {
                    let inner = state.source.inner.read();
                    sum + Self::resolve_table(inner, state, with)
                } else {
                    sum
                }
            },
        )
    }

    fn resolve_tables(
        mut source: RwLockWriteGuard<table::Inner>,
        target: RwLockUpgradableReadGuard<table::Inner>,
        state: &StateAll<T>,
        count: NonZeroUsize,
        database: &Database<impl Listen>,
        mut with: impl FnMut() -> T,
    ) {
        let (start, target) = table::Inner::reserve(target, count);
        let remain = source.release(count);
        debug_assert_eq!(remain, 0);

        let source_inner = &mut *source;
        let source_keys = source_inner.keys.get_mut();
        let target_inner = &*target;
        let target_keys = unsafe { &mut *target.keys.get() };
        copy_to(
            (0, source_keys, &mut source_inner.columns),
            (start, target_keys, target_inner.columns()),
            count,
            &state.inner.copy,
            &[],
        );

        // SAFETY: Since this row is not yet observable by any thread but this one, bypass locks.
        let context = ApplyContext::new(target_keys, target.columns());
        for i in 0..count.get() {
            unsafe { with().apply(&state.inner.state, context.with(start + i)) };
        }

        target.commit(count);
        // Slots must be updated after the table `commit` to prevent a `query::find` to be able to observe a row which
        // has an index greater than the `table.count()`. As long as the slots remain in the source table, all accesses
        // to these keys will block at the table access and will correct their table index after they acquire the source
        // table lock.
        database.keys().initialize(
            target_keys,
            state.target.index(),
            start..start + count.get(),
        );
        drop(source);
        // Although `source` has been dropped, coherence with be maintained since the `target` lock prevent the keys
        // moving again before `on_add` is done.
        database.listen.on_add(
            &target_keys[start..start + count.get()],
            &state.source,
            &state.target,
        );
    }

    fn resolve_table(
        inner: RwLockReadGuard<table::Inner>,
        state: &StateAll<T>,
        with: &mut impl FnMut() -> T,
    ) -> usize {
        let keys = inner.keys();
        let Some(count) = NonZeroUsize::new(keys.len()) else {
            return 0;
        };
        lock(&state.inner.apply, &inner, || {
            let context = ApplyContext::new(keys, inner.columns());
            for i in 0..count.get() {
                unsafe { with().apply(&state.inner.state, context.with(i)) };
            }
        });
        count.get()
    }
}

impl<T: Template> ShareTable<T> {
    pub fn from(
        table: u32,
        tables: &Tables,
        resources: &Resources,
    ) -> Result<(Arc<Table>, Arc<Table>, Arc<Inner<T>>), Error> {
        let share = resources.try_global_with(table, || {
            let metas = ShareMeta::<T>::from(resources)?;
            let source = tables.get_shared(table as usize)?;
            let target = {
                let mut metas = metas.to_vec();
                for &meta in source.metas() {
                    match metas.binary_search_by_key(&meta.identifier(), |meta| meta.identifier()) {
                        Ok(_) => {}
                        Err(index) => metas.insert(index, meta),
                    }
                }
                tables.find_or_add(&metas)
            };
            let state = T::initialize(InitializeContext::new(&target))?;

            let mut copy = Vec::new();
            for source in source.metas().iter().enumerate() {
                let target = target.column_with(source.1.identifier())?;
                debug_assert_eq!(source.1.identifier(), target.1.identifier());
                if target.1.size > 0 {
                    copy.push((source.0, target.0));
                }
            }

            let mut apply = Vec::new();
            for meta in metas.iter() {
                let (index, meta) = target.column_with(meta.identifier())?;
                if meta.size > 0 {
                    apply.push(index);
                }
            }

            Ok(ShareTable::<T> {
                source,
                target,
                inner: Arc::new(Inner {
                    state,
                    apply: apply.into_boxed_slice(),
                    copy: copy.into_boxed_slice(),
                }),
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

pub(crate) fn move_to<'d, 'a>(
    database: &crate::Inner,
    mut source: RwLockWriteGuard<'a, table::Inner>,
    (index, target): (u32, RwLockUpgradableReadGuard<'a, table::Inner>),
    (low, high, count): (u32, u32, NonZeroUsize),
    rows: &mut Rows<'d>,
    copy: &[(usize, usize)],
    drop: &[usize],
    initialize: impl FnOnce(&[Key], &[Column], usize),
    emit: impl FnOnce(&[Key]),
) {
    let (start, target) = table::Inner::reserve(target, count);
    // Move data from source to target.
    let range = low..high + 1;
    let head = source.release(count);
    let source_inner = &mut *source;
    let source_keys = source_inner.keys.get_mut();
    let target_inner = &*target;
    let target_keys = unsafe { &mut *target_inner.keys.get() };
    let (low, high) = (range.start as usize, range.end as usize);

    if range.len() == count.get() {
        // Fast path. The move range is contiguous. Copy everything from source to target at once.
        copy_to(
            (low, source_keys, &mut source_inner.columns),
            (start, target_keys, target_inner.columns()),
            count,
            copy,
            drop,
        );

        for &index in drop {
            let source = unsafe { get_unchecked_mut(&mut source_inner.columns, index) };
            unsafe { source.drop(low, count) };
        }

        // Swap remove without dropping.
        let over = high.saturating_sub(head);
        let end = count.get() - over;
        if let Some(end) = NonZeroUsize::new(end) {
            let start = head + over;
            // Copy the range at the end of the table on the beginning of the removed range.
            for column in source_inner.columns.iter_mut() {
                unsafe { column.copy(start, low, end) };
            }

            // Update the keys.
            source_keys.copy_within(start..start + end.get(), low);
            database.keys.update(source_keys, low..low + end.get());
        }
    } else {
        // Range is not contiguous; use the slow path.
        for (i, &(.., row)) in rows.iter().enumerate() {
            copy_to(
                (row as usize, source_keys, &mut source_inner.columns),
                (start + i, target_keys, target_inner.columns()),
                ONE,
                copy,
                drop,
            );
            // Tag keys that are going to be removed such that removed keys and valid keys can be differentiated.
            unsafe { *get_unchecked_mut(source_keys, row as usize) = Key::NULL };
        }

        let mut cursor = head;
        for &(.., row) in rows.iter() {
            let row = row as usize;
            if row < head {
                // Find the next valid row to move.
                while unsafe { *get_unchecked(source_keys, cursor) } == Key::NULL {
                    cursor += 1;
                }
                debug_assert!(cursor < head + count.get());

                for column in source_inner.columns.iter_mut() {
                    unsafe { column.squash(cursor, row, ONE) };
                }

                let key = unsafe { *get_unchecked_mut(source_keys, cursor) };
                unsafe { *get_unchecked_mut(source_keys, row) = key };
                let slot = unsafe { database.keys.get_unchecked(key) };
                slot.update(row as _);
                cursor += 1;
            }
        }
    }

    initialize(
        unsafe { &*target_inner.keys.get() },
        target.columns(),
        start,
    );
    target_inner.commit(count);
    // Slots must be updated after the table `commit` to prevent a `query::find` to be able to observe a row which
    // has an index greater than the `table.count()`. As long as the slots remain in the source table, all accesses
    // to these keys will block at the table access and will correct their table index after they acquire the source
    // table lock.
    for (i, (key, slot, _)) in rows.drain(..).enumerate() {
        slot.initialize(key.generation(), index, (start + i) as u32);
    }

    mem::drop(source);
    // Although `source` has been dropped, coherence with be maintained since the `target` lock prevent the keys
    // moving again before `emit` is done.
    emit(&target_keys[start..start + count.get()]);
    mem::drop(target);
}

#[inline]
pub(crate) fn copy_to(
    source: (usize, &[Key], &mut [Column]),
    target: (usize, &mut [Key], &[Column]),
    count: NonZeroUsize,
    copy: &[(usize, usize)],
    drop: &[usize],
) {
    target.1[target.0..target.0 + count.get()]
        .copy_from_slice(&source.1[source.0..source.0 + count.get()]);
    for &indices in copy {
        let source = (unsafe { get_unchecked_mut(source.2, indices.0) }, source.0);
        let target = (unsafe { get_unchecked(target.2, indices.1) }, target.0);
        unsafe { Column::copy_to(source, target, count) };
    }

    for &index in drop {
        let source = (unsafe { get_unchecked_mut(source.2, index) }, source.0);
        unsafe { source.0.drop(source.1, count) };
    }
}

#[inline]
fn lock<T>(indices: &[usize], inner: &table::Inner, with: impl FnOnce() -> T) -> T {
    match indices.split_first() {
        Some((&index, rest)) => {
            let column = unsafe { get_unchecked(inner.columns(), index) };
            debug_assert!(column.meta().size > 0);
            let _guard = column.data().write();
            lock(rest, inner, with)
        }
        None => with(),
    }
}
