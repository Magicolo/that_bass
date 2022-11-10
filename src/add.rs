use crate::{
    core::utility::{fold_swap, get_unchecked, get_unchecked_mut, ONE},
    filter::Filter,
    key::{Key, Slot},
    table::{self, Column, Table},
    template::{ApplyContext, InitializeContext, ShareMeta, Template},
    Database, Error,
};
use parking_lot::{RwLockReadGuard, RwLockUpgradableReadGuard, RwLockWriteGuard};
use std::{collections::HashMap, mem, num::NonZeroUsize, sync::Arc};

/// Adds template `T` to accumulated add operations.
pub struct Add<'d, T: Template, F: Filter = ()> {
    database: &'d Database,
    pairs: HashMap<Key, T>,
    pending: Vec<(Key, &'d Slot, T, u32)>,
    states: Vec<Result<State<'d, T>, u32>>,
    indices: Vec<usize>,
    filter: F,
}

/// Adds template `T` to all keys in tables that satisfy the filter `F`.
pub struct AddAll<'d, T: Template, F: Filter = ()> {
    database: &'d Database,
    index: usize,
    states: Vec<StateAll<T>>,
    filter: F,
}

struct State<'d, T: Template> {
    source: Arc<Table>,
    target: Arc<Table>,
    inner: Arc<Inner<T>>,
    // TODO: Merge `rows` and `templates`? How does this affect `slot.update/initialize`?
    rows: Vec<(Key, &'d Slot, u32)>,
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
    // TODO: A `Vec<usize>` should suffice here where its indices map to its values `source_column -> target_column`.
    copy: Box<[(usize, usize)]>,
}

struct ShareTable<T: Template> {
    source: Arc<Table>,
    target: Arc<Table>,
    inner: Arc<Inner<T>>,
}

impl Database {
    pub fn add<T: Template>(&self) -> Result<Add<T>, Error> {
        // Validate metas here, but there is no need to store them.
        ShareMeta::<T>::from(self).map(|_| Add {
            database: self,
            pairs: HashMap::new(),
            pending: Vec::new(),
            indices: Vec::new(),
            states: Vec::new(),
            filter: (),
        })
    }

    pub fn add_all<T: Template>(&self) -> Result<AddAll<T>, Error> {
        // Validate metas here, but there is no need to store them.
        ShareMeta::<T>::from(self).map(|_| AddAll {
            database: self,
            index: 0,
            states: Vec::new(),
            filter: (),
        })
    }
}

impl<'d, T: Template, F: Filter> Add<'d, T, F> {
    #[inline]
    pub fn one(&mut self, key: Key, template: T) {
        self.pairs.insert(key, template);
    }

    #[inline]
    pub fn all<I: IntoIterator<Item = (Key, T)>>(&mut self, templates: I) {
        self.pairs.extend(templates);
    }

    pub fn filter<G: Filter>(mut self, filter: G) -> Add<'d, T, (F, G)> {
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
        Add {
            database: self.database,
            pairs: self.pairs,
            pending: self.pending,
            states: self.states,
            indices: self.indices,
            filter: self.filter.and(filter),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn clear(&mut self) {
        debug_assert_eq!(self.pending.len(), 0);
        debug_assert_eq!(self.indices.len(), 0);
        self.pairs.clear();
    }

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

            for (key, slot, template, table) in self.pending.drain(..) {
                Self::sort(
                    key,
                    slot,
                    template,
                    table,
                    &mut self.indices,
                    &mut self.states,
                    &self.filter,
                    self.database,
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
                    unreachable!();
                };
                if state.rows.len() == 0 {
                    return Ok(sum);
                }
                let Some(source) = state.source.inner.try_write() else {
                    return Err(sum);
                };
                let Some(target) = state.target.inner.try_upgradable_read() else {
                    return Err(sum);
                };
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
                    self.database,
                    source,
                    (state.target.index(), target),
                    (low, high, count),
                    &mut state.rows,
                    &state.inner.copy,
                    &[],
                    |inner, index| {
                        // Initialize missing data `T` in target.
                        // SAFETY: Since this row is not yet observable by any thread but this one, bypass locks.
                        let context = ApplyContext::new(inner.columns());
                        for (i, template) in state.templates.drain(..).enumerate() {
                            unsafe { template.apply(&state.inner.state, context.with(index + i)) };
                        }
                    },
                );
                Ok(sum + count.get())
            },
            |sum, (states, pending), index| {
                let Some(Ok(state)) = states.get_mut(*index) else {
                    unreachable!();
                };
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
                    } else if state.source.index() > state.target.index() {
                        let target = state.target.inner.upgradable_read();
                        let source = state.source.inner.write();
                        let (low, high) = Self::retain(&state.source, &mut state.rows, &mut state.templates, pending);
                        let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                        // Happens if all keys from this table have been moved or destroyed between here and the sorting.
                        return sum ;
                    };
                        (source, target, low, high, count)
                    } else {
                        unreachable!()
                    };
                move_to(
                    self.database,
                    source,
                    (state.target.index(), target),
                    (low, high, count),
                    &mut state.rows,
                    &state.inner.copy,
                    &[],
                    |inner, index| {
                        // Initialize missing data `T` in target.
                        // SAFETY: Since this row is not yet observable by any thread but this one, bypass locks.
                        let context = ApplyContext::new(inner.columns());
                        for (i, template) in state.templates.drain(..).enumerate() {
                            unsafe { template.apply(&state.inner.state, context.with(index + i)) };
                        }
                    },
                );
                sum + count.get()
            },
        )
    }

    fn sort(
        key: Key,
        slot: &'d Slot,
        template: T,
        table: u32,
        indices: &mut Vec<usize>,
        states: &mut Vec<Result<State<'d, T>, u32>>,
        filter: &F,
        database: &'d Database,
    ) {
        let index = match states.binary_search_by_key(&table, |result| match result {
            Ok(state) => state.source.index(),
            Err(index) => *index,
        }) {
            Ok(index) => index,
            Err(index) => {
                let result = match ShareTable::<T>::from(table, database) {
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
            state.rows.push((key, slot, u32::MAX));
            state.templates.push(template);
        }
    }

    /// Call this while holding a lock on `table`.
    fn retain<'a>(
        table: &'a Table,
        rows: &mut Vec<(Key, &'d Slot, u32)>,
        templates: &mut Vec<T>,
        pending: &mut Vec<(Key, &'d Slot, T, u32)>,
    ) -> (u32, u32) {
        let mut low = u32::MAX;
        let mut high = 0;
        for i in (0..rows.len()).rev() {
            let (key, slot, row) = unsafe { get_unchecked_mut(rows, i) };
            if let Ok(table_index) = slot.table(key.generation()) {
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

impl<'d, T: Template, F: Filter> AddAll<'d, T, F> {
    #[inline]
    pub fn resolve(&mut self, set: bool) -> usize
    where
        T: Default,
    {
        self.resolve_with(set, T::default)
    }

    pub fn resolve_with<G: FnMut() -> T>(&mut self, set: bool, with: G) -> usize {
        while let Ok(table) = self.database.tables().get(self.index) {
            self.index += 1;
            match ShareTable::from(table.index(), self.database) {
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
                let count = if state.source.index() < state.target.index() {
                    let source = state.source.inner.try_write().ok_or(sum)?;
                    let target = state.target.inner.try_upgradable_read().ok_or(sum)?;
                    Self::resolve_tables(source, target, state, self.database, with)
                } else if state.source.index() > state.target.index() {
                    let target = state.target.inner.try_upgradable_read().ok_or(sum)?;
                    let source = state.source.inner.try_write().ok_or(sum)?;
                    Self::resolve_tables(source, target, state, self.database, with)
                } else if set {
                    let inner = state.source.inner.try_read().ok_or(sum)?;
                    Self::resolve_table(inner, state, with)
                } else {
                    0
                };
                Ok(sum + count)
            },
            |sum, with, state| {
                sum + if state.source.index() < state.target.index() {
                    let source = state.source.inner.write();
                    let target = state.target.inner.upgradable_read();
                    Self::resolve_tables(source, target, state, self.database, with)
                } else if state.source.index() > state.target.index() {
                    let target = state.target.inner.upgradable_read();
                    let source = state.source.inner.write();
                    Self::resolve_tables(source, target, state, self.database, with)
                } else if set {
                    let inner = state.source.inner.read();
                    Self::resolve_table(inner, state, with)
                } else {
                    0
                }
            },
        )
    }

    fn resolve_tables(
        mut source: RwLockWriteGuard<table::Inner>,
        target: RwLockUpgradableReadGuard<table::Inner>,
        state: &StateAll<T>,
        database: &Database,
        with: &mut impl FnMut() -> T,
    ) -> usize {
        let Some(count) = NonZeroUsize::new(*source.count.get_mut() as _) else {
            return 0;
        };
        source.release(count);

        let (start, target) = table::Inner::reserve(target, count);
        let (source, target) = (&mut *source, &*target);
        let keys = (source.keys.get_mut(), unsafe { &mut *target.keys.get() });
        copy_to(
            (0, keys.0, &mut source.columns),
            (start, keys.1, target.columns()),
            count,
            &state.inner.copy,
            &[],
        );

        // SAFETY: Since this row is not yet observable by any thread but this one, bypass locks.
        let context = ApplyContext::new(target.columns());
        for i in 0..count.get() {
            unsafe { with().apply(&state.inner.state, context.with(start + i)) };
        }

        target.commit(count);
        // Slots must be updated after the table `commit` to prevent a `query::find` to be able to observe a row which
        // has an index greater than the `table.count()`. As long as the slots remain in the source table, all accesses
        // to these keys will block at the table access and will correct their table index after they acquire the source
        // table lock.
        database
            .keys()
            .initialize(keys.1, state.target.index(), start..start + count.get());
        count.get()
        // Keep the `source` and `target` locks until all table operations are fully completed.
    }

    fn resolve_table(
        inner: RwLockReadGuard<table::Inner>,
        state: &StateAll<T>,
        with: &mut impl FnMut() -> T,
    ) -> usize {
        let Some(count) = NonZeroUsize::new(inner.count()) else {
            return 0;
        };
        lock(&state.inner.apply, &inner, || {
            let context = ApplyContext::new(inner.columns());
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
        database: &Database,
    ) -> Result<(Arc<Table>, Arc<Table>, Arc<Inner<T>>), Error> {
        let share = database.resources().try_global_with(table, || {
            let metas = ShareMeta::<T>::from(database)?;
            let source = database.tables().get_shared(table as usize)?;
            let target = {
                let mut metas = metas.to_vec();
                for &meta in source.metas() {
                    match metas.binary_search_by_key(&meta.identifier(), |meta| meta.identifier()) {
                        Ok(_) => {}
                        Err(index) => metas.insert(index, meta),
                    }
                }
                database.tables().find_or_add(&metas)
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

type Rows<'d> = Vec<(Key, &'d Slot, u32)>;

pub(crate) fn move_to<'d, 'a>(
    database: &'d Database,
    mut source: RwLockWriteGuard<'a, table::Inner>,
    (index, target): (u32, RwLockUpgradableReadGuard<'a, table::Inner>),
    (low, high, count): (u32, u32, NonZeroUsize),
    rows: &mut Rows<'d>,
    copy: &[(usize, usize)],
    drop: &[usize],
    initialize: impl FnOnce(&table::Inner, usize),
) {
    let (start, target) = table::Inner::reserve(target, count);
    // Move data from source to target.
    let range = low..high + 1;
    let head = source.release(count);
    let (low, high) = (range.start as usize, range.end as usize);

    if range.len() == count.get() {
        // Fast path. The move range is contiguous. Copy everything from source to target at once.
        let source = &mut *source;
        let keys = (source.keys.get_mut(), unsafe { &mut *target.keys.get() });
        copy_to(
            (low, keys.0, &mut source.columns),
            (start, keys.1, target.columns()),
            count,
            copy,
            drop,
        );

        for &index in drop {
            let source = unsafe { get_unchecked_mut(&mut source.columns, index) };
            unsafe { source.drop(low, count) };
        }

        // Swap remove without dropping.
        let over = high.saturating_sub(head);
        let end = count.get() - over;
        if let Some(end) = NonZeroUsize::new(end) {
            let start = head + over;
            // Copy the range at the end of the table on the beginning of the removed range.
            for column in source.columns.iter_mut() {
                unsafe { column.copy(start, low, end) };
            }

            // Update the keys.
            keys.0.copy_within(start..start + end.get(), low);
            database.keys().update(keys.0, low..low + end.get());
        }
    } else {
        // Range is not contiguous; use the slow path.
        let source = &mut *source;
        let keys = (source.keys.get_mut(), unsafe { &mut *target.keys.get() });
        for (i, &(.., row)) in rows.iter().enumerate() {
            copy_to(
                (row as usize, keys.0, &mut source.columns),
                (start + i, keys.1, target.columns()),
                ONE,
                copy,
                drop,
            );
            // Tag keys that are going to be removed such that removed keys and valid keys can be differentiated.
            unsafe { *get_unchecked_mut(keys.0, row as usize) = Key::NULL };
        }

        let mut cursor = head;
        for &(.., row) in rows.iter() {
            let row = row as usize;
            if row < head {
                // Find the next valid row to move.
                while unsafe { *get_unchecked(keys.0, cursor) } == Key::NULL {
                    cursor += 1;
                }
                debug_assert!(cursor < head + count.get());

                for column in source.columns.iter_mut() {
                    unsafe { column.squash(cursor, row, ONE) };
                }

                let key = unsafe { *get_unchecked_mut(keys.0, cursor) };
                unsafe { *get_unchecked_mut(keys.0, row) = key };
                let slot = unsafe { database.keys().get_unchecked(key) };
                slot.update(row as _);
                cursor += 1;
            }
        }
    }

    initialize(&target, start);
    target.commit(count);
    // Slots must be updated after the table `commit` to prevent a `query::find` to be able to observe a row which
    // has an index greater than the `table.count()`. As long as the slots remain in the source table, all accesses
    // to these keys will block at the table access and will correct their table index after they acquire the source
    // table lock.
    for (i, (key, slot, ..)) in rows.drain(..).enumerate() {
        slot.initialize(key.generation(), index, (start + i) as u32);
    }
    // Keep the `source` and `target` locks until all table operations are fully completed.
    mem::drop(source);
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
            if column.meta().size == 0 {
                // TODO: No need to recurse here.
                lock(rest, inner, with)
            } else {
                let _guard = column.data().write();
                lock(rest, inner, with)
            }
        }
        None => with(),
    }
}
