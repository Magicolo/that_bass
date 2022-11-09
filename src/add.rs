use crate::{
    core::utility::{fold_swap, get_unchecked, get_unchecked_mut, ONE},
    filter::Filter,
    key::{Key, Slot},
    resources::Global,
    table::{self, Column, Table},
    template::{ApplyContext, InitializeContext, ShareMeta, Template},
    Database, Error,
};
use parking_lot::{RwLockReadGuard, RwLockUpgradableReadGuard, RwLockWriteGuard};
use std::{collections::HashMap, marker::PhantomData, mem, num::NonZeroUsize, sync::Arc};

/// Adds template `T` to accumulated add operations.
pub struct Add<'d, T: Template> {
    database: &'d Database,
    keys: HashMap<Key, u32>,
    pending: Vec<(Key, &'d Slot, T, u32)>,
    sorted: HashMap<u32, State<'d, T>>,
}

/// Adds template `T` to all keys in tables that satisfy the filter `F`.
pub struct AddAll<'d, T: Template, F: Filter> {
    database: &'d Database,
    index: usize,
    states: Vec<StateAll<T>>,
    _marker: PhantomData<fn(F)>,
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
            keys: HashMap::new(),
            pending: Vec::new(),
            sorted: HashMap::new(),
        })
    }

    pub fn add_all<T: Template, F: Filter>(&self) -> Result<AddAll<T, F>, Error> {
        // Validate metas here, but there is no need to store them.
        ShareMeta::<T>::from(self).map(|_| AddAll {
            database: self,
            index: 0,
            states: Vec::new(),
            _marker: PhantomData,
        })
    }
}

impl<'d, T: Template> Add<'d, T> {
    #[inline]
    pub fn one(&mut self, key: Key, template: T) -> Result<(), Error> {
        let (slot, table) = self.database.keys().get(key)?;
        Self::sort(key, slot, template, table, &mut self.sorted, self.database)
    }

    #[inline]
    pub fn all<I: IntoIterator<Item = (Key, T)>>(&mut self, templates: I) -> usize {
        templates
            .into_iter()
            .filter_map(|(key, template)| self.one(key, template).ok())
            .count()
    }

    pub fn resolve(&mut self) -> usize {
        loop {
            self.resolve_sorted();
            if self.pending.len() == 0 {
                break;
            }

            for (key, slot, template, table) in self.pending.drain(..) {
                let _ = Self::sort(key, slot, template, table, &mut self.sorted, self.database);
            }
        }
        let count = self.keys.len();
        self.keys.clear();
        count
    }

    fn resolve_sorted(&mut self) {
        // TODO: Maybe do not iterate over all pairs?
        for state in self.sorted.values_mut() {
            move_to(
                (&mut self.keys, &mut state.templates, &mut self.pending),
                self.database,
                &state.source,
                &state.target,
                &mut state.rows,
                &state.inner.copy,
                &[],
                |(keys, templates, pending), rows| {
                    Self::filter(&state.source, keys, rows, templates, pending)
                },
                |(keys, templates, pending), rows| {
                    // The keys do not need to be moved, simply write the row data.
                    let inner = state.source.inner.read();
                    Self::filter(&state.source, keys, rows, templates, pending);
                    if rows.len() > 0 && T::SIZE > 0 {
                        lock(&state.inner.apply, &inner, || {
                            let context = ApplyContext::new(inner.columns());
                            for (i, template) in templates.drain(..).enumerate() {
                                let &(.., row) = unsafe { get_unchecked(rows, i) };
                                debug_assert!(row < u32::MAX);
                                unsafe {
                                    template.apply(&state.inner.state, context.with(row as _))
                                };
                            }
                            rows.clear();
                        });
                    } else {
                        rows.clear();
                        templates.clear();
                    }
                },
                |(_, templates, _), inner, index| {
                    // Initialize missing data `T` in target.
                    if T::SIZE > 0 {
                        // SAFETY: Since this row is not yet observable by any thread but this one, bypass locks.
                        let context = ApplyContext::new(inner.columns());
                        for (i, template) in templates.drain(..).enumerate() {
                            unsafe { template.apply(&state.inner.state, context.with(index + i)) };
                        }
                    } else {
                        templates.clear();
                    }
                },
            );

            // Sanity checks.
            debug_assert!(state.rows.is_empty());
            debug_assert!(state.templates.is_empty());
        }
    }

    fn sort(
        key: Key,
        slot: &'d Slot,
        template: T,
        table: u32,
        sorted: &mut HashMap<u32, State<'d, T>>,
        database: &'d Database,
    ) -> Result<(), Error> {
        match sorted.get_mut(&table) {
            Some(state) => {
                state.rows.push((key, slot, u32::MAX));
                state.templates.push(template);
                Ok(())
            }
            None => {
                let share = ShareTable::from(table, database)?;
                let share = share.read();
                sorted.insert(
                    table,
                    State {
                        source: share.source.clone(),
                        target: share.target.clone(),
                        inner: share.inner.clone(),
                        rows: vec![(key, slot, u32::MAX)],
                        templates: vec![template],
                    },
                );
                Ok(())
            }
        }
    }

    /// Call this while holding a lock on `table`.
    fn filter<'a>(
        table: &'a Table,
        keys: &mut HashMap<Key, u32>,
        rows: &mut Vec<(Key, &'d Slot, u32)>,
        templates: &mut Vec<T>,
        pending: &mut Vec<(Key, &'d Slot, T, u32)>,
    ) -> (u32, u32) {
        let mut low = u32::MAX;
        let mut high = 0;
        let mut index = rows.len();

        // Iterate in reverse to prevent the `ABBAA` problem where `A2` is considered the latest `A` in place of `A3`. This happened
        // with the previous template swapping algorithm.
        while index > 0 {
            index -= 1;

            let (key, slot, row) = unsafe { get_unchecked_mut(rows, index) };
            if let Ok(table_index) = slot.table(key.generation()) {
                if table_index == table.index() {
                    // Duplicates must only be checked here where the key would be guaranteed to be added.
                    // - This way, a proper count of added can be collected.
                    // - The removal algorithm also assumes that there is no duplicate rows.
                    match keys.insert(*key, table_index) {
                        Some(table) if table == table_index => {
                            // If the key was already seen, discard the earlier template.
                            rows.swap_remove(index);
                            templates.swap_remove(index);
                        }
                        _ => {
                            // It is possible that the key has already been processed in another table and moved to this one.
                            // - If this is the case, it needs to be reprocessed.
                            *row = slot.row();
                            low = low.min(*row);
                            high = high.max(*row);
                        }
                    }
                } else {
                    let (key, slot, _) = rows.swap_remove(index);
                    let template = templates.swap_remove(index);
                    pending.push((key, slot, template, table_index));
                }
            } else {
                rows.swap_remove(index);
                templates.swap_remove(index);
            }
        }

        debug_assert_eq!(low <= high, rows.len() > 0);
        (low, high)
    }
}

impl<T: Template> Drop for Add<'_, T> {
    fn drop(&mut self) {
        self.resolve();
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
            if let Ok(share) = ShareTable::from(table.index(), self.database) {
                let share = share.read();
                self.states.push(StateAll {
                    source: share.source.clone(),
                    target: share.target.clone(),
                    inner: share.inner.clone(),
                });
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
                } else if T::SIZE > 0 && set {
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
                } else if T::SIZE > 0 && set {
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

        if T::SIZE > 0 {
            // SAFETY: Since this row is not yet observable by any thread but this one, bypass locks.
            let context = ApplyContext::new(target.columns());
            for i in 0..count.get() {
                unsafe { with().apply(&state.inner.state, context.with(start + i)) };
            }
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
    pub fn from(table: u32, database: &Database) -> Result<Global<Self>, Error> {
        database.resources().try_global_with(table, || {
            let metas = ShareMeta::<T>::from(database)?;
            let source = database.tables().get_shared(table as usize)?;
            let mut target_metas = metas.to_vec();
            target_metas.extend(source.inner.read().columns().iter().map(Column::meta));
            let target = database.tables().find_or_add(target_metas);
            let state = T::initialize(InitializeContext::new(&target))?;

            let mut copy = Vec::new();
            for (source, &identifier) in source.types().iter().enumerate() {
                copy.push((source, target.column_with(identifier)?));
            }

            let mut add = Vec::new();
            for meta in metas.iter() {
                add.push(target.column_with(meta.identifier())?);
            }

            debug_assert_eq!(source.types().len(), copy.len());
            debug_assert_eq!(target.types().len(), copy.len() + add.len());

            Ok(ShareTable::<T> {
                source,
                target,
                inner: Arc::new(Inner {
                    state,
                    apply: add.into_boxed_slice(),
                    copy: copy.into_boxed_slice(),
                }),
            })
        })
    }
}

type Rows<'d> = Vec<(Key, &'d Slot, u32)>;

pub(crate) fn move_to<'d, 'a, S>(
    mut state: S,
    database: &'d Database,
    source: &'a Table,
    target: &'a Table,
    rows: &mut Rows<'d>,
    copy: &[(usize, usize)],
    drop: &[usize],
    filter: impl FnOnce(&mut S, &mut Rows<'d>) -> (u32, u32),
    same: impl FnOnce(S, &mut Rows<'d>),
    initialize: impl FnOnce(S, &table::Inner, usize),
) {
    if rows.len() == 0 {
        return;
    }

    let indices = (source.index(), target.index());
    // If locks are always taken in order (lower index first), there can not be a deadlock between move operations.
    let (mut source, target, low, high, count) = if source.index() < target.index() {
        let source = source.inner.write();
        let (low, high) = filter(&mut state, rows);
        let Some(count) = NonZeroUsize::new(rows.len()) else {
            // Happens if all keys from this table have been moved or destroyed between here and the sorting.
            return;
        };
        let target = target.inner.upgradable_read();
        (source, target, low, high, count)
    } else if source.index() > target.index() {
        let target = target.inner.upgradable_read();
        let source = source.inner.write();
        let (low, high) = filter(&mut state, rows);
        let Some(count) = NonZeroUsize::new(rows.len()) else {
            // Happens if all keys from this table have been moved or destroyed between here and the sorting.
            return;
        };
        (source, target, low, high, count)
    } else {
        return same(state, rows);
    };

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

    initialize(state, &target, start);

    target.commit(count);
    // Slots must be updated after the table `commit` to prevent a `query::find` to be able to observe a row which
    // has an index greater than the `table.count()`. As long as the slots remain in the source table, all accesses
    // to these keys will block at the table access and will correct their table index after they acquire the source
    // table lock.
    for (i, (key, slot, ..)) in rows.drain(..).enumerate() {
        slot.initialize(key.generation(), indices.1, (start + i) as u32);
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
