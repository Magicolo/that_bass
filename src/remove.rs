use parking_lot::{RwLockUpgradableReadGuard, RwLockWriteGuard};

use crate::{
    add::{copy_to, move_to},
    core::utility::{fold_swap, get_unchecked_mut},
    filter::Filter,
    key::{Key, Slot},
    table::{self, Table},
    template::{ShareMeta, Template},
    Database, Error,
};
use std::{collections::HashSet, marker::PhantomData, num::NonZeroUsize, sync::Arc};

pub struct Remove<'d, T: Template, F: Filter = ()> {
    database: &'d Database,
    keys: HashSet<Key>, // A `HashSet` is used because the move algorithm assumes that rows will be unique.
    indices: Vec<usize>, // May be reordered (ex: by `fold_swap`).
    states: Vec<Result<State<'d>, u32>>, // Must remain sorted by `state.source.index()` for `binary_search` to work.
    pending: Vec<(Key, &'d Slot, u32)>,
    filter: F,
    _marker: PhantomData<fn(T)>,
}

/// Removes template `T` to all keys in tables that satisfy the filter `F`.
pub struct RemoveAll<'d, T: Template, F: Filter = ()> {
    database: &'d Database,
    index: usize,
    states: Vec<StateAll>,
    filter: F,
    _marker: PhantomData<fn(T)>,
}

struct State<'d> {
    source: Arc<Table>,
    target: Arc<Table>,
    inner: Arc<Inner>,
    rows: Vec<(Key, &'d Slot, u32)>,
}

struct StateAll {
    source: Arc<Table>,
    target: Arc<Table>,
    inner: Arc<Inner>,
}

struct Inner {
    copy: Box<[(usize, usize)]>,
    drop: Box<[usize]>,
}

struct ShareTable<T> {
    source: Arc<Table>,
    target: Arc<Table>,
    inner: Arc<Inner>,
    _marker: PhantomData<fn(T)>,
}

impl Database {
    pub fn remove<T: Template>(&self) -> Result<Remove<T>, Error> {
        // Validate metas here, but there is no need to store them.
        ShareMeta::<T>::from(self).map(|_| Remove {
            database: self,
            keys: HashSet::new(),
            pending: Vec::new(),
            states: Vec::new(),
            indices: Vec::new(),
            filter: (),
            _marker: PhantomData,
        })
    }

    pub fn remove_all<T: Template>(&self) -> Result<RemoveAll<T>, Error> {
        // Validate metas here, but there is no need to store them.
        ShareMeta::<T>::from(self).map(|_| RemoveAll {
            database: self,
            index: 0,
            states: Vec::new(),
            filter: (),
            _marker: PhantomData,
        })
    }
}

impl<'d, T: Template, F: Filter> Remove<'d, T, F> {
    // Note: `one` and `all` methods should minimize the amount of work they do since they are meant to be called inside a query,
    // thus while potentially holding many locks. Therefore, all validation is moved to `resolve` instead.
    #[inline]
    pub fn one(&mut self, key: Key) {
        self.keys.insert(key);
    }

    #[inline]
    pub fn all(&mut self, keys: impl IntoIterator<Item = Key>) {
        self.keys.extend(keys);
    }

    pub fn filter<G: Filter + Default>(self) -> Remove<'d, T, (F, G)> {
        self.filter_with(G::default())
    }

    pub fn filter_with<G: Filter>(mut self, filter: G) -> Remove<'d, T, (F, G)> {
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
        Remove {
            database: self.database,
            keys: self.keys,
            pending: self.pending,
            filter: self.filter.and(filter),
            indices: self.indices,
            states: self.states,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = Key> + '_ {
        self.keys.iter().copied()
    }

    pub fn drain(&mut self) -> impl ExactSizeIterator<Item = Key> + '_ {
        debug_assert_eq!(self.pending.len(), 0);
        debug_assert_eq!(self.indices.len(), 0);
        self.keys.drain()
    }

    pub fn clear(&mut self) {
        debug_assert_eq!(self.pending.len(), 0);
        debug_assert_eq!(self.indices.len(), 0);
        self.keys.clear();
    }

    pub fn resolve(&mut self) -> usize {
        for (key, result) in self.database.keys().get_all(self.keys.drain()) {
            if let Ok((slot, table)) = result {
                Self::sort(
                    key,
                    slot,
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
            for (key, slot, table) in self.pending.drain(..) {
                Self::sort(
                    key,
                    slot,
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
                debug_assert_ne!(state.source.index(), state.target.index());
                if state.rows.len() == 0 {
                    return Ok(sum);
                }
                let source = state.source.inner.try_write().ok_or(sum)?;
                let target = state.target.inner.try_upgradable_read().ok_or(sum)?;
                let (low, high) = Self::retain(&state.source, &mut state.rows, pending);
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
                    &state.inner.drop,
                    |_, _, _| {},
                );
                Ok(sum + count.get())
            },
            |sum, (states, pending), index| {
                let Some(Ok(state)) = states.get_mut(*index) else {
                    unreachable!();
                };
                debug_assert_ne!(state.source.index(), state.target.index());
                let (source, target, low, high, count) =
                    // If locks are always taken in order (lower index first), there can not be a deadlock between move operations.
                    if state.source.index() < state.target.index() {
                        let source = state.source.inner.write();
                        let (low, high) = Self::retain(&state.source, &mut state.rows, pending);
                        let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                            // Happens if all keys from this table have been moved or destroyed between here and the sorting.
                            return sum;
                        };
                        let target = state.target.inner.upgradable_read();
                        (source, target, low, high, count)
                    } else {
                        let target = state.target.inner.upgradable_read();
                        let source = state.source.inner.write();
                        let (low, high) = Self::retain(&state.source, &mut state.rows, pending);
                        let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                            // Happens if all keys from this table have been moved or destroyed between here and the sorting.
                            return sum ;
                        };
                        (source, target, low, high, count)
                    };
                move_to(
                    self.database,
                    source,
                    (state.target.index(), target),
                    (low, high, count),
                    &mut state.rows,
                    &state.inner.copy,
                    &state.inner.drop,
                    |_, _, _| {},
                );
                sum + count.get()
            },
        )
    }

    fn sort(
        key: Key,
        slot: &'d Slot,
        table: u32,
        indices: &mut Vec<usize>,
        states: &mut Vec<Result<State<'d>, u32>>,
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
        }
    }

    /// Call this while holding a lock on `table`.
    fn retain<'a>(
        table: &'a Table,
        rows: &mut Vec<(Key, &'d Slot, u32)>,
        pending: &mut Vec<(Key, &'d Slot, u32)>,
    ) -> (u32, u32) {
        let mut low = u32::MAX;
        let mut high = 0;
        for i in (0..rows.len()).rev() {
            let (key, slot, row) = unsafe { get_unchecked_mut(rows, i) };
            if let Ok(new_table) = slot.table(key.generation()) {
                if new_table == table.index() {
                    *row = slot.row();
                    low = low.min(*row);
                    high = high.max(*row);
                } else {
                    let (key, slot, _) = rows.swap_remove(i);
                    pending.push((key, slot, new_table));
                }
            } else {
                rows.swap_remove(i);
            }
        }
        debug_assert_eq!(low <= high, rows.len() > 0);
        (low, high)
    }
}

impl<'d, T: Template, F: Filter> RemoveAll<'d, T, F> {
    pub fn filter<G: Filter + Default>(self) -> RemoveAll<'d, T, (F, G)> {
        self.filter_with(G::default())
    }

    pub fn filter_with<G: Filter>(mut self, filter: G) -> RemoveAll<'d, T, (F, G)> {
        self.states
            .retain(|state| filter.filter(&state.source, self.database));
        RemoveAll {
            database: self.database,
            index: self.index,
            states: self.states,
            filter: self.filter.and(filter),
            _marker: PhantomData,
        }
    }

    pub fn resolve(&mut self) -> usize {
        while let Ok(table) = self.database.tables().get(self.index) {
            self.index += 1;
            match ShareTable::<T>::from(table.index(), self.database) {
                Ok((source, target, inner)) if self.filter.filter(table, self.database) => {
                    self.states.push(StateAll {
                        source,
                        target,
                        inner,
                    })
                }
                _ => {}
            }
        }

        fold_swap(
            &mut self.states,
            0,
            (),
            |sum, _, state| {
                let count = if state.source.index() < state.target.index() {
                    let source = state.source.inner.try_write().ok_or(sum)?;
                    let target = state.target.inner.try_upgradable_read().ok_or(sum)?;
                    Self::resolve_tables(source, target, state, self.database)
                } else if state.source.index() > state.target.index() {
                    let target = state.target.inner.try_upgradable_read().ok_or(sum)?;
                    let source = state.source.inner.try_write().ok_or(sum)?;
                    Self::resolve_tables(source, target, state, self.database)
                } else {
                    unreachable!("Same source and target is supposed to be filtered.")
                };
                Ok(sum + count)
            },
            |sum, _, state| {
                sum + if state.source.index() < state.target.index() {
                    let source = state.source.inner.write();
                    let target = state.target.inner.upgradable_read();
                    Self::resolve_tables(source, target, state, self.database)
                } else if state.source.index() > state.target.index() {
                    let target = state.target.inner.upgradable_read();
                    let source = state.source.inner.write();
                    Self::resolve_tables(source, target, state, self.database)
                } else {
                    unreachable!("Same source and target is supposed to be filtered.")
                }
            },
        )
    }

    fn resolve_tables(
        mut source: RwLockWriteGuard<table::Inner>,
        target: RwLockUpgradableReadGuard<table::Inner>,
        state: &StateAll,
        database: &Database,
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
            &state.inner.drop,
        );

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
}

impl<T: Template> ShareTable<T> {
    pub fn from(
        table: u32,
        database: &Database,
    ) -> Result<(Arc<Table>, Arc<Table>, Arc<Inner>), Error> {
        let share = database.resources().try_global_with(table, || {
            let metas = ShareMeta::<T>::from(database)?;
            let source = database.tables().get_shared(table as usize)?;
            let target = {
                let mut targets = Vec::new();
                for &meta in source.metas() {
                    match metas.binary_search_by_key(&meta.identifier(), |meta| meta.identifier()) {
                        Ok(_) => {}
                        Err(_) => targets.push(meta),
                    }
                }
                database.tables().find_or_add(&targets)
            };
            if source.index() == target.index() {
                return Err(Error::TablesMustDiffer);
            }

            let mut copy = Vec::new();
            for target in target.metas().iter().enumerate() {
                let source = source.column_with(target.1.identifier())?;
                debug_assert_eq!(source.1.identifier(), target.1.identifier());
                if source.1.size > 0 {
                    copy.push((source.0, target.0));
                }
            }

            let mut drop = Vec::new();
            for meta in metas.iter() {
                if let Ok((index, meta)) = source.column_with(meta.identifier()) {
                    if meta.drop.0() {
                        drop.push(index);
                    }
                }
            }

            Ok(Self {
                source,
                target,
                inner: Arc::new(Inner {
                    copy: copy.into_boxed_slice(),
                    drop: drop.into_boxed_slice(),
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
