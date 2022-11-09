use parking_lot::{RwLockUpgradableReadGuard, RwLockWriteGuard};

use crate::{
    add::{copy_to, move_to},
    core::utility::{fold_swap, get_unchecked_mut},
    filter::Filter,
    key::{Key, Slot},
    resources::Global,
    table::{self, Table},
    template::{ShareMeta, Template},
    Database, Error,
};
use std::{collections::HashMap, marker::PhantomData, num::NonZeroUsize, sync::Arc};

pub struct Remove<'d, T: Template, F: Filter = ()> {
    database: &'d Database,
    keys: HashMap<Key, u32>,
    pending: Vec<(Key, &'d Slot, u32)>,
    sorted: HashMap<u32, Option<State<'d>>>,
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
            keys: HashMap::new(),
            pending: Vec::new(),
            sorted: HashMap::new(),
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
    #[inline]
    pub fn one(&mut self, key: Key) -> Result<(), Error> {
        let (slot, table) = self.database.keys().get(key)?;
        Self::sort(
            key,
            slot,
            table,
            &mut self.sorted,
            &self.filter,
            self.database,
        )
    }

    #[inline]
    pub fn all(&mut self, keys: impl IntoIterator<Item = Key>) -> usize {
        keys.into_iter()
            .filter_map(|key| self.one(key).ok())
            .count()
    }

    pub fn filter<G: Filter>(mut self, filter: G) -> Remove<'d, T, (F, G)> {
        debug_assert_eq!(self.keys.len(), 0);
        debug_assert_eq!(self.pending.len(), 0);
        self.sorted.retain(|_, state| match state {
            Some(state) => filter.filter(&state.source, self.database),
            None => true,
        });
        Remove {
            database: self.database,
            keys: self.keys,
            pending: self.pending,
            sorted: self.sorted,
            filter: self.filter.and(filter),
            _marker: PhantomData,
        }
    }

    pub fn clear(&mut self) {
        debug_assert_eq!(self.keys.len(), 0);
        debug_assert_eq!(self.pending.len(), 0);
        for state in self.sorted.values_mut().flatten() {
            state.rows.clear();
        }
    }

    pub fn resolve(&mut self) -> usize {
        loop {
            self.resolve_sorted();
            if self.pending.len() == 0 {
                break;
            }

            for (key, slot, table) in self.pending.drain(..) {
                let _ = Self::sort(
                    key,
                    slot,
                    table,
                    &mut self.sorted,
                    &self.filter,
                    self.database,
                );
            }
        }
        let count = self.keys.len();
        self.keys.clear();
        count
    }

    fn resolve_sorted(&mut self) {
        for state in self.sorted.values_mut().flatten() {
            move_to(
                (),
                self.database,
                &state.source,
                &state.target,
                &mut state.rows,
                &state.inner.copy,
                &state.inner.drop,
                |_, rows| Self::retain(&state.source, &mut self.keys, rows, &mut self.pending),
                |_, rows| rows.clear(),
                |_, _, _| {},
            );

            // Sanity checks.
            debug_assert!(state.rows.is_empty());
        }
    }

    fn sort(
        key: Key,
        slot: &'d Slot,
        table: u32,
        sorted: &mut HashMap<u32, Option<State<'d>>>,
        filter: &F,
        database: &'d Database,
    ) -> Result<(), Error> {
        match sorted.get_mut(&table) {
            Some(state) => match state {
                Some(state) => Ok(state.rows.push((key, slot, u32::MAX))),
                None => Err(Error::FilterDoesNotMatch),
            },
            None => {
                let share = ShareTable::<T>::from(table, database)?;
                let share = share.read();
                let state = if filter.filter(&share.source, database) {
                    Some(State {
                        source: share.source.clone(),
                        target: share.target.clone(),
                        inner: share.inner.clone(),
                        rows: vec![(key, slot, u32::MAX)],
                    })
                } else {
                    None
                };
                sorted.insert(table, state);
                Ok(())
            }
        }
    }

    /// Call this while holding a lock on `table`.
    fn retain<'a>(
        table: &'a Table,
        keys: &mut HashMap<Key, u32>,
        rows: &mut Vec<(Key, &'d Slot, u32)>,
        pending: &mut Vec<(Key, &'d Slot, u32)>,
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
                    pending.push((key, slot, table_index));
                }
            } else {
                rows.swap_remove(index);
            }
        }

        debug_assert_eq!(low <= high, rows.len() > 0);
        (low, high)
    }
}

impl<'d, T: Template, F: Filter> RemoveAll<'d, T, F> {
    pub fn filter<G: Filter>(mut self, filter: G) -> RemoveAll<'d, T, (F, G)> {
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
            if self.filter.filter(table, self.database) {
                if let Ok(share) = ShareTable::<T>::from(table.index(), self.database) {
                    let share = share.read();
                    if share.source.index() != share.target.index() {
                        self.states.push(StateAll {
                            source: share.source.clone(),
                            target: share.target.clone(),
                            inner: share.inner.clone(),
                        });
                    }
                }
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
    pub fn from(table: u32, database: &Database) -> Result<Global<Self>, Error> {
        database.resources().try_global_with(table, || {
            let metas = ShareMeta::<T>::from(database)?;
            let source = database.tables().get_shared(table as usize)?;
            let mut target_metas = Vec::new();
            for column in source.inner.read().columns() {
                if metas.contains(&column.meta()) {
                    continue;
                } else {
                    target_metas.push(column.meta());
                }
            }
            let target = database.tables().find_or_add(target_metas);

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
        })
    }
}
