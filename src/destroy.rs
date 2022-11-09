use crate::{
    core::utility::{fold_swap, get_unchecked, get_unchecked_mut, ONE},
    filter::Filter,
    key::{Key, Slot},
    table::{self, Column, Table},
    Database, Error,
};
use parking_lot::RwLockWriteGuard;
use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
    num::NonZeroUsize,
};

pub struct Destroy<'d, F: Filter = ()> {
    database: &'d Database,
    keys: HashSet<Key>,
    pending: Vec<(Key, &'d Slot, u32)>,
    sorted: HashMap<u32, Option<State<'d>>>,
    _marker: PhantomData<fn(F)>,
}

/// Destroys all keys in tables that satisfy the filter `F`.
pub struct DestroyAll<'d, F: Filter = ()> {
    database: &'d Database,
    index: usize,
    tables: Vec<&'d Table>,
    _marker: PhantomData<fn(F)>,
}

struct State<'d> {
    table: &'d Table,
    rows: Vec<(Key, &'d Slot, u32)>,
}

impl Database {
    pub fn destroy(&self) -> Destroy {
        Destroy {
            database: self,
            keys: HashSet::new(),
            pending: Vec::new(),
            sorted: HashMap::new(),
            _marker: PhantomData,
        }
    }

    pub fn destroy_all(&self) -> DestroyAll {
        DestroyAll {
            database: self,
            index: 0,
            tables: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl<'d, F: Filter> Destroy<'d, F> {
    #[inline]
    pub fn one(&mut self, key: Key) -> Result<(), Error> {
        let (slot, table) = self.database.keys().get(key)?;
        Self::sort(key, slot, table, &mut self.sorted, self.database)
    }

    #[inline]
    pub fn all<I: IntoIterator<Item = Key>>(&mut self, keys: I) -> usize {
        keys.into_iter()
            .filter_map(|key| self.one(key).ok())
            .count()
    }

    pub fn filter<G: Filter>(mut self) -> Destroy<'d, (F, G)> {
        self.sorted.retain(|_, state| match state {
            Some(state) => G::filter(state.table, self.database),
            None => true,
        });
        Destroy {
            database: self.database,
            keys: self.keys,
            pending: self.pending,
            sorted: self.sorted,
            _marker: PhantomData,
        }
    }

    pub fn resolve(&mut self) -> usize {
        self.resolve_sorted();
        while self.pending.len() > 0 {
            for (key, slot, table) in self.pending.drain(..) {
                let _ = Self::sort(key, slot, table, &mut self.sorted, self.database);
            }
            self.resolve_sorted();
        }
        let count = self.keys.len();
        // Note: Since releasing of keys is done here, there is a period of time where some keys are unavailable for reuse.
        self.database.keys.recycle(self.keys.drain());
        count
    }

    pub fn clear(&mut self) {
        debug_assert_eq!(self.keys.len(), 0);
        debug_assert_eq!(self.pending.len(), 0);
        for state in self.sorted.values_mut().flatten() {
            state.rows.clear();
        }
    }

    fn resolve_sorted(&mut self) {
        for state in self.sorted.values_mut().flatten() {
            let (mut inner, low, high) = Self::retain(
                state.table,
                &mut state.rows,
                &mut self.keys,
                &mut self.pending,
            );
            let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                continue;
            };
            debug_assert!(low <= high);

            let range = low..high + 1;
            let inner = &mut *inner;
            let head = inner.release(count);
            let keys = inner.keys.get_mut();
            let (low, high) = (range.start as usize, range.end as usize);

            if range.len() == count.get() {
                // The destroy range is contiguous.
                let over = high.saturating_sub(head);
                let end = count.get() - over;
                if let Some(end) = NonZeroUsize::new(end) {
                    // Squash the range at the end of the table on the beginning of the removed range.
                    let start = head + over;
                    squash(self.database, keys, &mut inner.columns, start, low, end);
                }

                if let Some(over) = NonZeroUsize::new(over) {
                    for column in inner.columns.iter_mut() {
                        unsafe { column.drop(head, over) };
                    }
                }
            } else {
                // Tag keys that are going to be removed such that removed keys and valid keys can be differentiated.
                for &(.., row) in state.rows.iter() {
                    *unsafe { get_unchecked_mut(keys, row as usize) } = Key::NULL;
                }

                let mut index = 0;
                let mut cursor = head;
                while let Some(&(.., row)) = state.rows.get(index) {
                    let row = row as usize;
                    let mut previous = row;
                    let start = index;
                    index += 1;

                    if row < head {
                        // Find the next valid row to move.
                        while unsafe { *get_unchecked(keys, cursor) } == Key::NULL {
                            cursor += 1;
                        }
                        debug_assert!(cursor < head + count.get());
                        let one = ONE;
                        squash(self.database, keys, &mut inner.columns, cursor, row, one);
                        cursor += 1;
                    } else {
                        // Try to batch contiguous drops.
                        while let Some(&(.., current)) = state.rows.get(index) {
                            let current = current as usize;
                            if previous + 1 == current {
                                previous = current;
                                index += 1;
                            } else {
                                break;
                            }
                        }

                        let count = unsafe { NonZeroUsize::new_unchecked(index - start) };
                        for column in inner.columns.iter_mut() {
                            unsafe { column.drop(row, count) };
                        }
                    }
                }
            }

            for &(key, slot, row) in state.rows.iter() {
                debug_assert_eq!(slot.table(key.generation()), Ok(state.table.index()));
                debug_assert_eq!(slot.row(), row);
                slot.release();
            }
        }
    }

    fn sort(
        key: Key,
        slot: &'d Slot,
        table: u32,
        sorted: &mut HashMap<u32, Option<State<'d>>>,
        database: &'d Database,
    ) -> Result<(), Error> {
        match sorted.get_mut(&table) {
            Some(state) => match state {
                Some(state) => Ok(state.rows.push((key, slot, u32::MAX))),
                None => Err(Error::FilterDoesNotMatch),
            },
            None => {
                let table = unsafe { database.tables().get_unchecked(table as _) };
                let state = if F::filter(table, database) {
                    let rows = vec![(key, slot, u32::MAX)];
                    Some(State { table, rows })
                } else {
                    None
                };
                sorted.insert(table.index(), state);
                Ok(())
            }
        }
    }

    fn retain<'a>(
        table: &'a Table,
        rows: &mut Vec<(Key, &'d Slot, u32)>,
        keys: &mut HashSet<Key>,
        pending: &mut Vec<(Key, &'d Slot, u32)>,
    ) -> (RwLockWriteGuard<'a, table::Inner>, u32, u32) {
        let mut low = u32::MAX;
        let mut high = 0;
        let mut index = 0;
        let inner = table.inner.write();
        while let Some((key, slot, row)) = rows.get_mut(index) {
            if let Ok(table_index) = slot.table(key.generation()) {
                if table_index == table.index() {
                    // Duplicates must only be checked here where the key would be guaranteed to be destroyed.
                    // - This way `database.keys().release()` can be called with `keys.drain()` at the end of `resolve`.
                    if keys.insert(*key) {
                        *row = slot.row();
                        low = low.min(*row);
                        high = high.max(*row);
                        index += 1;
                    } else {
                        rows.swap_remove(index);
                    }
                } else {
                    let (key, slot, _) = rows.swap_remove(index);
                    pending.push((key, slot, table_index));
                }
            } else {
                rows.swap_remove(index);
            }
        }
        (inner, low, high)
    }
}

impl<'d, F: Filter> DestroyAll<'d, F> {
    pub fn filter<G: Filter>(mut self) -> DestroyAll<'d, (F, G)> {
        self.tables.retain(|table| G::filter(table, self.database));
        DestroyAll {
            database: self.database,
            index: self.index,
            tables: self.tables,
            _marker: PhantomData,
        }
    }

    pub fn resolve(&mut self) -> usize {
        while let Ok(table) = self.database.tables().get(self.index) {
            self.index += 1;
            if F::filter(table, self.database) {
                self.tables.push(table);
            }
        }

        fold_swap(
            &mut self.tables,
            0,
            (),
            |sum, _, table| {
                Ok(sum + Self::resolve_table(table.inner.try_write().ok_or(sum)?, self.database))
            },
            |sum, _, table| sum + Self::resolve_table(table.inner.write(), self.database),
        )
    }

    fn resolve_table(mut inner: RwLockWriteGuard<'d, table::Inner>, database: &Database) -> usize {
        let Some(count) = NonZeroUsize::new(*inner.count.get_mut() as _) else {
            return 0;
        };
        inner.release(count);

        for column in inner.columns.iter_mut() {
            unsafe { column.drop(0, count) };
        }
        let keys = inner.keys.get_mut();
        database.keys().release(&keys[..count.get()]);
        return count.get();
    }
}

#[inline]
fn squash(
    database: &Database,
    keys: &mut [Key],
    columns: &mut [Column],
    source: usize,
    target: usize,
    count: NonZeroUsize,
) {
    for column in columns {
        unsafe { column.squash(source, target, count) };
    }

    // Update the keys.
    keys.copy_within(source..source + count.get(), target);
    database.keys().update(keys, target..target + count.get());
}
