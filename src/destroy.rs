use crate::{
    core::utility::{fold_swap, get_unchecked, get_unchecked_mut, ONE},
    filter::Filter,
    key::{Key, Slot},
    table::{self, Column, Table},
    Database,
};
use parking_lot::RwLockWriteGuard;
use std::{collections::HashSet, num::NonZeroUsize};

pub struct Destroy<'d, F: Filter = ()> {
    database: &'d Database,
    keys: HashSet<Key>, // A `HashSet` is used because the move algorithm assumes that rows will be unique.
    indices: Vec<usize>, // May be reordered (ex: by `fold_swap`).
    states: Vec<Result<State<'d>, u32>>, // Must remain sorted by `state.table.index()` for `binary_search` to work.
    pending: Vec<(Key, &'d Slot, u32)>,
    filter: F,
}

/// Destroys all keys in tables that satisfy the filter `F`.
pub struct DestroyAll<'d, F: Filter = ()> {
    database: &'d Database,
    index: usize,
    tables: Vec<&'d Table>,
    filter: F,
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
            states: Vec::new(),
            indices: Vec::new(),
            filter: (),
        }
    }

    pub fn destroy_all(&self) -> DestroyAll {
        DestroyAll {
            database: self,
            index: 0,
            tables: Vec::new(),
            filter: (),
        }
    }
}

impl<'d, F: Filter> Destroy<'d, F> {
    #[inline]
    pub fn one(&mut self, key: Key) {
        self.keys.insert(key);
    }

    #[inline]
    pub fn all<I: IntoIterator<Item = Key>>(&mut self, keys: I) {
        self.keys.extend(keys);
    }

    pub fn filter<G: Filter + Default>(self) -> Destroy<'d, (F, G)> {
        self.filter_with(G::default())
    }

    pub fn filter_with<G: Filter>(mut self, filter: G) -> Destroy<'d, (F, G)> {
        for state in self.states.iter_mut() {
            let index = match state {
                Ok(state) if filter.filter(&state.table, self.database) => None,
                Ok(state) => Some(state.table.index()),
                Err(_) => None,
            };
            if let Some(index) = index {
                *state = Err(index);
            }
        }
        Destroy {
            database: self.database,
            keys: self.keys,
            pending: self.pending,
            states: self.states,
            indices: self.indices,
            filter: self.filter.and(filter),
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
                if state.rows.len() == 0 {
                    return Ok(sum);
                }
                let inner = state.table.inner.try_write().ok_or(sum)?;
                let (low, high) = Self::retain(state.table, &mut state.rows, pending);
                let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                    return Ok(sum);
                };
                Self::resolve_rows(state, inner, (low, high, count), self.database);
                Ok(sum + count.get())
            },
            |sum, (states, pending), index| {
                let Some(Ok(state)) = states.get_mut(*index) else {
                    unreachable!();
                };
                if state.rows.len() == 0 {
                    return sum;
                }
                let inner = state.table.inner.write();
                let (low, high) = Self::retain(state.table, &mut state.rows, pending);
                let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                    return sum;
                };
                Self::resolve_rows(state, inner, (low, high, count), self.database);
                sum + count.get()
            },
        )
    }

    fn resolve_rows(
        state: &mut State<'d>,
        mut inner: RwLockWriteGuard<'d, table::Inner>,
        (low, high, count): (u32, u32, NonZeroUsize),
        database: &'d Database,
    ) {
        debug_assert!(low <= high);

        let range = low..high + 1;
        let head = inner.release(count);
        let (low, high) = (range.start as usize, range.end as usize);

        if range.len() == count.get() {
            // The destroy range is contiguous.
            let inner = &mut *inner;
            let keys = inner.keys.get_mut();
            let over = high.saturating_sub(head);
            let end = count.get() - over;
            if let Some(end) = NonZeroUsize::new(end) {
                // Squash the range at the end of the table on the beginning of the removed range.
                let start = head + over;
                squash(database, keys, &mut inner.columns, start, low, end);
            }

            if let Some(over) = NonZeroUsize::new(over) {
                for column in inner.columns.iter_mut() {
                    unsafe { column.drop(head, over) };
                }
            }
        } else {
            let inner = &mut *inner;
            let keys = inner.keys.get_mut();

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
                    squash(database, keys, &mut inner.columns, cursor, row, one);
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
            debug_assert_eq!(slot.table(key), Ok(state.table.index()));
            debug_assert_eq!(slot.row(), row);
            slot.release();
        }
        drop(inner);
        // The `recycle` step can be done outside of the lock. This means that the keys within `state.rows` may be very briefly
        // non-reusable for other threads, which is fine.
        database
            .keys
            .recycle(state.rows.drain(..).map(|(key, ..)| key));
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
            Ok(state) => state.table.index(),
            Err(index) => *index,
        }) {
            Ok(index) => index,
            Err(index) => {
                let table = unsafe { database.tables().get_unchecked(table as _) };
                let result = if filter.filter(table, database) {
                    Ok(State {
                        table,
                        rows: Vec::new(),
                    })
                } else {
                    Err(table.index())
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
            if let Ok(table_index) = slot.table(*key) {
                if table_index == table.index() {
                    *row = slot.row();
                    low = low.min(*row);
                    high = high.max(*row);
                } else {
                    let (key, slot, _) = rows.swap_remove(i);
                    pending.push((key, slot, table_index));
                }
            } else {
                rows.swap_remove(i);
            }
        }
        debug_assert_eq!(low <= high, rows.len() > 0);
        (low, high)
    }
}

impl<'d, F: Filter> DestroyAll<'d, F> {
    pub fn filter<G: Filter + Default>(self) -> DestroyAll<'d, (F, G)> {
        self.filter_with(G::default())
    }

    pub fn filter_with<G: Filter>(mut self, filter: G) -> DestroyAll<'d, (F, G)> {
        self.tables
            .retain(|table| filter.filter(table, self.database));
        DestroyAll {
            database: self.database,
            index: self.index,
            tables: self.tables,
            filter: self.filter.and(filter),
        }
    }

    pub fn resolve(&mut self) -> usize {
        while let Ok(table) = self.database.tables().get(self.index) {
            self.index += 1;
            if self.filter.filter(table, self.database) {
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
