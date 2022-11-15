use crate::{
    core::utility::{fold_swap, get_unchecked, get_unchecked_mut, swap_unchecked, ONE},
    event::Listen,
    filter::Filter,
    key::{Key, Slot},
    table::Table,
    Database,
};
use parking_lot::{RwLockUpgradableReadGuard, RwLockWriteGuard};
use std::{collections::HashSet, num::NonZeroUsize, sync::atomic::Ordering};

pub struct Destroy<'d, F, L> {
    database: &'d Database<L>,
    keys: HashSet<Key>, // A `HashSet` is used because the move algorithm assumes that rows will be unique.
    indices: Vec<usize>, // May be reordered (ex: by `fold_swap`).
    states: Vec<Result<State<'d>, u32>>, // Must remain sorted by `state.table.index()` for `binary_search` to work.
    pending: Vec<(Key, &'d Slot, u32)>,
    moves: Vec<(usize, usize, NonZeroUsize)>,
    drops: Vec<(usize, NonZeroUsize)>,
    filter: F,
}

/// Destroys all keys in tables that satisfy the filter `F`.
pub struct DestroyAll<'d, F = (), L = ()> {
    database: &'d Database<L>,
    index: usize,
    tables: Vec<&'d Table>,
    filter: F,
}

type Rows<'d> = Vec<(Key, &'d Slot, u32)>;

struct State<'d> {
    table: &'d Table,
    rows: Rows<'d>,
}

impl<L> Database<L> {
    pub fn destroy(&self) -> Destroy<'_, (), L> {
        Destroy {
            database: self,
            keys: HashSet::new(),
            pending: Vec::new(),
            states: Vec::new(),
            indices: Vec::new(),
            moves: Vec::new(),
            drops: Vec::new(),
            filter: (),
        }
    }

    pub fn destroy_all(&self) -> DestroyAll<'_, (), L> {
        DestroyAll {
            database: self,
            index: 0,
            tables: Vec::new(),
            filter: (),
        }
    }
}

impl<'d, F, L> Destroy<'d, F, L> {
    #[inline]
    pub fn one(&mut self, key: Key) {
        self.keys.insert(key);
    }

    #[inline]
    pub fn all<I: IntoIterator<Item = Key>>(&mut self, keys: I) {
        self.keys.extend(keys);
    }

    pub fn filter<G: Filter + Default>(self) -> Destroy<'d, (F, G), L> {
        self.filter_with(G::default())
    }

    pub fn filter_with<G: Filter>(mut self, filter: G) -> Destroy<'d, (F, G), L> {
        for state in self.states.iter_mut() {
            let index = match state {
                Ok(state) if filter.filter(&state.table, self.database.into()) => None,
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
            moves: self.moves,
            drops: self.drops,
            filter: (self.filter, filter),
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
}

impl<'d, F: Filter, L: Listen> Destroy<'d, F, L> {
    pub fn resolve(&mut self) -> usize {
        for (key, result) in self.database.keys().get_all(self.keys.iter().copied()) {
            if let Ok((slot, table)) = result {
                Self::sort(
                    key,
                    slot,
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

            for (key, slot, table) in self.pending.drain(..) {
                Self::sort(
                    key,
                    slot,
                    table,
                    &mut self.indices,
                    &mut self.states,
                    &self.filter,
                    &self.database.inner,
                );
            }
        }
        self.keys.clear();
        debug_assert_eq!(self.keys.len(), 0);
        debug_assert_eq!(self.moves.len(), 0);
        debug_assert_eq!(self.drops.len(), 0);
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
                &mut self.drops,
            ),
            |sum, (states, pending, moves, drops), index| {
                let result = unsafe { get_unchecked_mut(states, *index) };
                let state = unsafe { result.as_mut().unwrap_unchecked() };
                let keys = state.table.keys.try_upgradable_read().ok_or(sum)?;
                let (low, high) = Self::retain(state.table, &mut state.rows, pending);
                let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                    return Ok(sum);
                };
                Self::resolve_rows(
                    state,
                    &self.keys,
                    moves,
                    drops,
                    keys,
                    (low, high, count),
                    self.database,
                );
                Ok(sum + count.get())
            },
            |sum, (states, pending, moves, drops), index| {
                let result = unsafe { get_unchecked_mut(states, *index) };
                let state = unsafe { result.as_mut().unwrap_unchecked() };
                let keys = state.table.keys.upgradable_read();
                let (low, high) = Self::retain(state.table, &mut state.rows, pending);
                let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                    return sum;
                };
                Self::resolve_rows(
                    state,
                    &self.keys,
                    moves,
                    drops,
                    keys,
                    (low, high, count),
                    self.database,
                );
                sum + count.get()
            },
        )
    }

    fn resolve_rows(
        state: &mut State<'d>,
        set: &HashSet<Key>,
        moves: &mut Vec<(usize, usize, NonZeroUsize)>,
        drops: &mut Vec<(usize, NonZeroUsize)>,
        keys: RwLockUpgradableReadGuard<'d, Vec<Key>>,
        (low, high, count): (u32, u32, NonZeroUsize),
        database: &'d Database<impl Listen>,
    ) {
        debug_assert_eq!(moves.len(), 0);
        debug_assert_eq!(drops.len(), 0);
        debug_assert!(low <= high);

        let range = low..high + 1;
        let head = state.table.count.load(Ordering::Acquire) - count.get();
        let (low, high) = (range.start as usize, range.end as usize);
        if range.len() == count.get() {
            // The destroy range is contiguous.
            let over = high.saturating_sub(head);
            if let Some(end) = NonZeroUsize::new(count.get() - over) {
                // Move the range at the end of the table on the beginning of the removed range.
                moves.push((head + over, low, end));
            }
            if let Some(over) = NonZeroUsize::new(over) {
                drops.push((head, over));
            }
        } else {
            let mut index = 0;
            let mut cursor = head;
            while let Some(&(.., row)) = state.rows.get(index) {
                let row = row as usize;
                let mut previous = row;
                let start = index;
                index += 1;

                if row < head {
                    // Find the next valid row to move.
                    while set.contains(unsafe { get_unchecked(&keys, cursor) }) {
                        cursor += 1;
                    }
                    debug_assert!(cursor < head + count.get());
                    moves.push((cursor, row, ONE));
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
                    drops.push((row, unsafe { NonZeroUsize::new_unchecked(index - start) }));
                }
            }
        }

        let mut keys = RwLockUpgradableReadGuard::upgrade(keys);
        for &(source, target, count) in moves.iter() {
            for i in 0..count.get() {
                // Swap is used such that destroyed keys are gathered at the end of the table when the operation is complete.
                // - This allows `on_destroy` to use this fact.
                unsafe { swap_unchecked(&mut keys, source + i, target + i) };
            }
            database.keys().update(&keys, target..target + count.get());
        }
        for column in state.table.columns() {
            if column.meta().size > 0 {
                for &(source, target, count) in moves.iter() {
                    // Since a write lock is held on `keys`, no need to take a column lock.
                    unsafe { column.squash(source, target, count) };
                }
            }
        }
        // Must be decremented under the write lock to ensure that no query can observe the `table.count` with the invalid rows
        // at the end.
        state
            .table
            .count
            .fetch_sub(count.get() as _, Ordering::Release);

        // Keep an upgradable lock to prevent `create/add` operations to resolve during these last steps.
        let keys = RwLockWriteGuard::downgrade_to_upgradable(keys);
        // It is ok to release the slots under the upgradable lock (vs under the write lock) over `keys` because queries does an
        // additionnal validation (`keys.get(row) == Some(&key)`) which prevents the case where a key would be thought to be in the
        // query while going through `database.keys().get()` and yet its `slot.row` points to the wrong row.
        // - Note that it is possible to observe an inconsistency between `query.has` and `query.find` for a short time since
        // `query.has` does not do the additionnal validation.
        for &(key, slot, row) in state.rows.iter() {
            debug_assert_eq!(slot.table(key), Ok(state.table.index()));
            debug_assert_eq!(slot.row(), row);
            slot.release();
        }
        for column in state.table.columns() {
            if column.meta().drop.0() {
                for &(index, count) in drops.iter() {
                    // Since the `table.count` has been decremented under the `keys` write lock was held, these indices are only observable
                    // by this thread (assuming the indices are `> head`).
                    debug_assert!(index >= head);
                    unsafe { column.drop(index, count) };
                }
            }
        }

        database
            .listen
            .on_destroy(&keys[head..head + count.get()], state.table);
        drop(keys);
        // The `recycle` step can be done outside of the lock. This means that the keys within `state.rows` may be very briefly
        // non-reusable for other threads, which is fine.
        database
            .keys()
            .recycle(state.rows.drain(..).map(|(key, ..)| key));
        moves.clear();
        drops.clear();
    }

    fn sort(
        key: Key,
        slot: &'d Slot,
        table: u32,
        indices: &mut Vec<usize>,
        states: &mut Vec<Result<State<'d>, u32>>,
        filter: &F,
        database: &'d crate::Inner,
    ) {
        let index = match states.binary_search_by_key(&table, |result| match result {
            Ok(state) => state.table.index(),
            Err(index) => *index,
        }) {
            Ok(index) => index,
            Err(index) => {
                let table = unsafe { database.tables.get_unchecked(table as _) };
                let result = if filter.filter(table, database.into()) {
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
        rows: &mut Rows<'d>,
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

impl<'d, F, L> DestroyAll<'d, F, L> {
    pub fn filter<G: Filter + Default>(self) -> DestroyAll<'d, (F, G), L> {
        self.filter_with(G::default())
    }

    pub fn filter_with<G: Filter>(mut self, filter: G) -> DestroyAll<'d, (F, G), L> {
        self.tables
            .retain(|table| filter.filter(table, self.database.into()));
        DestroyAll {
            database: self.database,
            index: self.index,
            tables: self.tables,
            filter: (self.filter, filter),
        }
    }
}

impl<'d, F: Filter, L: Listen> DestroyAll<'d, F, L> {
    pub fn resolve(&mut self) -> usize {
        while let Ok(table) = self.database.tables().get(self.index) {
            self.index += 1;
            if self.filter.filter(table, self.database.into()) {
                self.tables.push(table);
            }
        }

        fold_swap(
            &mut self.tables,
            0,
            (),
            |sum, _, table| {
                let keys = table.keys.try_upgradable_read().ok_or(sum)?;
                Ok(sum + Self::resolve_table(keys, table, self.database))
            },
            |sum, _, table| {
                let keys = table.keys.upgradable_read();
                sum + Self::resolve_table(keys, table, self.database)
            },
        )
    }

    fn resolve_table(
        keys: RwLockUpgradableReadGuard<Vec<Key>>,
        table: &Table,
        database: &Database<impl Listen>,
    ) -> usize {
        let count = table.count.swap(0, Ordering::AcqRel);
        let Some(count) = NonZeroUsize::new(count) else {
            return 0;
        };
        for column in table.columns() {
            if column.meta().drop.0() {
                let write = column.data().write();
                unsafe { column.drop(0, count) };
                // No need to accumulate locks since new queries will observe the `table.count` to be 0 and old ones that are still
                // iterating the table will not be able to access this column. Note that the `table.count` only changes while a
                // `table.keys` upgradable lock is held.
                drop(write);
            }
        }
        database.keys().release(&keys[..count.get()]);
        database.listen.on_destroy(&keys[..count.get()], table);
        return count.get();
    }
}
