use crate::{
    core::utility::{fold_swap, get_unchecked, get_unchecked_mut, swap_unchecked, ONE},
    event::Events,
    filter::Filter,
    key::{Key, Keys},
    table::{self, Table, Tables},
    Database,
};
use parking_lot::{RwLockUpgradableReadGuard, RwLockWriteGuard};
use std::{
    collections::HashSet,
    num::NonZeroUsize,
    sync::{atomic::Ordering, Arc},
};

pub struct Destroy<'d, F = ()> {
    database: &'d Database,
    keys: Keys<'d>,
    events: Events<'d>,
    set: HashSet<Key>, // A `HashSet` is used because the move algorithm assumes that rows will be unique.
    indices: Vec<usize>, // May be reordered (ex: by `fold_swap`).
    states: Vec<Result<State, u32>>, // Must remain sorted by `state.table.index()` for `binary_search` to work.
    pending: Vec<(Key, u32)>,
    moves: Vec<(usize, usize, NonZeroUsize)>,
    drops: Vec<(usize, NonZeroUsize)>,
    filter: F,
}

/// Destroys all keys in tables that satisfy the filter `F`.
pub struct DestroyAll<'d, F = ()> {
    database: &'d Database,
    tables: Tables<'d>,
    keys: Keys<'d>,
    events: Events<'d>,
    index: usize,
    states: Vec<Arc<Table>>,
    filter: F,
}

struct State {
    table: Arc<Table>,
    rows: Vec<(Key, u32)>,
}

impl Database {
    pub fn destroy(&self) -> Destroy<'_> {
        Destroy {
            database: self,
            keys: self.keys(),
            events: self.events(),
            set: HashSet::new(),
            pending: Vec::new(),
            states: Vec::new(),
            indices: Vec::new(),
            moves: Vec::new(),
            drops: Vec::new(),
            filter: (),
        }
    }

    pub fn destroy_all(&self) -> DestroyAll<'_> {
        DestroyAll {
            database: self,
            tables: self.tables(),
            keys: self.keys(),
            events: self.events(),
            index: 0,
            states: Vec::new(),
            filter: (),
        }
    }
}

impl<'d, F> Destroy<'d, F> {
    #[inline]
    pub fn one(&mut self, key: Key) {
        self.set.insert(key);
    }

    #[inline]
    pub fn all<I: IntoIterator<Item = Key>>(&mut self, keys: I) {
        self.set.extend(keys);
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
            events: self.events,
            set: self.set,
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
        self.set.len()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = Key> + '_ {
        self.set.iter().copied()
    }

    pub fn drain(&mut self) -> impl ExactSizeIterator<Item = Key> + '_ {
        debug_assert_eq!(self.pending.len(), 0);
        debug_assert_eq!(self.indices.len(), 0);
        self.set.drain()
    }

    pub fn clear(&mut self) {
        debug_assert_eq!(self.pending.len(), 0);
        debug_assert_eq!(self.indices.len(), 0);
        self.set.clear();
    }
}

impl<'d, F: Filter> Destroy<'d, F> {
    pub fn resolve(&mut self) -> usize {
        for (key, result) in self.keys.get_all(self.set.iter().copied()) {
            if let Ok((_, table)) = result {
                Self::sort(
                    key,
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

            for (key, table) in self.pending.drain(..) {
                Self::sort(
                    key,
                    table,
                    &mut self.indices,
                    &mut self.states,
                    &self.filter,
                    self.database,
                );
            }
        }
        self.set.clear();
        debug_assert_eq!(self.set.len(), 0);
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
                let (low, high) = Self::retain(&self.keys, &state.table, &mut state.rows, pending);
                let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                    return Ok(sum);
                };
                Self::resolve_rows(
                    &state.table,
                    &mut state.rows,
                    &self.set,
                    moves,
                    drops,
                    keys,
                    (low, high, count),
                    &self.keys,
                    &self.events,
                );
                Ok(sum + count.get())
            },
            |sum, (states, pending, moves, drops), index| {
                let result = unsafe { get_unchecked_mut(states, *index) };
                let state = unsafe { result.as_mut().unwrap_unchecked() };
                let keys = state.table.keys.upgradable_read();
                let (low, high) = Self::retain(&self.keys, &state.table, &mut state.rows, pending);
                let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                    return sum;
                };
                Self::resolve_rows(
                    &state.table,
                    &mut state.rows,
                    &self.set,
                    moves,
                    drops,
                    keys,
                    (low, high, count),
                    &self.keys,
                    &self.events,
                );
                sum + count.get()
            },
        )
    }

    fn resolve_rows(
        table: &Table,
        rows: &mut Vec<(Key, u32)>,
        set: &HashSet<Key>,
        moves: &mut Vec<(usize, usize, NonZeroUsize)>,
        drops: &mut Vec<(usize, NonZeroUsize)>,
        table_keys: RwLockUpgradableReadGuard<table::Keys>,
        (low, high, count): (u32, u32, NonZeroUsize),
        keys: &Keys,
        events: &Events,
    ) {
        debug_assert_eq!(moves.len(), 0);
        debug_assert_eq!(drops.len(), 0);
        debug_assert!(low <= high);

        let range = low..high + 1;
        let head = table.count.load(Ordering::Acquire) - count.get();
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
            while let Some(&(.., row)) = rows.get(index) {
                let row = row as usize;
                let mut previous = row;
                let start = index;
                index += 1;

                if row < head {
                    // Find the next valid row to move.
                    while set.contains(unsafe { get_unchecked(&table_keys, cursor) }) {
                        cursor += 1;
                    }
                    debug_assert!(cursor < head + count.get());
                    moves.push((cursor, row, ONE));
                    cursor += 1;
                } else {
                    // Try to batch contiguous drops.
                    while let Some(&(.., current)) = rows.get(index) {
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

        let mut table_keys = RwLockUpgradableReadGuard::upgrade(table_keys);
        for &(source, target, count) in moves.iter() {
            for i in 0..count.get() {
                // Swap is used such that destroyed keys are gathered at the end of the table when the operation is complete.
                // - This allows `on_destroy` to use this fact.
                unsafe { swap_unchecked(&mut table_keys, source + i, target + i) };
            }
            keys.update_all(&table_keys, target..target + count.get());
        }
        for column in table.columns() {
            if column.meta().size() > 0 {
                for &(source, target, count) in moves.iter() {
                    // Since a write lock is held on `keys`, no need to take a column lock.
                    unsafe { column.squash(source, target, count) };
                }
            }
        }
        // Must be decremented under the write lock to ensure that no query can observe the `table.count` with the invalid rows
        // at the end.
        table.count.fetch_sub(count.get() as _, Ordering::Release);

        // Keep an upgradable lock to prevent `create/add` operations to resolve during these last steps.
        let table_keys = RwLockWriteGuard::downgrade_to_upgradable(table_keys);
        // It is ok to release the slots under the upgradable lock (rather than under the write lock) over `keys` because queries do an
        // additionnal validation (`keys.get(row) == Some(&key)`) which prevents the case where a key would be thought to be in the
        // query while going through `database.keys().get()` and yet its `slot.row` points to the wrong row.
        // - Note that it is possible to observe an inconsistency between `query.has` and `query.find` for a short time since
        // `query.has` does not do the additionnal validation.
        keys.release_all(&table_keys[head..head + count.get()]);
        for column in table.columns() {
            if column.meta().drop.0 {
                for &(index, count) in drops.iter() {
                    // Since the `table.count` has been decremented under the `keys` write lock was held, these indices are only observable
                    // by this thread (assuming the indices are `> head`).
                    debug_assert!(index >= head);
                    unsafe { column.drop(index, count) };
                }
            }
        }

        events.emit_destroy(&table_keys[head..head + count.get()], table);
        drop(table_keys);
        // The `recycle` step can be done outside of the lock. This means that the keys within `rows` may be very briefly
        // non-reusable for other threads, which is fine.
        keys.recycle_all(rows.drain(..).map(|(key, ..)| key));
        moves.clear();
        drops.clear();
    }

    fn sort(
        key: Key,
        table: u32,
        indices: &mut Vec<usize>,
        states: &mut Vec<Result<State, u32>>,
        filter: &F,
        database: &'d Database,
    ) {
        let index = match states.binary_search_by_key(&table, |result| match result {
            Ok(state) => state.table.index(),
            Err(index) => *index,
        }) {
            Ok(index) => index,
            Err(index) => {
                let result = match database.tables().get_shared(table as _) {
                    Ok(table) if filter.filter(&table, database) => Ok(State {
                        table,
                        rows: Vec::new(),
                    }),
                    Ok(table) => Err(table.index()),
                    Err(_) => unreachable!("Provided table index must be in range."),
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
        }
    }

    /// Call this while holding a lock on `table`.
    fn retain<'a>(
        keys: &Keys,
        table: &'a Table,
        rows: &mut Vec<(Key, u32)>,
        pending: &mut Vec<(Key, u32)>,
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
                    pending.push((key, table_index));
                }
                Err(_) => {
                    rows.swap_remove(i);
                }
            }
        }
        debug_assert_eq!(low <= high, rows.len() > 0);
        (low, high)
    }
}

impl<'d, F> DestroyAll<'d, F> {
    pub fn filter<G: Filter + Default>(self) -> DestroyAll<'d, (F, G)> {
        self.filter_with(G::default())
    }

    pub fn filter_with<G: Filter>(mut self, filter: G) -> DestroyAll<'d, (F, G)> {
        self.states
            .retain(|table| filter.filter(table, self.database));
        DestroyAll {
            database: self.database,
            tables: self.tables,
            keys: self.keys,
            events: self.events,
            index: self.index,
            states: self.states,
            filter: (self.filter, filter),
        }
    }
}

impl<'d, F: Filter> DestroyAll<'d, F> {
    pub fn resolve(&mut self) -> usize {
        while let Ok(table) = self.tables.get_shared(self.index) {
            self.index += 1;
            if self.filter.filter(&table, self.database) {
                self.states.push(table);
            }
        }

        fold_swap(
            &mut self.states,
            0,
            (&mut self.keys, &self.events),
            |sum, (keys, events), table| {
                let table_keys = table.keys.try_upgradable_read().ok_or(sum)?;
                Ok(sum + Self::resolve_table(table_keys, table, keys, events))
            },
            |sum, (keys, events), table| {
                let table_keys = table.keys.upgradable_read();
                sum + Self::resolve_table(table_keys, table, keys, events)
            },
        )
    }

    fn resolve_table(
        table_keys: RwLockUpgradableReadGuard<table::Keys>,
        table: &Table,
        keys: &mut Keys,
        events: &Events,
    ) -> usize {
        let count = table.count.swap(0, Ordering::AcqRel);
        let Some(count) = NonZeroUsize::new(count) else {
            return 0;
        };
        for column in table.columns() {
            if column.meta().drop.0 {
                let write = column.data().write();
                unsafe { column.drop(0, count) };
                // No need to accumulate locks since new queries will observe the `table.count` to be 0 and old ones that are still
                // iterating the table will not be able to access this column. Note that the `table.count` only changes while a
                // `table.keys` upgradable lock is held.
                drop(write);
            }
        }
        keys.update();
        keys.release_all(&table_keys[..count.get()]);
        events.emit_destroy(&table_keys[..count.get()], table);
        keys.recycle_all(table_keys[..count.get()].iter().copied());
        return count.get();
    }
}
