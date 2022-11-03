use std::{
    collections::{HashMap, HashSet},
    num::NonZeroUsize,
};

use parking_lot::RwLockWriteGuard;

use crate::{
    key::{Key, Slot},
    table::{self, Table},
    Database, Error,
};

pub struct Destroy<'d> {
    database: &'d Database,
    keys: HashSet<Key>,
    pending: Vec<(Key, &'d Slot, u32)>,
    sorted: HashMap<u32, State<'d>>,
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
        }
    }
}

impl<'d> Destroy<'d> {
    #[inline]
    pub fn one(&mut self, key: Key) -> bool {
        // TODO: Prevent duplicate keys to be added...
        match self.database.keys().get(key) {
            Ok((slot, table)) => {
                Self::sort(key, slot, table, &mut self.sorted, self.database).is_ok()
            }
            Err(_) => false,
        }
    }

    #[inline]
    pub fn all<I: IntoIterator<Item = Key>>(&mut self, keys: I) -> usize {
        keys.into_iter()
            .filter_map(|key| self.one(key).then_some(()))
            .count()
    }

    #[inline]
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
        self.database.keys.release(self.keys.drain());
        count
    }

    pub fn clear(&mut self) {
        self.pending.clear();
        for state in self.sorted.values_mut() {
            state.rows.clear();
        }
    }

    fn resolve_sorted(&mut self) {
        #[inline]
        fn batch<'d>(mut previous: usize, index: &mut usize, rows: &[(Key, &'d Slot, u32)]) {
            // Try to batch contiguous rows.
            while let Some(&(.., current)) = rows.get(*index) {
                let current = current as usize;
                if previous + 1 == current {
                    previous = current;
                    *index += 1;
                } else {
                    break;
                }
            }
        }

        for state in self.sorted.values_mut() {
            let (mut inner, low, high) = Self::filter(
                state.table,
                &mut state.rows,
                &mut self.keys,
                &mut self.pending,
            );
            let count = state.rows.len();
            if count == 0 {
                continue;
            }
            debug_assert!(low <= high);

            let range = low..high + 1;
            let inner = &mut *inner;
            let head = inner.release(count);
            let keys = inner.keys.get_mut();
            let (low, high) = (range.start as usize, range.end as usize);
            if range.len() == count {
                // The destroy range is contiguous.
                let over = high.saturating_sub(head);
                let end = count - over;
                if let Some(end) = NonZeroUsize::new(end) {
                    // Squash the range at the end of the table on the beginning of the removed range.
                    let start = head + over;
                    for store in inner.stores.iter_mut() {
                        unsafe { store.squash(start, low, end) };
                    }

                    // Update the keys.
                    keys.copy_within(start..start + end.get(), low);
                    for i in low..low + end.get() {
                        let key = unsafe { *keys.get_unchecked(i) };
                        let slot = unsafe { self.database.keys().get_unchecked(key) };
                        slot.update(i as _);
                    }
                }

                if let Some(over) = NonZeroUsize::new(over) {
                    for store in inner.stores.iter_mut() {
                        unsafe { store.drop(head, over) };
                    }
                }
            } else {
                let mut index = 0;
                let mut last = keys.len();

                // Try to consume the rows in the range `end..keys.len()` since they are guaranteed to be valid.
                let end = high.max(head);
                while last > end {
                    match state.rows.get(index) {
                        Some(&(.., row)) => {
                            let row = row as usize;
                            let mut previous = row;
                            let start = index;
                            index += 1;

                            if row < head {
                                last -= 1;

                                // Try to batch contiguous squashes.
                                while last > end {
                                    if let Some(&(.., row)) = state.rows.get(index) {
                                        let row = row as usize;
                                        if previous + 1 == row && row < head {
                                            previous = row;
                                            last -= 1;
                                            index += 1;
                                        } else {
                                            break;
                                        }
                                    }
                                }

                                let count = unsafe { NonZeroUsize::new_unchecked(index - start) };
                                for store in inner.stores.iter_mut() {
                                    unsafe { store.squash(last, row, count) };
                                }
                                keys.copy_within(last..last + count.get(), row);

                                for row in row..row + count.get() {
                                    let key = unsafe { *keys.get_unchecked(row) };
                                    let slot = unsafe { self.database.keys().get_unchecked(key) };
                                    slot.update(row as _);
                                }
                            } else {
                                // Try to batch contiguous drops.
                                batch(previous, &mut index, &state.rows);
                                let count = unsafe { NonZeroUsize::new_unchecked(index - start) };
                                for store in inner.stores.iter_mut() {
                                    unsafe { store.drop(row, count) };
                                }
                            }
                        }
                        None => break,
                    }
                }

                // Tag keys that are going to be removed such removed keys and valid keys can be differentiated.
                for &(.., row) in &state.rows[index..] {
                    *unsafe { keys.get_unchecked_mut(row as usize) } = Key::NULL;
                }

                // Remove that remaining rows the slow way.
                while let Some(&(.., row)) = state.rows.get(index) {
                    let row = row as usize;
                    let previous = row;
                    let start = index;
                    index += 1;

                    if row < head {
                        // Find the next valid row to move.
                        while keys.get(last).copied() == Some(Key::NULL) {
                            last -= 1;
                        }
                        debug_assert!(last >= head);

                        for store in inner.stores.iter_mut() {
                            unsafe { store.squash(last, row, NonZeroUsize::new_unchecked(1)) };
                        }

                        let key = unsafe { *keys.get_unchecked_mut(last) };
                        unsafe { *keys.get_unchecked_mut(row) = key };
                        let slot = unsafe { self.database.keys().get_unchecked(key) };
                        slot.update(row as _);
                        last -= 1;
                    } else {
                        // Try to batch contiguous drops.
                        batch(previous, &mut index, &state.rows);
                        let count = unsafe { NonZeroUsize::new_unchecked(index - start) };
                        for store in inner.stores.iter_mut() {
                            unsafe { store.drop(row, count) };
                        }
                    }
                }
            }

            for &(key, slot, row) in state.rows.iter() {
                debug_assert_eq!(slot.indices(), (key.generation(), state.table.index()));
                debug_assert_eq!(slot.row(), row);
                slot.release();
            }
        }
    }

    fn sort(
        key: Key,
        slot: &'d Slot,
        table: u32,
        sorted: &mut HashMap<u32, State<'d>>,
        database: &'d Database,
    ) -> Result<(), Error> {
        match sorted.get_mut(&table) {
            Some(state) => state.rows.push((key, slot, u32::MAX)),
            None => {
                sorted.insert(
                    table,
                    State {
                        table: unsafe { database.tables().get_unchecked(table as _) },
                        rows: vec![(key, slot, u32::MAX)],
                    },
                );
            }
        }
        Ok(())
    }

    fn filter<'a>(
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

impl Drop for Destroy<'_> {
    fn drop(&mut self) {
        self.resolve();
    }
}
