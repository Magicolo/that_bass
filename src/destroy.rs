use std::{collections::HashMap, num::NonZeroUsize};

use crate::{
    key::{Key, Slot},
    table::Table,
    Database,
};

pub struct Destroy<'d> {
    database: &'d Database,
    pending: Vec<(Key, &'d Slot)>, // Note that `Slot::release` will remove duplicates.
    sorted: HashMap<u32, State<'d>>,
}

struct State<'d> {
    table: &'d Table,
    rows: Vec<u32>,
    low: u32,
    high: u32,
}

impl Database {
    pub fn destroy(&self) -> Destroy {
        Destroy {
            database: self,
            pending: Vec::new(),
            sorted: HashMap::new(),
        }
    }
}

impl Destroy<'_> {
    #[inline]
    pub fn one(&mut self, key: Key) -> bool {
        match self.database.keys().get(key) {
            Ok(slot) => {
                self.pending.push((key, slot));
                true
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

    pub fn resolve(&mut self) -> usize {
        self.pending.retain(|(key, slot)| {
            // TODO: Between this `slot.release` and the table lock, this key may be found in a `fold` query
            // while the same query would fail a `find` with the same key.
            if let Ok((table, row)) = slot.release(key.generation()) {
                let state = self.sorted.entry(table).or_insert_with(|| State {
                    table: unsafe { self.database.tables().get_unchecked(table as _) },
                    rows: Vec::new(),
                    low: u32::MAX,
                    high: 0,
                });
                state.rows.push(row);
                (state.low, state.high) = (row.min(state.low), row.max(state.high));
                true
            } else {
                false
            }
        });

        let count = self.pending.len();
        let keys = self.database.keys();
        keys.release(self.pending.drain(..).map(|(key, _)| key));

        for state in self.sorted.values_mut() {
            let count = state.rows.len();
            if count == 0 {
                continue;
            }

            debug_assert!(state.low <= state.high);
            let range = state.low..state.high + 1;
            let mut inner = state.table.inner.write();
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
                        unsafe { self.database.keys().get_unchecked(*keys.get_unchecked(i)) }
                            .update(state.table.index(), i as _);
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
                        Some(&row) => {
                            let row = row as usize;
                            let mut previous = row;
                            let start = index;
                            index += 1;

                            if row < head {
                                last -= 1;

                                // Try to batch contiguous squashes.
                                while last > end {
                                    if let Some(&row) = state.rows.get(index) {
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
                                    unsafe {
                                        self.database.keys().get_unchecked(*keys.get_unchecked(row))
                                    }
                                    .update(state.table.index(), row as _);
                                }
                            } else {
                                // Try to batch contiguous drops.
                                while let Some(&row) = state.rows.get(index) {
                                    let row = row as usize;
                                    if previous + 1 == row {
                                        previous = row;
                                        index += 1;
                                    } else {
                                        break;
                                    }
                                }

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
                for &row in &state.rows[index..] {
                    *unsafe { keys.get_unchecked_mut(row as usize) } = Key::NULL;
                }

                // Remove that remaining rows the slow way.
                while let Some(&row) = state.rows.get(index) {
                    let row = row as usize;
                    let mut previous = row;
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

                        unsafe {
                            let key = *keys.get_unchecked_mut(last);
                            *keys.get_unchecked_mut(row) = key;
                            self.database
                                .keys()
                                .get_unchecked(key)
                                .update(state.table.index(), row as _);
                        }

                        last -= 1;
                    } else {
                        // Try to batch contiguous drops.
                        while let Some(&row) = state.rows.get(index) {
                            let row = row as usize;
                            if previous + 1 == row {
                                previous = row;
                                index += 1;
                            } else {
                                break;
                            }
                        }

                        let count = unsafe { NonZeroUsize::new_unchecked(index - start) };
                        for store in inner.stores.iter_mut() {
                            unsafe { store.drop(row, count) };
                        }
                    }
                }
            }

            state.rows.clear();
            (state.low, state.high) = (u32::MAX, 0);
        }

        count
    }

    pub fn clear(&mut self) {
        self.pending.clear();
    }
}

impl Drop for Destroy<'_> {
    fn drop(&mut self) {
        self.resolve();
    }
}
