use std::collections::HashMap;

use crate::{
    database::Database,
    key::{Key, Slot},
    table::Table,
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
            // TODO: Between this release and the table lock, this key may be found in a `fold` query
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
            if state.rows.len() == 0 {
                continue;
            }

            debug_assert!(state.low <= state.high);
            let range = state.low..state.high + 1;
            self.database
                .remove_from_table(state.table, &state.rows, range);
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
