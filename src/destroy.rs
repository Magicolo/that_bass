use std::collections::HashMap;

use crate::{
    database::Database,
    key::{Key, Slot},
};

pub struct Destroy<'a> {
    database: &'a Database,
    slots: Vec<(Key, &'a Slot)>,
    map: HashMap<u32, (Vec<u32>, u32, u32)>,
}

impl Database {
    pub fn destroy(&self) -> Destroy {
        Destroy {
            database: self,
            slots: Vec::new(),
            map: HashMap::new(),
        }
    }
}

impl Destroy<'_> {
    #[inline]
    pub fn one(&mut self, key: Key) -> bool {
        match self.database.keys().get(key) {
            Ok(slot) => {
                self.slots.push((key, slot));
                true
            }
            Err(_) => false,
        }
    }

    #[inline]
    pub fn all<I: IntoIterator<Item = Key>>(&mut self, keys: I) -> usize {
        keys.into_iter().filter(|&key| self.one(key)).count()
    }

    pub fn resolve(&mut self) -> usize {
        self.slots.retain(|(key, slot)| {
            if let Ok((table, row)) = slot.release(key.generation()) {
                let (rows, low, high) = self
                    .map
                    .entry(table)
                    .or_insert_with(|| (Vec::new(), u32::MAX, 0));
                rows.push(row);
                (*low, *high) = (row.min(*low), row.max(*high));
                true
            } else {
                false
            }
        });

        let count = self.slots.len();
        let keys = self.database.keys();
        keys.release(self.slots.drain(..).map(|(key, _)| key));

        let tables = self.database.tables();
        for (&table, (rows, low, high)) in self.map.iter_mut() {
            if rows.len() == 0 {
                continue;
            }

            debug_assert!(low <= high);
            let table = unsafe { tables.get_unchecked(table as _) };
            let range = *low..*high + 1;
            self.database.remove_from_table(table, rows, range);
            rows.clear();
            (*low, *high) = (u32::MAX, 0);
        }

        count
    }

    pub fn clear(&mut self) {
        self.slots.clear();
    }
}

impl Drop for Destroy<'_> {
    fn drop(&mut self) {
        self.resolve();
    }
}
