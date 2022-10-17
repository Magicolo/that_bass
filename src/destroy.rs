use std::collections::HashMap;

use crate::{database::Database, key::Key};

pub struct Destroy<'a> {
    database: &'a Database,
    keys: Vec<Key>,
    map: HashMap<u32, (Vec<u32>, u32, u32)>,
}

impl Database {
    pub fn destroy(&self) -> Destroy {
        Destroy {
            database: self,
            keys: Vec::new(),
            map: HashMap::new(),
        }
    }
}

impl Destroy<'_> {
    #[inline]
    pub fn one(&mut self, key: Key) {
        self.keys.push(key);
    }

    #[inline]
    pub fn all<I: IntoIterator<Item = Key>>(&mut self, keys: I) {
        self.keys.extend(keys);
    }

    pub fn resolve(&mut self) -> usize {
        let mut count = 0;
        let keys = self.database.keys();
        for key in self.keys.drain(..) {
            if let Ok((table, row)) = keys.release(key) {
                let rows = self
                    .map
                    .entry(table)
                    .or_insert_with(|| (Vec::new(), u32::MAX, 0));
                rows.0.push(row);
                rows.1 = rows.1.min(row);
                rows.2 = rows.2.max(row);
                count += 1;
            }
        }

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
}

impl Drop for Destroy<'_> {
    fn drop(&mut self) {
        self.resolve();
    }
}
