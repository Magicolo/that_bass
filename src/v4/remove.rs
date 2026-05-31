use crate::v4::{Error, Row, Store, module::Module, table::Tables, utility::ranges};
use core::cmp::Reverse;
use itertools::Itertools;

pub struct Build(());
pub struct Remove {
    rows: Vec<(u32, u32)>,
    tables: Tables,
}

impl Remove {
    pub const fn builder() -> Build {
        Build::new()
    }

    pub fn one(&mut self, row: Row<'_>) {
        self.rows.push((row.table(), row.row()));
    }

    pub fn all(&mut self) {
        todo!()
    }

    pub fn resolve(&mut self) {
        self.rows
            .sort_unstable_by_key(|pair| (pair.0, Reverse(pair.1)));
        for chunk in self.rows.chunk_by(|old, new| old.0 == new.0) {
            let Some(&(table, _)) = chunk.first() else {
                continue;
            };
            let rows = chunk.iter().map(|pair| pair.1);
            self.tables.map(table, |table| table.remove(rows));
        }
        self.rows.clear();
    }
}

impl Build {
    pub const fn new() -> Self {
        Self(())
    }

    pub fn build(self, store: &Store) -> Remove {
        Remove {
            rows: Vec::new(),
            tables: store.tables().clone(),
        }
    }
}
