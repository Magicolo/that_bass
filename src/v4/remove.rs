use crate::v4::{Error, Row, Store, module::Module, table::Tables, utility::ranges};

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
        self.rows.sort();
        for (table, rows) in ranges(self.rows.drain(..).rev()) {
            self.tables.map(table, |table| table.release(rows));
        }
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
