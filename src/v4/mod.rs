pub mod column;
pub mod error;
pub mod meta;
pub mod module;
pub mod query;
pub mod row;
pub mod table;
pub mod template;
pub mod utility;
pub mod vector;

pub use column::Column;
pub use error::Error;
pub use meta::Meta;
pub use row::{Row, Rows};
pub use table::Table;
pub use utility::{At, AtMut};
pub use vector::Vector;

#[derive(Default)]
pub struct Store {
    tables: Vec<Table>,
}

impl Store {
    pub const fn new() -> Self {
        Self { tables: Vec::new() }
    }

    fn find_table(&self, metas: &[Meta]) -> Option<u32> {
        self.tables
            .iter()
            .position(|table| {
                table
                    .columns()
                    .iter()
                    .map(|column| column.meta().identifier)
                    .eq(metas.iter().map(|meta| meta.identifier))
            })?
            .try_into()
            .ok()
    }
}
