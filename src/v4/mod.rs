pub mod column;
pub mod error;
pub mod meta;
pub mod module;
pub mod query;
pub mod row;
pub mod state;
pub mod table;
pub mod template;
pub mod utility;
pub mod vector;

pub use column::Column;
pub use error::Error;
pub use meta::Meta;
pub use row::{Row, Rows};
pub use table::Table;
use template::column;
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

    fn find_or_insert_table(
        &mut self,
        metas: impl IntoIterator<Item = Meta>,
    ) -> Result<u32, Error> {
        let metas = sort(metas)?;
        Ok(match self.find_table(&metas) {
            Some(index) => index,
            None => {
                let index = self
                    .tables
                    .len()
                    .try_into()
                    .map_err(Error::TablesOverflow)?;
                self.tables.push(Table::new(index, metas)?);
                index
            }
        })
    }
}

fn sort(metas: impl IntoIterator<Item = Meta>) -> Result<Vec<Meta>, Error> {
    let mut metas = metas.into_iter().collect::<Vec<_>>();
    metas.sort_unstable_by_key(|meta| meta.identifier);
    for [left, right] in metas.array_windows::<2>() {
        if left.identifier == right.identifier {
            return Err(Error::DuplicateMeta);
        }
    }
    Ok(metas)
}
