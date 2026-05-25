use crate::v4::{Error, Row, Store, module, utility::ranges};

pub struct Module(());

pub struct Remove<'a> {
    state: &'a mut Vec<(u32, u32)>,
}

impl Remove<'_> {
    pub const fn build() -> Module {
        Module(())
    }

    pub fn one(&mut self, row: Row) {
        self.state.push((row.table(), row.row()));
    }
}

impl module::Module for Module {
    type Item<'a>
        = Remove<'a>
    where
        Self: 'a;
    type State = Vec<(u32, u32)>;

    fn initialize(&self, _: &mut Store) -> Result<Self::State, Error> {
        Ok(Vec::new())
    }

    fn update(&self, _: &mut Self::State, _: &Store) -> Result<bool, Error> {
        Ok(false)
    }

    fn get<'a>(&'a self, state: &'a mut Self::State, _: &'a Store) -> Self::Item<'a>
    where
        Self: 'a,
    {
        Remove { state }
    }

    fn resolve(&self, state: &mut Self::State, store: &mut Store) -> Result<(), Error> {
        state.sort();
        for (table, rows) in ranges(state.drain(..).rev()) {
            if let Some(table) = store.tables.get_mut(table as usize) {
                table.release(rows);
            }
        }
        Ok(())
    }
}
