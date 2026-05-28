use crate::v4::{
    Error, Index, Store,
    module::{Dependency, Module},
    utility::ranges,
};
use core::iter::empty;

pub struct Build(());
pub struct State(Vec<(u32, u32)>);

pub struct Remove<'a> {
    state: &'a mut Vec<(u32, u32)>,
}

impl Store {
    pub fn remove(&mut self, remove: Build) -> super::State<State> {
        self.state(State(Vec::new()))
    }
}

impl Remove<'_> {
    pub fn one(&mut self, row: Index) {
        self.state.push((row.table(), row.row()));
    }
}

impl Module for State {
    type Item<'a>
        = Remove<'a>
    where
        Self: 'a;

    fn declare(&self, _: &Store) -> impl Iterator<Item = Dependency> {
        empty()
    }

    fn update(&mut self, _: &mut Store) -> Result<bool, Error> {
        Ok(false)
    }

    fn resolve(&mut self, store: &mut Store) -> Result<(), Error> {
        self.0.sort();
        for (table, rows) in ranges(self.0.drain(..).rev()) {
            if let Some(table) = store.tables.get_mut(table as usize) {
                table.release(rows);
            }
        }
        Ok(())
    }

    fn get<'a>(&'a mut self, _: &'a Store) -> Self::Item<'a>
    where
        Self: 'a,
    {
        Remove { state: &mut self.0 }
    }
}

pub const fn remove() -> Build {
    Build(())
}
