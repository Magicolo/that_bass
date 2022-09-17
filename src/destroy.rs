use crate::{
    database::{Database, Inner},
    key::Key,
};

pub struct Destroy<'a> {
    inner: &'a Inner,
}

impl Database {
    pub fn destroy(&mut self) -> Destroy {
        todo!()
    }
}

impl Destroy<'_> {
    pub fn one(&mut self, key: Key) -> bool {
        todo!()
    }

    /// Destroys all provided `keys` and returns the count of the keys that were successfully destroyed.
    pub fn all<I: IntoIterator<Item = Key>>(&mut self, keys: I) -> usize {
        todo!()
    }
}
