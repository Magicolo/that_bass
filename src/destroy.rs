use crate::{database::Database, key::Key};

pub struct Destroy<'a> {
    database: &'a Database,
}

impl Database {
    pub fn destroy(&self) -> Destroy {
        Destroy { database: self }
    }
}

impl Destroy<'_> {
    #[inline]
    pub fn one(&mut self, key: Key) -> bool {
        self.database.remove_from_table(key).is_some()
    }

    /// Destroys all provided `keys` and returns the count of the keys that were successfully destroyed.
    #[inline]
    pub fn all<I: IntoIterator<Item = Key>>(&mut self, keys: I) -> usize {
        keys.into_iter().filter(|&key| self.one(key)).count()
    }
}
