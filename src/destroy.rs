use crate::{
    database::{Database, Inner},
    key::Key,
};

pub struct Destroy<'a> {
    inner: &'a Inner,
}

impl Database {
    pub fn destroy(&mut self) -> Destroy {
        Destroy {
            inner: self.inner(),
        }
    }
}

impl Destroy<'_> {
    #[inline]
    pub fn one(&mut self, key: Key) -> bool {
        self.inner.destroy(key).is_some()
    }

    /// Destroys all provided `keys` and returns the count of the keys that were successfully destroyed.
    #[inline]
    pub fn all<I: IntoIterator<Item = Key>>(&mut self, keys: I) -> usize {
        keys.into_iter().filter(|&key| self.one(key)).count()
    }
}
