use crate::{key::Key, Database};
use std::{
    any::TypeId,
    collections::{HashMap, HashSet},
    ptr::NonNull,
};

/*
    - Link mutably the operations with `Defer` such that no conflicting operations may happen at once.
    - Determine at 'operation-time' if an ordering is required between the previous operations and the new ones.
    - Only `Defer` has a public resolve method?
    - Users may use multiple `Defer` at the same time at the cost of getting no ordering between them.
    - More sharing of buffers is possible!
*/

pub struct Defer<'d> {
    database: &'d Database,
    indices: HashMap<TypeId, usize>,
    keys: (Vec<Key>, HashSet<Key>, HashMap<Key, u32>),
    pointers: Vec<NonNull<()>>,
}

impl Database {
    pub fn defer(&self) -> Defer {
        todo!()
    }
}

impl<'d> Defer<'d> {
    // pub fn create<T: Template>(&mut self) -> Result<Create<'d, '_, T>, Error> {
    //     todo!()
    // }

    // pub fn destroy(&mut self) -> Destroy<'d, '_> {
    //     todo!()
    // }

    // pub fn destroy_all<F: Filter>(&mut self) -> DestroyAll<'d, '_, F> {
    //     todo!()
    // }

    // pub fn add<T: Template>(&mut self) -> Result<Add<'d, '_, T>, Error> {
    //     todo!()
    // }

    // pub fn add_all<T: Template, F: Filter>(&mut self) -> Result<AddAll<'d, '_, T, F>, Error> {
    //     todo!()
    // }

    // pub fn remove<T: Template>(&mut self) -> Result<Remove<'d, '_, T>, Error> {
    //     todo!()
    // }

    // pub fn remove_all<T: Template, F: Filter>(&mut self) -> Result<RemoveAll<'d, '_, T, F>, Error> {
    //     todo!()
    // }

    /// Resolves all deferred operations.
    ///
    /// In order to prevent deadlocks, **do not call this method while using a `Query`**.
    pub fn resolve(&mut self) {
        todo!()
    }

    /// Removes all deferred operations without resolving them.
    pub fn clear(&mut self) {
        todo!()
    }
}

impl Drop for Defer<'_> {
    fn drop(&mut self) {
        self.resolve()
    }
}
