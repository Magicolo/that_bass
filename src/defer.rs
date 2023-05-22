use crate::Database;
use std::{
    any::{Any, TypeId},
    cell::RefCell,
    collections::HashMap,
};

/*
    There would be 3 operations internally:
    - Create
    - Destroy
    - Move(source, target) -> Will be sorted in the lowest index table.


    - Link mutably the operations with `Defer` such that no conflicting operations may happen at once.
    - Determine at 'operation-time' if an ordering is required between the previous operations and the new ones.
    - Only `Defer` has a public resolve method?
    - Users may use multiple `Defer` at the same time at the cost of getting no ordering between them.
    - More sharing of buffers is possible!
*/

pub struct Defer<'d> {
    database: &'d Database,
    creates: RefCell<Creates>,
    states: Vec<State>,
}

pub(crate) struct Inner {}

struct State {}

struct Creates {
    indices: HashMap<TypeId, usize>,
    resolvers: Vec<Resolver>,
}

trait Resolve {
    fn resolve_create();
    fn resolve_destroy();
    fn resolve_non_ordered_modify();
    fn resolve_ordered_modify();
}

// struct Destroys {
//     keys: HashSet<Key>,
//     pending: Vec<(Key, &'d Slot, u32)>,
//     sorted: HashMap<u32, State<'d>>,
// }

struct Resolver {
    state: Box<dyn Any>,
    resolve: fn(&mut dyn Any, usize),
}

impl Database {
    pub fn defer(&self) -> Defer {
        todo!()
    }
}

impl<'d> Defer<'d> {
    // pub fn create<T: Template>(&self) -> Result<Create<'d, '_, T>, Error> {
    //     let mut creates = self.creates.borrow_mut();
    //     let identifier = TypeId::of::<T>();
    //     let index = match creates.indices.get(&identifier) {
    //         Some(&index) => index,
    //         None => {
    //             let index = creates.resolvers.len();
    //             creates.indices.insert(identifier, index);
    //             creates.resolvers.push(Resolver {
    //                 state: Box::new(()),
    //                 resolve: |state, count| {},
    //             });
    //             index
    //         }
    //     };
    //     let Some(resolver) = creates.resolvers.get_mut(index) else {
    //         return Err(Error::MissingIndex);
    //     };
    //     todo!()
    // }

    // pub fn destroy(&self) -> Destroy<'d, '_> {
    //     todo!()
    // }

    // pub fn destroy_all<F: Filter>(&self) -> DestroyAll<'d, '_, F> {
    //     todo!()
    // }

    // pub fn add<T: Template>(&self) -> Result<Add<'d, '_, T>, Error> {
    //     todo!()
    // }

    // pub fn add_all<T: Template, F: Filter>(&self) -> Result<AddAll<'d, '_, T, F>, Error> {
    //     todo!()
    // }

    // pub fn remove<T: Template>(&self) -> Result<Remove<'d, '_, T>, Error> {
    //     todo!()
    // }

    // pub fn remove_all<T: Template, F: Filter>(&self) -> Result<RemoveAll<'d, '_, T, F>, Error> {
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
