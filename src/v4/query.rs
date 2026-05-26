use crate::v4::{
    Error, Meta, Rows, Store,
    module::{self, Access, Dependency, Resource},
    table,
    utility::Push,
};
use core::{
    any::TypeId,
    iter::{empty, once},
    marker::PhantomData,
    slice,
};

pub trait Item {
    type State;
    type Item<'a>
    where
        Self: 'a;

    fn declare(
        &self,
        state: &Self::State,
        table: &table::Table,
    ) -> impl Iterator<Item = Dependency>;
    fn initialize(&self, table: &table::Table) -> Option<Self::State>;
    fn get<'a>(&'a self, state: &'a Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a;
}

pub struct Query<'a, I: Item> {
    query: &'a I,
    states: &'a [(u32, I::State)],
    tables: &'a [table::Table],
}

pub struct Iter<'a, I: Item> {
    query: &'a I,
    states: slice::Iter<'a, (u32, I::State)>,
    tables: &'a [table::Table],
}

pub struct Module<A = ()>(pub(crate) A);

pub struct Row;
pub struct Table;
pub struct Read<T: ?Sized>(PhantomData<T>);
pub struct Write<T: ?Sized>(PhantomData<T>);
pub struct ReadWith(Meta);
pub struct WriteWith(Meta);

impl Module {
    pub const fn new() -> Self {
        Self(())
    }
}

impl<I: Item> Module<I> {
    pub fn row(self) -> Module<I::Out>
    where
        I: Push<Row>,
    {
        self.push(Row)
    }

    pub fn table(self) -> Module<I::Out>
    where
        I: Push<Table>,
    {
        self.push(Table)
    }

    pub fn read<T: 'static>(self) -> Module<I::Out>
    where
        I: Push<Read<T>>,
    {
        self.push(Read(PhantomData))
    }

    pub fn read_with(self, meta: Meta) -> Module<I::Out>
    where
        I: Push<ReadWith>,
    {
        self.push(ReadWith(meta))
    }

    pub fn write<T: 'static>(self) -> Module<I::Out>
    where
        I: Push<Write<T>>,
    {
        self.push(Write(PhantomData))
    }

    // pub fn write_with(self, meta: Meta) -> Build<(WriteWith, A)> {
    //     Build((WriteWith(meta), self.0))
    // }

    fn push<R: Item>(self, query: R) -> Module<I::Out>
    where
        I: Push<R>,
    {
        Module(self.0.push(query))
    }
}

impl<I: Item> module::Module for Module<I> {
    type Item<'a>
        = Query<'a, I>
    where
        Self: 'a;
    type State = (usize, Vec<(u32, I::State)>);

    fn declare(&self, state: &Self::State, store: &Store) -> impl Iterator<Item = Dependency> {
        state.1.iter().flat_map(|(table, state)| {
            let table = unsafe { store.tables.get_unchecked(*table as usize) };
            self.0.declare(state, table)
        })
    }

    fn initialize(&self, _: &mut Store) -> Result<Self::State, Error> {
        Ok((0, Vec::new()))
    }

    fn update(&self, state: &mut Self::State, store: &mut Store) -> Result<bool, Error> {
        let count = state.1.len();
        while let Some(table) = store.tables.get(state.0) {
            state.0 += 1;
            if let Some(query) = self.0.initialize(table) {
                state.1.push((table.index(), query));
            }
        }
        Ok(count < state.1.len())
    }

    fn get<'a>(&'a self, state: &'a mut Self::State, store: &'a Store) -> Self::Item<'a>
    where
        Self: 'a,
    {
        Query {
            query: &self.0,
            states: &state.1,
            tables: &store.tables,
        }
    }
}

impl<'a, I: Item> Iterator for Iter<'a, I> {
    type Item = I::Item<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let (table, state) = self.states.next()?;
        let table = unsafe { self.tables.get_unchecked(*table as usize) };
        Some(self.query.get(state, table))
    }
}

impl Query<'_, ()> {
    pub const fn build() -> Module {
        Module::new()
    }
}

impl<'a, I: Item> Query<'a, I> {
    // pub fn tables(&self) -> impl Iterator<Item = &Table> {
    //     self.states
    //         .iter()
    //         .map(|(table, _)| unsafe { self.tables.get_unchecked(*table as usize)
    // }) }

    // pub fn count(&mut self) -> usize {
    //     self.tables().map(|table| table.count() as usize).sum()
    // }

    pub fn iter(&mut self) -> Iter<'_, I> {
        Iter {
            query: self.query,
            states: self.states.iter(),
            tables: self.tables,
        }
    }
}

impl<I: Item> Item for &I {
    type Item<'a>
        = I::Item<'a>
    where
        Self: 'a;
    type State = I::State;

    fn declare(
        &self,
        state: &Self::State,
        table: &table::Table,
    ) -> impl Iterator<Item = Dependency> {
        I::declare(self, state, table)
    }

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        I::initialize(self, table)
    }

    fn get<'a>(&'a self, state: &'a Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        I::get(self, state, table)
    }
}

impl<I: Item> Item for &mut I {
    type Item<'a>
        = I::Item<'a>
    where
        Self: 'a;
    type State = I::State;

    fn declare(
        &self,
        state: &Self::State,
        table: &table::Table,
    ) -> impl Iterator<Item = Dependency> {
        I::declare(self, state, table)
    }

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        I::initialize(self, table)
    }

    fn get<'a>(&'a self, state: &'a Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        I::get(self, state, table)
    }
}

impl Item for () {
    type Item<'a>
        = ()
    where
        Self: 'a;
    type State = ();

    fn declare(&self, _: &Self::State, _: &table::Table) -> impl Iterator<Item = Dependency> {
        empty()
    }

    fn initialize(&self, _: &table::Table) -> Option<Self::State> {
        Some(())
    }

    fn get<'a>(&self, _: &'a Self::State, _: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
    }
}

impl<A0: Item, A1: Item> Item for (A0, A1) {
    type Item<'a>
        = (A0::Item<'a>, A1::Item<'a>)
    where
        Self: 'a;
    type State = (A0::State, A1::State);

    fn declare(
        &self,
        state: &Self::State,
        table: &table::Table,
    ) -> impl Iterator<Item = Dependency> {
        self.0
            .declare(&state.0, table)
            .chain(self.1.declare(&state.1, table))
    }

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Some((self.0.initialize(table)?, self.1.initialize(table)?))
    }

    fn get<'a>(&'a self, state: &'a Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        (self.0.get(&state.0, table), self.1.get(&state.1, table))
    }
}

impl<T: 'static> Item for Read<T> {
    type Item<'a>
        = &'a [T]
    where
        Self: 'a;
    type State = u32;

    fn declare(
        &self,
        state: &Self::State,
        table: &table::Table,
    ) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Read,
            resource: Resource::Column {
                table: table.index(),
                index: *state,
            },
        })
    }

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Some(table.column(TypeId::of::<T>())?.index())
    }

    fn get<'a>(&'a self, state: &'a Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        let column = unsafe { table.columns.get_unchecked(*state as usize) };
        unsafe { column.as_ref(table.count) }
    }
}

impl<T: 'static> Item for Write<T> {
    type Item<'a>
        = &'a mut [T]
    where
        Self: 'a;
    type State = u32;

    fn declare(
        &self,
        state: &Self::State,
        table: &table::Table,
    ) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Write,
            resource: Resource::Column {
                table: table.index(),
                index: *state,
            },
        })
    }

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Some(table.column(TypeId::of::<T>())?.index())
    }

    fn get<'a>(&'a self, state: &Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        let column = unsafe { table.columns.get_unchecked(*state as usize) };
        unsafe { column.as_mut(table.count) }
    }
}

impl Item for Row {
    type Item<'a>
        = Rows<'a>
    where
        Self: 'a;
    type State = u32;

    fn declare(
        &self,
        state: &Self::State,
        table: &table::Table,
    ) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Read,
            resource: Resource::Column {
                table: table.index(),
                index: *state,
            },
        })
    }

    fn initialize(&self, _: &table::Table) -> Option<Self::State> {
        None
    }

    fn get<'a>(&'a self, _: &Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        Rows::new(0..table.count(), table.index())
    }
}

impl Item for Table {
    type Item<'a>
        = &'a table::Table
    where
        Self: 'a;
    type State = ();

    fn declare(&self, _: &Self::State, table: &table::Table) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Read,
            resource: Resource::Table {
                index: table.index(),
            },
        })
    }

    fn initialize(&self, _: &table::Table) -> Option<Self::State> {
        Some(())
    }

    fn get<'a>(&'a self, _: &Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        table
    }
}

impl Item for ReadWith {
    type Item<'a>
        = &'a table::Column
    where
        Self: 'a;
    type State = u32;

    fn declare(
        &self,
        state: &Self::State,
        table: &table::Table,
    ) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Read,
            resource: Resource::Column {
                table: table.index(),
                index: *state,
            },
        })
    }

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Some(table.column(self.0.identifier)?.index())
    }

    fn get<'a>(&'a self, state: &Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        unsafe { table.columns.get_unchecked(*state as usize) }
    }
}

// impl Query for WriteWith {
//     type Item<'a> = &'a mut column::Column where
// Self: 'a;
//     type State = u32;
//
//     fn declare(
//         &self,
//         state: &Self::State,
//         table: &table::Table,
//     ) -> impl Iterator<Item = Dependency> {
//         once(Dependency::new(
//             Access::Write,
//             Resource::Column {
//                 table: table.index(),
//                 index: *state,
//             },
//         ))
//     }
//
//     fn initialize(&self, table: &table::Table) -> Option<Self::State> {
//         Some(table.column(self.0.identifier)?.index())
//     }

//     fn get<'a>(&'a self, state: &Self::State, table: &'a table::Table) ->
// Self::Item<'a> where
// Self: 'a {         unsafe { table.columns.get_unchecked(*state as
// usize) }     }
// }
