use crate::v4::{Error, Meta, Rows, Store, column, module, table, utility::Push};
use core::{any::TypeId, iter, marker::PhantomData, slice::Iter};

pub trait Boba {
    type State;
    type Item<'a>
    where
        Self: 'a;

    fn initialize(&self, table: &table::Table) -> Option<Self::State>;
    fn get<'a>(&'a self, state: &'a Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a;
}

pub struct Query<'a, Q: Boba> {
    query: &'a Q,
    states: &'a [(u32, Q::State)],
    tables: &'a [table::Table],
}

pub struct Iterator<'a, Q: Boba> {
    query: &'a Q,
    states: Iter<'a, (u32, Q::State)>,
    tables: &'a [table::Table],
}

pub struct Module<Q = ()>(pub(crate) Q);

pub struct Read<T: ?Sized>(PhantomData<T>);
pub struct Write<T: ?Sized>(PhantomData<T>);
pub struct Row;
pub struct Table;
pub struct ReadWith(Meta);
pub struct WriteWith(Meta);

impl Module {
    pub const fn new() -> Self {
        Self(())
    }
}

impl<Q: Boba> Module<Q> {
    pub fn read<T: 'static>(self) -> Module<Q::Out>
    where
        Q: Push<Read<T>>,
    {
        self.push(Read(PhantomData))
    }

    pub fn read_with(self, meta: Meta) -> Module<Q::Out>
    where
        Q: Push<ReadWith>,
    {
        self.push(ReadWith(meta))
    }

    pub fn write<T: 'static>(self) -> Module<Q::Out>
    where
        Q: Push<Write<T>>,
    {
        self.push(Write(PhantomData))
    }

    // pub fn write_with(self, meta: Meta) -> Build<(WriteWith, Q)> {
    //     Build((WriteWith(meta), self.0))
    // }

    fn push<R: Boba>(self, query: R) -> Module<Q::Out>
    where
        Q: Push<R>,
    {
        Module(self.0.push(query))
    }
}

impl<Q: Boba> module::Module for Module<Q> {
    type Item<'a>
        = Query<'a, Q>
    where
        Self: 'a;
    type State = (usize, Vec<(u32, Q::State)>);

    fn initialize(&self, _: &mut Store) -> Result<Self::State, Error> {
        Ok((0, Vec::new()))
    }

    fn update(&self, state: &mut Self::State, store: &Store) -> Result<bool, Error> {
        let count = state.1.len();
        while let Some(table) = store.tables.get(state.0) {
            state.0 += 1;
            if let Some(query) = self.0.initialize(table) {
                state.1.push((table.index(), query));
            }
        }
        Ok(count < state.1.len())
    }

    fn get<'a>(&'a self, state: &'a Self::State, store: &'a Store) -> Self::Item<'a>
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

impl<'a, Q: Boba> iter::Iterator for Iterator<'a, Q> {
    type Item = Q::Item<'a>;

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

impl<'a, Q: Boba> Query<'a, Q> {
    // pub fn tables(&self) -> impl iter::Iterator<Item = &Table> {
    //     self.states
    //         .iter()
    //         .map(|(table, _)| unsafe { self.tables.get_unchecked(*table as usize)
    // }) }

    // pub fn count(&mut self) -> usize {
    //     self.tables().map(|table| table.count() as usize).sum()
    // }

    pub fn iter(&mut self) -> Iterator<'_, Q> {
        Iterator {
            query: self.query,
            states: self.states.iter(),
            tables: self.tables,
        }
    }
}

impl Boba for () {
    type Item<'a>
        = ()
    where
        Self: 'a;
    type State = ();

    fn initialize(&self, _: &table::Table) -> Option<Self::State> {
        Some(())
    }

    fn get<'a>(&self, _: &'a Self::State, _: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
    }
}

impl<T0: Boba, T1: Boba> Boba for (T0, T1) {
    type Item<'a>
        = (T0::Item<'a>, T1::Item<'a>)
    where
        Self: 'a;
    type State = (T0::State, T1::State);

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

impl<T: 'static> Boba for Read<T> {
    type Item<'a>
        = &'a [T]
    where
        Self: 'a;
    type State = u32;

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

impl<T: 'static> Boba for Write<T> {
    type Item<'a>
        = &'a mut [T]
    where
        Self: 'a;
    type State = u32;

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

impl Boba for Row {
    type Item<'a>
        = Rows<'a>
    where
        Self: 'a;
    type State = ();

    fn initialize(&self, _: &table::Table) -> Option<Self::State> {
        Some(())
    }

    fn get<'a>(&'a self, _: &Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        Rows::new(0..table.count(), table.index())
    }
}

impl Boba for Table {
    type Item<'a>
        = &'a table::Table
    where
        Self: 'a;
    type State = ();

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

impl Boba for ReadWith {
    type Item<'a>
        = &'a column::Column
    where
        Self: 'a;
    type State = u32;

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

//     fn initialize(&self, table: &table::Table) -> Option<Self::State> {
//         Some(table.column(self.0.identifier)?.index())
//     }

//     fn get<'a>(&'a self, state: &Self::State, table: &'a table::Table) ->
// Self::Item<'a> where
// Self: 'a {         unsafe { table.columns.get_unchecked(*state as
// usize) }     }
// }
