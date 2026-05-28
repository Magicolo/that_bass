use crate::v4::{
    Error, Meta, Rows, Store,
    module::{Access, Dependency, Module, Resource},
    table,
    utility::{IntoFlat, Push},
};
use core::{
    any::TypeId,
    iter::{empty, once},
    marker::PhantomData,
    ptr::NonNull,
    slice,
};

pub trait Query {
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
    fn get<'a>(&'a self, state: &'a mut Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a;
}

pub trait Filter {
    fn filter(&self, table: &table::Table) -> bool;
}

pub struct Item<'a, Q: Query> {
    query: &'a Q,
    states: &'a mut [(u32, Q::State)],
    tables: &'a [table::Table],
}

pub struct Iter<'a, Q: Query> {
    query: &'a Q,
    states: slice::IterMut<'a, (u32, Q::State)>,
    tables: &'a [table::Table],
}

pub struct Build<Q = (), F = ()>(Q, F);
pub struct State<S> {
    count: usize,
    states: Vec<(u32, S)>,
}

pub struct Key(());
pub struct Row(());
pub struct Table(());
pub struct Read<T: ?Sized>(PhantomData<T>);
pub struct Write<T: ?Sized>(PhantomData<T>);
pub struct ReadWith(Meta);
pub struct WriteWith(Meta);
pub struct Has<T: ?Sized>(PhantomData<T>);
pub struct HasWith(Meta);
pub struct Not<T: ?Sized>(PhantomData<T>);
pub struct NotWith(Meta);
pub struct ColumnRef<'a>(&'a Meta, &'a NonNull<u8>);
pub struct ColumnMut<'a>(&'a Meta, &'a NonNull<u8>);

impl<Q, F> Build<Q, F> {
    pub fn key(self) -> Build<Q::Out, F>
    where
        Q: Push<Key>,
    {
        self.push_query(Key(()))
    }

    pub fn row(self) -> Build<Q::Out, F>
    where
        Q: Push<Row>,
    {
        self.push_query(Row(()))
    }

    pub fn table(self) -> Build<Q::Out, F>
    where
        Q: Push<Table>,
    {
        self.push_query(Table(()))
    }

    pub fn read<T: 'static>(self) -> Build<Q::Out, F>
    where
        Q: Push<Read<T>>,
    {
        self.push_query(Read(PhantomData))
    }

    pub fn read_with(self, meta: Meta) -> Build<Q::Out, F>
    where
        Q: Push<ReadWith>,
    {
        self.push_query(ReadWith(meta))
    }

    pub fn write<T: 'static>(self) -> Build<Q::Out, F>
    where
        Q: Push<Write<T>>,
    {
        self.push_query(Write(PhantomData))
    }

    pub fn write_with(self, meta: Meta) -> Build<Q::Out, F>
    where
        Q: Push<WriteWith>,
    {
        self.push_query(WriteWith(meta))
    }

    pub fn has<T: 'static>(self) -> Build<Q, F::Out>
    where
        F: Push<Has<T>>,
    {
        self.push_filter(Has(PhantomData))
    }

    pub fn has_with(self, meta: Meta) -> Build<Q, F::Out>
    where
        F: Push<HasWith>,
    {
        self.push_filter(HasWith(meta))
    }

    pub fn not<T: 'static>(self) -> Build<Q, F::Out>
    where
        F: Push<Not<T>>,
    {
        self.push_filter(Not(PhantomData))
    }

    pub fn not_with(self, meta: Meta) -> Build<Q, F::Out>
    where
        F: Push<NotWith>,
    {
        self.push_filter(NotWith(meta))
    }

    fn push_query<R>(self, query: R) -> Build<Q::Out, F>
    where
        Q: Push<R>,
    {
        Build(self.0.push(query), self.1)
    }

    fn push_filter<G>(self, filter: G) -> Build<Q, F::Out>
    where
        F: Push<G>,
    {
        Build(self.0, self.1.push(filter))
    }
}

impl<Q: Query, F: Filter> Module for Build<Q, F> {
    type Item<'a>
        = Item<'a, Q>
    where
        Self: 'a;
    type State = State<Q::State>;

    fn declare(&self, state: &Self::State, store: &Store) -> impl Iterator<Item = Dependency> {
        state.states.iter().flat_map(|(table, state)| {
            let table = unsafe { store.tables.get_unchecked(*table as usize) };
            self.0.declare(state, table)
        })
    }

    fn initialize(&self, _: &mut Store) -> Result<Self::State, Error> {
        Ok(State {
            count: 0,
            states: Vec::new(),
        })
    }

    fn update(&self, state: &mut Self::State, store: &mut Store) -> Result<bool, Error> {
        let count = state.count;
        while let Some(table) = store.tables.get(state.count) {
            state.count += 1;
            if self.1.filter(table)
                && let Some(query) = self.0.initialize(table)
            {
                state.states.push((table.index(), query));
            }
        }
        Ok(count < state.count)
    }

    fn get<'a>(&'a self, state: &'a mut Self::State, store: &'a Store) -> Self::Item<'a>
    where
        Self: 'a,
    {
        Item {
            query: &self.0,
            states: &mut state.states,
            tables: &store.tables,
        }
    }
}

impl<'a, Q: Query<Item<'a>: IntoFlat>> Iterator for Iter<'a, Q> {
    type Item = <Q::Item<'a> as IntoFlat>::Flat;

    fn next(&mut self) -> Option<Self::Item> {
        let (table, state) = self.states.next()?;
        let table = unsafe { self.tables.get_unchecked(*table as usize) };
        Some(self.query.get(state, table).into_flat())
    }
}

impl<'a, Q: Query> Item<'a, Q> {
    // pub fn tables(&self) -> impl Iterator<Item = &Table> {
    //     self.states
    //         .iter()
    //         .map(|(table, _)| unsafe { self.tables.get_unchecked(*table as usize)
    // }) }

    // pub fn count(&mut self) -> usize {
    //     self.tables().map(|table| table.count() as usize).sum()
    // }

    pub fn iter(&mut self) -> Iter<'_, Q> {
        Iter {
            query: self.query,
            states: self.states.iter_mut(),
            tables: self.tables,
        }
    }
}

impl<Q: Query> Query for &Q {
    type Item<'a>
        = Q::Item<'a>
    where
        Self: 'a;
    type State = Q::State;

    fn declare(
        &self,
        state: &Self::State,
        table: &table::Table,
    ) -> impl Iterator<Item = Dependency> {
        Q::declare(self, state, table)
    }

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Q::initialize(self, table)
    }

    fn get<'a>(&'a self, state: &'a mut Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        Q::get(self, state, table)
    }
}

impl<Q: Query> Query for &mut Q {
    type Item<'a>
        = Q::Item<'a>
    where
        Self: 'a;
    type State = Q::State;

    fn declare(
        &self,
        state: &Self::State,
        table: &table::Table,
    ) -> impl Iterator<Item = Dependency> {
        Q::declare(self, state, table)
    }

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Q::initialize(self, table)
    }

    fn get<'a>(&'a self, state: &'a mut Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        Q::get(self, state, table)
    }
}

impl Query for () {
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

    fn get<'a>(&self, _: &'a mut Self::State, _: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
    }
}

impl<A0: Query, A1: Query> Query for (A0, A1) {
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

    fn get<'a>(&'a self, state: &'a mut Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        (
            self.0.get(&mut state.0, table),
            self.1.get(&mut state.1, table),
        )
    }
}

impl<T: 'static> Query for Read<T> {
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
        table.column(TypeId::of::<T>())
    }

    fn get<'a>(&'a self, state: &'a mut Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        let column = unsafe { table.columns.get_unchecked(*state as usize) };
        unsafe { column.as_ref(table.count) }
    }
}

impl<T: 'static> Query for Write<T> {
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
        table.column(TypeId::of::<T>())
    }

    fn get<'a>(&'a self, state: &'a mut Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        let column = unsafe { table.columns.get_unchecked(*state as usize) };
        unsafe { column.as_mut(table.count) }
    }
}

// TODO: Implement
impl Query for Key {
    type Item<'a>
        = ()
    where
        Self: 'a;
    type State = ();

    fn declare(&self, _: &Self::State, _: &table::Table) -> impl Iterator<Item = Dependency> {
        empty()
    }

    fn initialize(&self, _: &table::Table) -> Option<Self::State> {
        None
    }

    fn get<'a>(&'a self, _: &'a mut Self::State, _: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
    }
}

// TODO: Implement
impl Query for Row {
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

    fn get<'a>(&'a self, _: &'a mut Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        Rows::new(0..table.count(), table.index())
    }
}

impl Query for Table {
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

    fn get<'a>(&'a self, _: &'a mut Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        table
    }
}

impl Query for ReadWith {
    type Item<'a>
        = ColumnRef<'a>
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
        table.column(self.0.identifier)
    }

    fn get<'a>(&'a self, state: &'a mut Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        let column = unsafe { table.columns.get_unchecked(*state as usize) };
        ColumnRef(&column.meta, &column.data)
    }
}

impl Query for WriteWith {
    type Item<'a>
        = ColumnMut<'a>
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
        table.column(self.0.identifier)
    }

    fn get<'a>(&'a self, state: &'a mut Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        let column = unsafe { table.columns.get_unchecked(*state as usize) };
        ColumnMut(&column.meta, &column.data)
    }
}

impl<F: Filter> Filter for &F {
    fn filter(&self, table: &table::Table) -> bool {
        F::filter(self, table)
    }
}

impl<F: Filter> Filter for &mut F {
    fn filter(&self, table: &table::Table) -> bool {
        F::filter(self, table)
    }
}

impl Filter for () {
    fn filter(&self, _: &table::Table) -> bool {
        true
    }
}

impl<F0: Filter, F1: Filter> Filter for (F0, F1) {
    fn filter(&self, table: &table::Table) -> bool {
        self.0.filter(table) && self.1.filter(table)
    }
}

impl<T: 'static> Filter for Has<T> {
    fn filter(&self, table: &table::Table) -> bool {
        table.column(TypeId::of::<T>()).is_some()
    }
}

impl Filter for HasWith {
    fn filter(&self, table: &table::Table) -> bool {
        table.column(self.0.identifier).is_some()
    }
}

impl<T: 'static> Filter for Not<T> {
    fn filter(&self, table: &table::Table) -> bool {
        table.column(TypeId::of::<T>()).is_none()
    }
}

impl Filter for NotWith {
    fn filter(&self, table: &table::Table) -> bool {
        table.column(self.0.identifier).is_none()
    }
}

pub const fn query() -> Build {
    Build((), ())
}
