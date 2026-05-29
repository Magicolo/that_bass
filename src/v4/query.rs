use crate::v4::{
    Error, Meta, Rows, Store,
    module::{self, Access, Dependency, IntoModule, Resource},
    slice::Slice,
    table,
    utility::{IntoFlat, Push},
};
use core::{
    any::TypeId,
    iter::{empty, once},
    marker::PhantomData,
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
        store: &Store,
    ) -> impl Iterator<Item = Dependency>;
    fn initialize(&self, table: &table::Table, store: &Store) -> Option<Self::State>;
    fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        table: &'a table::Table,
        store: &'a Store,
    ) -> Self::Item<'a>
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
    store: &'a Store,
}

pub struct Tables<'a, Q: Query> {
    states: slice::Iter<'a, (u32, Q::State)>,
    tables: &'a [table::Table],
}

pub struct Columns<'a, Q: Query> {
    query: &'a Q,
    states: slice::IterMut<'a, (u32, Q::State)>,
    tables: &'a [table::Table],
    store: &'a Store,
}

pub struct Build<Q = (), F = ()>(Q, F);
pub struct Module<Q: Query, F: Filter> {
    query: Q,
    filter: F,
    count: usize,
    states: Vec<(u32, Q::State)>,
}

pub struct Key(());
pub struct Row(());
pub struct Table(());
pub struct Try<Q: ?Sized>(Q);
pub struct Read<T: ?Sized>(PhantomData<T>);
pub struct Write<T: ?Sized>(PhantomData<T>);
pub struct ReadWith(Meta);
pub struct WriteWith(Meta);
pub struct Has<T: ?Sized>(PhantomData<T>);
pub struct HasWith(Meta);
pub struct Not<F: ?Sized>(F);

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

    pub fn try_read<T: 'static>(self) -> Build<Q::Out, F>
    where
        Q: Push<Try<Read<T>>>,
    {
        self.push_query(Try(Read(PhantomData)))
    }

    pub fn read<T: 'static>(self) -> Build<Q::Out, F>
    where
        Q: Push<Read<T>>,
    {
        self.push_query(Read(PhantomData))
    }

    pub fn try_read_with(self, meta: Meta) -> Build<Q::Out, F>
    where
        Q: Push<Try<ReadWith>>,
    {
        self.push_query(Try(ReadWith(meta)))
    }

    pub fn read_with(self, meta: Meta) -> Build<Q::Out, F>
    where
        Q: Push<ReadWith>,
    {
        self.push_query(ReadWith(meta))
    }

    pub fn try_write<T: 'static>(self) -> Build<Q::Out, F>
    where
        Q: Push<Try<Write<T>>>,
    {
        self.push_query(Try(Write(PhantomData)))
    }

    pub fn try_write_with(self, meta: Meta) -> Build<Q::Out, F>
    where
        Q: Push<Try<WriteWith>>,
    {
        self.push_query(Try(WriteWith(meta)))
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
        F: Push<Not<Has<T>>>,
    {
        self.push_filter(Not(Has(PhantomData)))
    }

    pub fn not_with(self, meta: Meta) -> Build<Q, F::Out>
    where
        F: Push<Not<HasWith>>,
    {
        self.push_filter(Not(HasWith(meta)))
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

impl<Q: Query, F: Filter> IntoModule for Build<Q, F> {
    type Module = Module<Q, F>;

    fn into_module(self, _: &mut Store) -> Result<Self::Module, Error> {
        Ok(Module {
            query: self.0,
            filter: self.1,
            count: 0,
            states: Vec::new(),
        })
    }
}

impl<Q: Query, F: Filter> module::Module for Module<Q, F> {
    type Item<'a>
        = Item<'a, Q>
    where
        Self: 'a;

    fn declare(&self, store: &Store) -> impl Iterator<Item = Dependency> {
        self.states.iter().flat_map(|(table, state)| {
            let table = unsafe { store.tables.get_unchecked(*table as usize) };
            self.query.declare(state, table, store)
        })
    }

    fn update(&mut self, store: &mut Store) -> Result<bool, Error> {
        let count = self.count;
        while let Some(table) = store.tables.get(self.count) {
            self.count += 1;
            if self.filter.filter(table)
                && let Some(query) = self.query.initialize(table, store)
            {
                self.states.push((table.index(), query));
            }
        }
        Ok(count < self.count)
    }

    fn resolve(&mut self, _: &mut Store) -> Result<(), Error> {
        Ok(())
    }

    fn get<'a>(&'a mut self, store: &'a Store) -> Self::Item<'a>
    where
        Self: 'a,
    {
        Item {
            query: &self.query,
            states: &mut self.states,
            tables: &store.tables,
            store: &store,
        }
    }
}

impl<'a, Q: Query> Iterator for Tables<'a, Q> {
    type Item = &'a table::Table;

    fn next(&mut self) -> Option<Self::Item> {
        let (table, _) = self.states.next()?;
        Some(unsafe { self.tables.get_unchecked(*table as usize) })
    }
}

impl<'a, Q: Query> Iterator for Columns<'a, Q> {
    type Item = Q::Item<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let (table, state) = self.states.next()?;
        let table = unsafe { self.tables.get_unchecked(*table as usize) };
        Some(self.query.get(state, table, self.store))
    }
}

impl<'a, Q: Query> Item<'a, Q> {
    pub fn tables(&self) -> Tables<'_, Q> {
        Tables {
            states: self.states.iter(),
            tables: self.tables,
        }
    }

    pub fn columns(&mut self) -> Columns<'_, Q> {
        Columns {
            query: self.query,
            states: self.states.iter_mut(),
            tables: self.tables,
            store: self.store,
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
        store: &Store,
    ) -> impl Iterator<Item = Dependency> {
        Q::declare(self, state, table, store)
    }

    fn initialize(&self, table: &table::Table, store: &Store) -> Option<Self::State> {
        Q::initialize(self, table, store)
    }

    fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        table: &'a table::Table,
        store: &'a Store,
    ) -> Self::Item<'a>
    where
        Self: 'a,
    {
        Q::get(self, state, table, store)
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
        store: &Store,
    ) -> impl Iterator<Item = Dependency> {
        Q::declare(self, state, table, store)
    }

    fn initialize(&self, table: &table::Table, store: &Store) -> Option<Self::State> {
        Q::initialize(self, table, store)
    }

    fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        table: &'a table::Table,
        store: &'a Store,
    ) -> Self::Item<'a>
    where
        Self: 'a,
    {
        Q::get(self, state, table, store)
    }
}

impl Query for () {
    type Item<'a>
        = ()
    where
        Self: 'a;
    type State = ();

    fn declare(
        &self,
        _: &Self::State,
        _: &table::Table,
        _: &Store,
    ) -> impl Iterator<Item = Dependency> {
        empty()
    }

    fn initialize(&self, _: &table::Table, _: &Store) -> Option<Self::State> {
        Some(())
    }

    fn get<'a>(&self, _: &'a mut Self::State, _: &'a table::Table, _: &'a Store) -> Self::Item<'a>
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
        store: &Store,
    ) -> impl Iterator<Item = Dependency> {
        self.0
            .declare(&state.0, table, store)
            .chain(self.1.declare(&state.1, table, store))
    }

    fn initialize(&self, table: &table::Table, store: &Store) -> Option<Self::State> {
        Some((
            self.0.initialize(table, store)?,
            self.1.initialize(table, store)?,
        ))
    }

    fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        table: &'a table::Table,
        store: &'a Store,
    ) -> Self::Item<'a>
    where
        Self: 'a,
    {
        (
            self.0.get(&mut state.0, table, store),
            self.1.get(&mut state.1, table, store),
        )
    }
}

impl<Q: Query> Query for Try<Q> {
    type Item<'a>
        = Option<Q::Item<'a>>
    where
        Self: 'a;
    type State = Option<Q::State>;

    fn declare(
        &self,
        state: &Self::State,
        table: &table::Table,
        store: &Store,
    ) -> impl Iterator<Item = Dependency> {
        state
            .as_ref()
            .into_iter()
            .flat_map(|state| self.0.declare(state, table, store))
    }

    fn initialize(&self, table: &table::Table, store: &Store) -> Option<Self::State> {
        Some(self.0.initialize(table, store))
    }

    fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        table: &'a table::Table,
        store: &'a Store,
    ) -> Self::Item<'a>
    where
        Self: 'a,
    {
        Some(self.0.get(state.as_mut()?, table, store))
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
        store: &Store,
    ) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Read,
            resource: Resource::Column {
                store: store.identifier,
                table: table.index(),
                index: *state,
            },
        })
    }

    fn initialize(&self, table: &table::Table, _: &Store) -> Option<Self::State> {
        table.column(TypeId::of::<T>())
    }

    fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        table: &'a table::Table,
        _: &'a Store,
    ) -> Self::Item<'a>
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
        store: &Store,
    ) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Write,
            resource: Resource::Column {
                store: store.identifier,
                table: table.index(),
                index: *state,
            },
        })
    }

    fn initialize(&self, table: &table::Table, _: &Store) -> Option<Self::State> {
        table.column(TypeId::of::<T>())
    }

    fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        table: &'a table::Table,
        _: &'a Store,
    ) -> Self::Item<'a>
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

    fn declare(
        &self,
        _: &Self::State,
        _: &table::Table,
        _: &Store,
    ) -> impl Iterator<Item = Dependency> {
        empty()
    }

    fn initialize(&self, _: &table::Table, _: &Store) -> Option<Self::State> {
        None
    }

    fn get<'a>(
        &'a self,
        _: &'a mut Self::State,
        _: &'a table::Table,
        _: &'a Store,
    ) -> Self::Item<'a>
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
        store: &Store,
    ) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Read,
            resource: Resource::Column {
                store: store.identifier,
                table: table.index(),
                index: *state,
            },
        })
    }

    fn initialize(&self, _: &table::Table, _: &Store) -> Option<Self::State> {
        None
    }

    fn get<'a>(
        &'a self,
        _: &'a mut Self::State,
        table: &'a table::Table,
        _: &'a Store,
    ) -> Self::Item<'a>
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

    fn declare(
        &self,
        _: &Self::State,
        table: &table::Table,
        store: &Store,
    ) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Read,
            resource: Resource::Table {
                store: store.identifier,
                index: table.index(),
            },
        })
    }

    fn initialize(&self, _: &table::Table, _: &Store) -> Option<Self::State> {
        Some(())
    }

    fn get<'a>(
        &'a self,
        _: &'a mut Self::State,
        table: &'a table::Table,
        _: &'a Store,
    ) -> Self::Item<'a>
    where
        Self: 'a,
    {
        table
    }
}

impl Query for ReadWith {
    type Item<'a>
        = &'a Slice
    where
        Self: 'a;
    type State = (u32, Slice);

    fn declare(
        &self,
        state: &Self::State,
        table: &table::Table,
        store: &Store,
    ) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Read,
            resource: Resource::Column {
                store: store.identifier,
                table: table.index(),
                index: state.0,
            },
        })
    }

    fn initialize(&self, table: &table::Table, _: &Store) -> Option<Self::State> {
        Some((
            table.column(self.0.identifier)?,
            Slice::empty(self.0.clone()),
        ))
    }

    fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        table: &'a table::Table,
        _: &'a Store,
    ) -> Self::Item<'a>
    where
        Self: 'a,
    {
        let column = unsafe { table.columns.get_unchecked(state.0 as usize) };
        unsafe { state.1.set_parts(column.data.cast(), table.count() as _) };
        &state.1
    }
}

impl Query for WriteWith {
    type Item<'a>
        = &'a mut Slice
    where
        Self: 'a;
    type State = (u32, Slice);

    fn declare(
        &self,
        state: &Self::State,
        table: &table::Table,
        store: &Store,
    ) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Write,
            resource: Resource::Column {
                store: store.identifier,
                table: table.index(),
                index: state.0,
            },
        })
    }

    fn initialize(&self, table: &table::Table, _: &Store) -> Option<Self::State> {
        Some((
            table.column(self.0.identifier)?,
            Slice::empty(self.0.clone()),
        ))
    }

    fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        table: &'a table::Table,
        _: &'a Store,
    ) -> Self::Item<'a>
    where
        Self: 'a,
    {
        let column = unsafe { table.columns.get_unchecked(state.0 as usize) };
        unsafe { state.1.set_parts(column.data.cast(), table.count() as _) };
        &mut state.1
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

impl<F: Filter> Filter for Not<F> {
    fn filter(&self, table: &table::Table) -> bool {
        !self.0.filter(table)
    }
}

pub const fn query() -> Build {
    Build((), ())
}
