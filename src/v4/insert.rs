use crate::v4::{
    Error, Meta, Store, Table, Vector,
    module::{self, Dependency, Module},
    utility::{IntoNest, Push},
};
use core::{
    any::{Any, TypeId},
    iter::{empty, once},
    marker::PhantomData,
    mem::take,
    ptr::NonNull,
};

pub trait Template {
    type Item;
    type State;

    fn declare(&self) -> impl Iterator<Item = Meta>;
    fn initialize(&self, table: &mut Table) -> Option<Self::State>;
    fn defer(&self, state: &mut Self::State, item: Self::Item) -> bool;
    unsafe fn resolve(&self, state: &mut Self::State, table: &Table) -> bool;
}

pub struct Build<T = ()>(T);
pub struct State<T: Template = ()> {
    template: T,
    count: u32,
    table: u32,
    state: T::State,
}

pub struct Insert<'a, T: Template = ()>(&'a mut State<T>);

pub struct Key;
pub struct Column<T: ?Sized>(PhantomData<T>);
pub struct ColumnWith(Meta);

impl Store {
    pub fn insert<T: Template>(
        &mut self,
        insert: Build<T>,
    ) -> Result<super::State<State<T>>, Error> {
        let template = insert.0;
        let table = self.find_or_insert_table(template.declare())?;
        let state = template
            .initialize(unsafe { self.tables.get_unchecked_mut(table as usize) })
            .ok_or(Error::FailedToInitialize)?;
        Ok(self.state(State {
            template,
            count: 0,
            table,
            state,
        }))
    }
}

impl<T: Template> Insert<'_, T> {
    pub fn one<N: IntoNest<Nest = T::Item>>(&mut self, item: N) {
        self.0.template.defer(&mut self.0.state, item.into_nest());
        self.0.count += 1;
    }
}

impl<T: Template> Build<T> {
    pub fn key(self) -> Build<T::Out>
    where
        T: Push<Key>,
    {
        Build(self.0.push(Key))
    }

    pub fn column<C: 'static>(self) -> Build<T::Out>
    where
        T: Push<Column<C>>,
    {
        Build(self.0.push(Column(PhantomData)))
    }

    pub fn column_with(self, meta: Meta) -> Build<T::Out>
    where
        T: Push<ColumnWith>,
    {
        Build(self.0.push(ColumnWith(meta)))
    }
}

impl<T: Template> Module for State<T> {
    type Item<'a>
        = Insert<'a, T>
    where
        Self: 'a;

    fn declare(&self, _: &Store) -> impl Iterator<Item = Dependency> {
        empty()
    }

    fn update(&mut self, _: &mut Store) -> Result<bool, Error> {
        Ok(false)
    }

    fn resolve(&mut self, store: &mut Store) -> Result<(), Error> {
        let count = take(&mut self.count);
        if count > 0 {
            let table = unsafe { store.tables.get_unchecked_mut(self.table as usize) };
            table.reserve(count)?;
            table.ensure()?;
            unsafe { self.template.resolve(&mut self.state, &*table) };
            table.commit();
        }
        Ok(())
    }

    fn get<'a>(&'a mut self, _: &'a Store) -> Self::Item<'a>
    where
        Self: 'a,
    {
        Insert(self)
    }
}

impl<T: Template> Template for &T {
    type Item = T::Item;
    type State = T::State;

    fn declare(&self) -> impl Iterator<Item = Meta> {
        T::declare(self)
    }

    fn initialize(&self, table: &mut Table) -> Option<Self::State> {
        T::initialize(self, table)
    }

    fn defer(&self, state: &mut Self::State, item: Self::Item) -> bool {
        T::defer(self, state, item)
    }

    unsafe fn resolve(&self, state: &mut Self::State, table: &Table) -> bool {
        unsafe { T::resolve(self, state, table) }
    }
}

impl<T: Template> Template for &mut T {
    type Item = T::Item;
    type State = T::State;

    fn declare(&self) -> impl Iterator<Item = Meta> {
        T::declare(self)
    }

    fn initialize(&self, table: &mut Table) -> Option<Self::State> {
        T::initialize(self, table)
    }

    fn defer(&self, state: &mut Self::State, item: Self::Item) -> bool {
        T::defer(self, state, item)
    }

    unsafe fn resolve(&self, state: &mut Self::State, table: &Table) -> bool {
        unsafe { T::resolve(self, state, table) }
    }
}

impl Template for () {
    type Item = ();
    type State = ();

    fn declare(&self) -> impl Iterator<Item = Meta> {
        empty()
    }

    fn initialize(&self, _: &mut Table) -> Option<Self::State> {
        Some(())
    }

    fn defer(&self, _: &mut Self::State, _: Self::Item) -> bool {
        true
    }

    unsafe fn resolve(&self, _: &mut Self::State, _: &Table) -> bool {
        false
    }
}

impl<T0: Template, T1: Template> Template for (T0, T1) {
    type Item = (T0::Item, T1::Item);
    type State = (T0::State, T1::State);

    fn declare(&self) -> impl Iterator<Item = Meta> {
        self.0.declare().chain(self.1.declare())
    }

    fn initialize(&self, table: &mut Table) -> Option<Self::State> {
        Some((self.0.initialize(table)?, self.1.initialize(table)?))
    }

    fn defer(&self, state: &mut Self::State, item: Self::Item) -> bool {
        self.0.defer(&mut state.0, item.0) && self.1.defer(&mut state.1, item.1)
    }

    unsafe fn resolve(&self, state: &mut Self::State, table: &Table) -> bool {
        unsafe { self.0.resolve(&mut state.0, table) && self.1.resolve(&mut state.1, table) }
    }
}

// TODO: Implement this when `Keys` will be implemented.
impl Template for Key {
    type Item = ();
    type State = ();

    fn declare(&self) -> impl Iterator<Item = Meta> {
        empty()
    }

    fn initialize(&self, table: &mut Table) -> Option<Self::State> {
        None
    }

    fn defer(&self, state: &mut Self::State, item: Self::Item) -> bool {
        true
    }

    unsafe fn resolve(&self, state: &mut Self::State, table: &Table) -> bool {
        true
    }
}

impl Template for ColumnWith {
    type Item = Box<dyn Any>;
    type State = (Vector, u32);

    fn declare(&self) -> impl Iterator<Item = Meta> {
        once(self.0.clone())
    }

    fn initialize(&self, table: &mut Table) -> Option<Self::State> {
        Some((
            Vector::new(self.0.clone()),
            table.column(self.0.identifier)?,
        ))
    }

    fn defer(&self, state: &mut Self::State, item: Self::Item) -> bool {
        state.0.push(item).is_ok()
    }

    unsafe fn resolve(&self, state: &mut Self::State, table: &Table) -> bool {
        let count = table.count();
        let column = unsafe { table.columns().get_unchecked(state.1 as usize) };
        debug_assert_eq!(self.0.identifier, column.meta.identifier);
        unsafe { state.0.move_at(column.data, count) }
    }
}

impl<T: 'static> Template for Column<T> {
    type Item = T;
    type State = (Vec<Self::Item>, u32);

    fn declare(&self) -> impl Iterator<Item = Meta> {
        once(Meta::of::<T>())
    }

    fn initialize(&self, table: &mut Table) -> Option<Self::State> {
        Some((Vec::new(), table.column(TypeId::of::<T>())?))
    }

    fn defer(&self, state: &mut Self::State, item: Self::Item) -> bool {
        state.0.push(item);
        true
    }

    unsafe fn resolve(&self, state: &mut Self::State, table: &Table) -> bool {
        if let Some(source) = NonNull::new(state.0.as_mut_ptr()) {
            if let Ok(count) = state.0.len().try_into() {
                let index = table.count();
                let column = unsafe { table.columns().get_unchecked(state.1 as usize) };
                if unsafe { column.copy(source, index, count) } {
                    unsafe { state.0.set_len(0) };
                    return true;
                }
            }
        }
        false
    }
}

pub const fn insert() -> Build {
    Build(())
}
