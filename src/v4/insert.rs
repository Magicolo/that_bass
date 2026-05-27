use crate::v4::{
    Error, Meta, Store, Table, Vector,
    module::{self, Dependency},
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

pub struct Module<T = ()>(pub T);

pub struct Insert<'a, T: Template = ()> {
    template: &'a T,
    state: &'a mut (u32, u32, T::State),
}

pub struct Key;
pub struct Column<T: ?Sized>(PhantomData<T>);
pub struct ColumnWith(Meta);

impl Insert<'_> {
    pub const fn build() -> Module {
        Module(())
    }
}

impl<T: Template> Insert<'_, T> {
    pub fn one<N: IntoNest<Nest = T::Item>>(&mut self, item: N) {
        self.template.defer(&mut self.state.2, item.into_nest());
        self.state.1 += 1;
    }
}

impl<T: Template> Module<T> {
    pub fn key(self) -> Module<T::Out>
    where
        T: Push<Key>,
    {
        Module(self.0.push(Key))
    }

    pub fn column<C: 'static>(self) -> Module<T::Out>
    where
        T: Push<Column<C>>,
    {
        Module(self.0.push(Column(PhantomData)))
    }

    pub fn column_with(self, meta: Meta) -> Module<T::Out>
    where
        T: Push<ColumnWith>,
    {
        Module(self.0.push(ColumnWith(meta)))
    }
}

impl<T: Template> module::Module for Module<T> {
    type Item<'a>
        = Insert<'a, T>
    where
        Self: 'a;
    type State = (u32, u32, T::State);

    fn declare(&self, _: &Self::State, _: &Store) -> impl Iterator<Item = Dependency> {
        empty()
    }

    fn initialize(&self, store: &mut Store) -> Result<Self::State, Error> {
        let table = store.find_or_insert_table(self.0.declare())?;
        let state = self
            .0
            .initialize(unsafe { store.tables.get_unchecked_mut(table as usize) })
            .ok_or(Error::FailedToInitialize)?;
        Ok((table, 0, state))
    }

    fn update(&self, _: &mut Self::State, _: &mut Store) -> Result<bool, Error> {
        Ok(false)
    }

    fn get<'a>(&'a self, state: &'a mut Self::State, _: &'a Store) -> Self::Item<'a>
    where
        Self: 'a,
    {
        Insert {
            template: &self.0,
            state,
        }
    }

    fn resolve(&self, state: &mut Self::State, store: &mut Store) -> Result<(), Error> {
        let count = take(&mut state.1);
        if count > 0 {
            let table = unsafe { store.tables.get_unchecked_mut(state.0 as usize) };
            table.reserve(count)?;
            table.ensure()?;
            unsafe { self.0.resolve(&mut state.2, &*table) };
            table.commit();
        }
        Ok(())
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
            table.column(self.0.identifier)?.index(),
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
        Some((Vec::new(), table.column(TypeId::of::<T>())?.index()))
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
