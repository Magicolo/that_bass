use crate::v4::{Meta, Table};
use core::{
    any::{Any, TypeId},
    iter::{empty, once},
    marker::PhantomData,
};

pub trait Template {
    type Item;
    type State;

    fn declare(&self) -> impl Iterator<Item = Meta>;
    fn initialize(&self, table: &Table) -> Option<Self::State>;
    unsafe fn apply(&self, state: &Self::State, item: Self::Item, index: u32, table: &Table);
}

pub struct Key(pub(crate) ());
pub struct Column<T: ?Sized>(pub(crate) PhantomData<T>);
pub struct ColumnWith(pub(crate) Meta);

impl<T: Template + ?Sized> Template for &T {
    type Item = T::Item;
    type State = T::State;

    fn declare(&self) -> impl Iterator<Item = Meta> {
        T::declare(self)
    }

    fn initialize(&self, table: &Table) -> Option<Self::State> {
        T::initialize(self, table)
    }

    unsafe fn apply(&self, state: &Self::State, item: Self::Item, index: u32, table: &Table) {
        unsafe { T::apply(self, state, item, index, table) }
    }
}

impl<T: Template + ?Sized> Template for &mut T {
    type Item = T::Item;
    type State = T::State;

    fn declare(&self) -> impl Iterator<Item = Meta> {
        T::declare(self)
    }

    fn initialize(&self, table: &Table) -> Option<Self::State> {
        T::initialize(self, table)
    }

    unsafe fn apply(&self, state: &Self::State, item: Self::Item, index: u32, table: &Table) {
        unsafe { T::apply(self, state, item, index, table) }
    }
}

impl Template for () {
    type Item = ();
    type State = ();

    fn declare(&self) -> impl Iterator<Item = Meta> {
        empty()
    }

    fn initialize(&self, _: &Table) -> Option<Self::State> {
        Some(())
    }

    unsafe fn apply(&self, _: &Self::State, _: Self::Item, _: u32, _: &Table) {}
}

impl<T0: Template, T1: Template> Template for (T0, T1) {
    type Item = (T0::Item, T1::Item);
    type State = (T0::State, T1::State);

    fn declare(&self) -> impl Iterator<Item = Meta> {
        self.0.declare().chain(self.1.declare())
    }

    fn initialize(&self, table: &Table) -> Option<Self::State> {
        Some((self.0.initialize(table)?, self.1.initialize(table)?))
    }

    unsafe fn apply(&self, state: &Self::State, item: Self::Item, index: u32, table: &Table) {
        unsafe {
            self.0.apply(&state.0, item.0, index, table);
            self.1.apply(&state.1, item.1, index, table);
        }
    }
}

// TODO: Implement this when `Keys` will be implemented.
impl Template for Key {
    type Item = ();
    type State = ();

    fn declare(&self) -> impl Iterator<Item = Meta> {
        empty()
    }

    fn initialize(&self, table: &Table) -> Option<Self::State> {
        None
    }

    unsafe fn apply(&self, state: &Self::State, item: Self::Item, index: u32, table: &Table) {}
}

impl Template for ColumnWith {
    type Item = Box<dyn Any>;
    type State = u32;

    fn declare(&self) -> impl Iterator<Item = Meta> {
        once(self.0.clone())
    }

    fn initialize(&self, table: &Table) -> Option<Self::State> {
        table.column(self.0.identifier())
    }

    unsafe fn apply(&self, state: &Self::State, item: Self::Item, index: u32, table: &Table) {
        unsafe {
            table
                .columns()
                .get_unchecked(*state as usize)
                .set_at(item, index)
        };
    }
}

impl<T: 'static> Template for Column<T> {
    type Item = T;
    type State = u32;

    fn declare(&self) -> impl Iterator<Item = Meta> {
        once(Meta::of::<T>())
    }

    fn initialize(&self, table: &Table) -> Option<Self::State> {
        table.column(TypeId::of::<T>())
    }

    unsafe fn apply(&self, state: &Self::State, item: Self::Item, index: u32, table: &Table) {
        unsafe {
            table
                .columns()
                .get_unchecked(*state as usize)
                .set(item, index)
        };
    }
}
