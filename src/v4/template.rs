use crate::v4::{
    Meta, Store, Table,
    buffer::Buffer,
    depend::{Depend, Dependency},
    key::Keys,
};
use core::{
    any::{Any, TypeId},
    iter::once,
    marker::PhantomData,
};

pub trait Template: Depend {
    type Item;
    type State;

    fn initialize(&self, table: &Table, store: &Store) -> Option<Self::State>;
    unsafe fn set(&self, state: &Self::State, item: Self::Item, buffer: &mut Buffer);
}

pub struct Key(pub(crate) ());
pub struct Column<T: ?Sized>(pub(crate) PhantomData<T>);
pub struct ColumnWith(pub(crate) Meta);

impl<T: Template + ?Sized> Template for &T {
    type Item = T::Item;
    type State = T::State;

    fn initialize(&self, table: &Table, store: &Store) -> Option<Self::State> {
        T::initialize(self, table, store)
    }

    #[inline]
    unsafe fn set(&self, state: &Self::State, item: Self::Item, buffer: &mut Buffer) {
        unsafe { T::set(self, state, item, buffer) }
    }
}

impl<T: Template + ?Sized> Template for &mut T {
    type Item = T::Item;
    type State = T::State;

    fn initialize(&self, table: &Table, store: &Store) -> Option<Self::State> {
        T::initialize(self, table, store)
    }

    #[inline]
    unsafe fn set(&self, state: &Self::State, item: Self::Item, buffer: &mut Buffer) {
        unsafe { T::set(self, state, item, buffer) }
    }
}

impl Template for () {
    type Item = ();
    type State = ();

    fn initialize(&self, _: &Table, _: &Store) -> Option<Self::State> {
        Some(())
    }

    #[inline]
    unsafe fn set(&self, _: &Self::State, _: Self::Item, _: &mut Buffer) {}
}

impl<T0: Template, T1: Template> Template for (T0, T1) {
    type Item = (T0::Item, T1::Item);
    type State = (T0::State, T1::State);

    fn initialize(&self, table: &Table, store: &Store) -> Option<Self::State> {
        Some((
            self.0.initialize(table, store)?,
            self.1.initialize(table, store)?,
        ))
    }

    unsafe fn set(&self, state: &Self::State, item: Self::Item, buffer: &mut Buffer) {
        unsafe { self.0.set(&state.0, item.0, buffer) };
        unsafe { self.1.set(&state.1, item.1, buffer) };
    }
}

unsafe impl Depend for Key {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(Dependency::write(Meta::key()))
    }
}

impl Template for Key {
    type Item = ();
    type State = Keys;

    fn initialize(&self, table: &Table, store: &Store) -> Option<Self::State> {
        if table.columns().get(0)?.meta().is_key() {
            Some(store.keys().clone())
        } else {
            None
        }
    }

    #[inline]
    unsafe fn set(&self, _: &Self::State, _: Self::Item, _: &mut Buffer) {
        todo!()
    }
}

unsafe impl Depend for ColumnWith {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(Dependency::write(self.0))
    }
}

impl Template for ColumnWith {
    type Item = Box<dyn Any>;
    type State = u32;

    fn initialize(&self, table: &Table, _: &Store) -> Option<Self::State> {
        table.column(self.0.identifier())
    }

    #[inline]
    unsafe fn set(&self, state: &Self::State, item: Self::Item, buffer: &mut Buffer) {
        unsafe { buffer.set(*state, item) };
    }
}

unsafe impl<T: 'static> Depend for Column<T> {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(Dependency::write(Meta::of::<T>()))
    }
}

impl<T: 'static> Template for Column<T> {
    type Item = T;
    type State = u32;

    fn initialize(&self, table: &Table, _: &Store) -> Option<Self::State> {
        table.column(TypeId::of::<T>())
    }

    #[inline]
    unsafe fn set(&self, state: &Self::State, item: Self::Item, buffer: &mut Buffer) {
        unsafe { buffer.set(*state, item) };
    }
}
