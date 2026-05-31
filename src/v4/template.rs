use crate::v4::{Meta, Table, Vector};
use core::{
    any::{Any, TypeId},
    iter::{empty, once},
    marker::PhantomData,
    ops::Range,
    ptr::NonNull,
};

pub trait Template {
    type Item;
    type State;

    fn declare(&self) -> impl Iterator<Item = Meta>;
    fn initialize(&self, table: &Table) -> Option<Self::State>;
    fn defer(&self, state: &mut Self::State, item: Self::Item) -> bool;
    unsafe fn resolve(&self, state: &mut Self::State, rows: Range<u32>, table: &Table);
}

pub struct Key(pub(crate) ());
pub struct Column<T: ?Sized>(pub(crate) PhantomData<T>);
pub struct ColumnWith(pub(crate) Meta);

impl<T: Template> Template for &T {
    type Item = T::Item;
    type State = T::State;

    fn declare(&self) -> impl Iterator<Item = Meta> {
        T::declare(self)
    }

    fn initialize(&self, table: &Table) -> Option<Self::State> {
        T::initialize(self, table)
    }

    fn defer(&self, state: &mut Self::State, item: Self::Item) -> bool {
        T::defer(self, state, item)
    }

    unsafe fn resolve(&self, state: &mut Self::State, rows: Range<u32>, table: &Table) {
        unsafe { T::resolve(self, state, rows, table) }
    }
}

impl<T: Template> Template for &mut T {
    type Item = T::Item;
    type State = T::State;

    fn declare(&self) -> impl Iterator<Item = Meta> {
        T::declare(self)
    }

    fn initialize(&self, table: &Table) -> Option<Self::State> {
        T::initialize(self, table)
    }

    fn defer(&self, state: &mut Self::State, item: Self::Item) -> bool {
        T::defer(self, state, item)
    }

    unsafe fn resolve(&self, state: &mut Self::State, rows: Range<u32>, table: &Table) {
        unsafe { T::resolve(self, state, rows, table) }
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

    fn defer(&self, _: &mut Self::State, _: Self::Item) -> bool {
        true
    }

    unsafe fn resolve(&self, _: &mut Self::State, _: Range<u32>, _: &Table) {}
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

    fn defer(&self, state: &mut Self::State, item: Self::Item) -> bool {
        self.0.defer(&mut state.0, item.0) && self.1.defer(&mut state.1, item.1)
    }

    unsafe fn resolve(&self, state: &mut Self::State, rows: Range<u32>, table: &Table) {
        unsafe {
            self.0.resolve(&mut state.0, rows.clone(), table);
            self.1.resolve(&mut state.1, rows, table);
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

    fn defer(&self, state: &mut Self::State, item: Self::Item) -> bool {
        true
    }

    unsafe fn resolve(&self, state: &mut Self::State, rows: Range<u32>, table: &Table) {}
}

impl Template for ColumnWith {
    type Item = Box<dyn Any>;
    type State = (Vector, u32);

    fn declare(&self) -> impl Iterator<Item = Meta> {
        once(self.0.clone())
    }

    fn initialize(&self, table: &Table) -> Option<Self::State> {
        Some((
            Vector::new(self.0.clone()),
            table.column(self.0.identifier())?,
        ))
    }

    fn defer(&self, state: &mut Self::State, item: Self::Item) -> bool {
        state.0.push(item).is_ok()
    }

    unsafe fn resolve(&self, state: &mut Self::State, rows: Range<u32>, table: &Table) {
        let count = rows.end - rows.start;
        debug_assert_eq!(count, state.0.len());
        let column = unsafe { table.columns().get_unchecked(state.1 as usize) };
        debug_assert_eq!(self.0, column.meta());
        unsafe { state.0.move_at(column.data(), count) }
    }
}

impl<T: 'static> Template for Column<T> {
    type Item = T;
    type State = (Vec<Self::Item>, u32);

    fn declare(&self) -> impl Iterator<Item = Meta> {
        once(Meta::of::<T>())
    }

    fn initialize(&self, table: &Table) -> Option<Self::State> {
        Some((Vec::new(), table.column(TypeId::of::<T>())?))
    }

    fn defer(&self, state: &mut Self::State, item: Self::Item) -> bool {
        state.0.push(item);
        true
    }

    unsafe fn resolve(&self, state: &mut Self::State, rows: Range<u32>, table: &Table) {
        debug_assert_eq!(rows.len(), state.0.len());
        let count = rows.end - rows.start;
        let source = unsafe { NonNull::new_unchecked(state.0.as_mut_ptr()) };
        let column = unsafe { table.columns().get_unchecked(state.1 as usize) };
        debug_assert_eq!(column.meta().identifier(), TypeId::of::<T>());
        unsafe { column.copy(source, rows.start, count) };
        unsafe { state.0.set_len(0) };
    }
}
