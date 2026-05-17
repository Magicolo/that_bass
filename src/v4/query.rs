use core::{any::TypeId, marker::PhantomData};

use crate::v4::{At, Meta, Rows};

pub trait Query {
    type State<'a>;
    type Item<'a>;

    fn initialize<'a>(&self, table: At<'a, crate::v4::Table>) -> Option<Self::State<'a>>;
    fn get<'a>(&self, state: &Self::State<'a>) -> Self::Item<'a>;
}

pub struct Read<T: ?Sized>(PhantomData<T>);
pub struct Row;
pub struct Table;
pub struct Column(Meta);

pub const fn read<T: ?Sized>() -> Read<T> {
    Read(PhantomData)
}

pub const fn column(meta: Meta) -> Column {
    Column(meta)
}

impl<T: ?Sized> Clone for Read<T> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl<T: ?Sized> Copy for Read<T> {}

impl Column {
    pub const fn meta(&self) -> &Meta {
        &self.0
    }
}

impl<T: 'static> Query for Read<T> {
    type Item<'a> = &'a [T];
    type State<'a> = (&'a crate::v4::Table, &'a crate::v4::Column);

    fn initialize<'a>(&self, table: At<'a, crate::v4::Table>) -> Option<Self::State<'a>> {
        Some((
            table.value(),
            table.value().column(TypeId::of::<T>())?.value(),
        ))
    }

    fn get<'a>(&self, state: &Self::State<'a>) -> Self::Item<'a> {
        unsafe { state.1.get_all(state.0.count()) }
    }
}

impl Query for Row {
    type Item<'a> = Rows<'a>;
    type State<'a> = Rows<'a>;

    fn initialize<'a>(&self, table: At<'a, crate::v4::Table>) -> Option<Self::State<'a>> {
        Some(Rows::new(0..table.value().count(), table.index()))
    }

    fn get<'a>(&self, state: &Self::State<'a>) -> Self::Item<'a> {
        state.clone()
    }
}

impl Query for Table {
    type Item<'a> = &'a crate::v4::Table;
    type State<'a> = &'a crate::v4::Table;

    fn initialize<'a>(&self, table: At<'a, crate::v4::Table>) -> Option<Self::State<'a>> {
        Some(table.value())
    }

    fn get<'a>(&self, state: &Self::State<'a>) -> Self::Item<'a> {
        state
    }
}

impl Query for Column {
    type Item<'a> = &'a crate::v4::Column;
    type State<'a> = &'a crate::v4::Column;

    fn initialize<'a>(&self, table: At<'a, crate::v4::Table>) -> Option<Self::State<'a>> {
        Some(table.value().column(self.0.identifier)?.value())
    }

    fn get<'a>(&self, state: &Self::State<'a>) -> Self::Item<'a> {
        state
    }
}
