use crate::v4::{Meta, Rows, column, table};
use core::{any::TypeId, marker::PhantomData};

pub trait Query {
    type State;
    type Item<'a>;

    fn initialize(&self, table: &table::Table) -> Option<Self::State>;
    fn get<'a>(&self, state: &Self::State, table: &'a mut table::Table) -> Self::Item<'a>;
}

pub struct Read<T: ?Sized>(PhantomData<T>);
pub struct Write<T: ?Sized>(PhantomData<T>);
pub struct Row;
pub struct Table;
pub struct ReadWith(Meta);
pub struct WriteWith(Meta);

pub const fn read<T: ?Sized>() -> Read<T> {
    Read(PhantomData)
}

pub const fn write<T: ?Sized>() -> Write<T> {
    Write(PhantomData)
}

pub const fn read_with(meta: Meta) -> ReadWith {
    ReadWith(meta)
}

pub const fn write_with(meta: Meta) -> WriteWith {
    WriteWith(meta)
}

impl<T: ?Sized> Clone for Read<T> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl<T: ?Sized> Copy for Read<T> {}

impl<T: ?Sized> Clone for Write<T> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl<T: ?Sized> Copy for Write<T> {}

impl ReadWith {
    pub const fn meta(&self) -> &Meta {
        &self.0
    }
}

impl<T: 'static> Query for Read<T> {
    type Item<'a> = &'a [T];
    type State = u32;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Some(table.column(TypeId::of::<T>())?.index())
    }

    fn get<'a>(&self, state: &Self::State, table: &'a mut table::Table) -> Self::Item<'a> {
        unsafe {
            table
                .columns()
                .get_unchecked(*state as usize)
                .get_all(*state)
        }
    }
}

impl<T: 'static> Query for Write<T> {
    type Item<'a> = &'a mut [T];
    type State = u32;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Some(table.column(TypeId::of::<T>())?.index())
    }

    fn get<'a>(&self, state: &Self::State, table: &'a mut table::Table) -> Self::Item<'a> {
        unsafe {
            table
                .columns_mut()
                .get_unchecked_mut(*state as usize)
                .get_all_mut(*state)
        }
    }
}

impl Query for Row {
    type Item<'a> = Rows<'a>;
    type State = ();

    fn initialize(&self, _: &table::Table) -> Option<Self::State> {
        Some(())
    }

    fn get<'a>(&self, _: &Self::State, table: &'a mut table::Table) -> Self::Item<'a> {
        Rows::new(0..table.count(), table.index())
    }
}

impl Query for Table {
    type Item<'a> = &'a table::Table;
    type State = ();

    fn initialize(&self, _: &table::Table) -> Option<Self::State> {
        Some(())
    }

    fn get<'a>(&self, _: &Self::State, table: &'a mut table::Table) -> Self::Item<'a> {
        table
    }
}

impl Query for ReadWith {
    type Item<'a> = &'a column::Column;
    type State = u32;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Some(table.column(self.0.identifier)?.index())
    }

    fn get<'a>(&self, state: &Self::State, table: &'a mut table::Table) -> Self::Item<'a> {
        unsafe { table.columns().get_unchecked(*state as usize) }
    }
}

impl Query for WriteWith {
    type Item<'a> = &'a mut column::Column;
    type State = u32;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Some(table.column(self.0.identifier)?.index())
    }

    fn get<'a>(&self, state: &Self::State, table: &'a mut table::Table) -> Self::Item<'a> {
        unsafe { table.columns_mut().get_unchecked_mut(*state as usize) }
    }
}
