use crate::v4::{
    Meta, Rows,
    depend::{Access, Depend, Dependency, Resource},
    slice::Slice,
    table,
};
use core::{
    any::TypeId,
    iter::{empty, once},
    marker::PhantomData,
};

pub trait Item: Depend {
    type State;
    type Item<'a>
    where
        Self: 'a;

    fn initialize(&self, table: &table::Table) -> Option<Self::State>;
    fn get<'a>(&'a self, state: &'a mut Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a;
}

pub struct Key(pub(crate) ());
pub struct Row(pub(crate) ());
pub struct Table(pub(crate) ());
pub struct Try<I: ?Sized>(pub(crate) I);
pub struct Read<T: ?Sized>(pub(crate) PhantomData<T>);
pub struct Write<T: ?Sized>(pub(crate) PhantomData<T>);
pub struct ReadWith(pub(crate) Meta);
pub struct WriteWith(pub(crate) Meta);

unsafe impl<I: Item> Depend for Try<I> {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        self.0.depend()
    }
}

impl<I: Item> Item for Try<I> {
    type Item<'a>
        = Option<I::Item<'a>>
    where
        Self: 'a;
    type State = Option<I::State>;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Some(self.0.initialize(table))
    }

    fn get<'a>(&'a self, state: &'a mut Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        Some(self.0.get(state.as_mut()?, table))
    }
}

unsafe impl Depend for Key {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        empty()
    }
}

// TODO: Implement
impl Item for Key {
    type Item<'a>
        = ()
    where
        Self: 'a;
    type State = ();

    fn initialize(&self, _: &table::Table) -> Option<Self::State> {
        None
    }

    fn get<'a>(&'a self, _: &'a mut Self::State, _: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
    }
}

unsafe impl<T: 'static> Depend for Read<T> {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Read,
            resource: Resource::Column2(TypeId::of::<T>()),
        })
    }
}

impl<T: 'static> Item for Read<T> {
    type Item<'a>
        = &'a [T]
    where
        Self: 'a;
    type State = u32;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        table.column(TypeId::of::<T>())
    }

    fn get<'a>(&'a self, state: &'a mut Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        let column = unsafe { table.columns().get_unchecked(*state as usize) };
        unsafe { column.as_ref(table.count()) }
    }
}

unsafe impl<T: 'static> Depend for Write<T> {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Write,
            resource: Resource::Column2(TypeId::of::<T>()),
        })
    }
}

impl<T: 'static> Item for Write<T> {
    type Item<'a>
        = &'a mut [T]
    where
        Self: 'a;
    type State = u32;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        table.column(TypeId::of::<T>())
    }

    fn get<'a>(&'a self, state: &'a mut Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        let column = unsafe { table.columns().get_unchecked(*state as usize) };
        unsafe { column.as_mut(table.count()) }
    }
}

unsafe impl Depend for Row {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Read,
            resource: Resource::Table2,
        })
    }
}

// TODO: Implement
impl Item for Row {
    type Item<'a>
        = Rows<'a>
    where
        Self: 'a;
    type State = ();

    fn initialize(&self, _: &table::Table) -> Option<Self::State> {
        Some(())
    }

    fn get<'a>(&'a self, _: &'a mut Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        Rows::new(0..table.count(), table)
    }
}

unsafe impl Depend for Table {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Read,
            resource: Resource::Table2,
        })
    }
}

impl Item for Table {
    type Item<'a>
        = &'a table::Table
    where
        Self: 'a;
    type State = ();

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

unsafe impl Depend for ReadWith {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Read,
            resource: Resource::Column2(self.0.identifier()),
        })
    }
}

impl Item for ReadWith {
    type Item<'a>
        = &'a Slice
    where
        Self: 'a;
    type State = (u32, Slice);

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Some((table.column(self.0.identifier())?, Slice::empty(self.0)))
    }

    fn get<'a>(&'a self, state: &'a mut Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        let column = unsafe { table.columns().get_unchecked(state.0 as usize) };
        unsafe { state.1.set_parts(column.data().cast(), table.count() as _) };
        &state.1
    }
}

unsafe impl Depend for WriteWith {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Write,
            resource: Resource::Column2(self.0.identifier()),
        })
    }
}

impl Item for WriteWith {
    type Item<'a>
        = &'a mut Slice
    where
        Self: 'a;
    type State = (u32, Slice);

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Some((table.column(self.0.identifier())?, Slice::empty(self.0)))
    }

    fn get<'a>(&'a self, state: &'a mut Self::State, table: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
        let column = unsafe { table.columns().get_unchecked(state.0 as usize) };
        unsafe { state.1.set_parts(column.data().cast(), table.count() as _) };
        &mut state.1
    }
}
