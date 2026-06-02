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
use itertools::Itertools;

pub trait Item: Depend {
    type State;
    type Item<'a>
    where
        Self: 'a;

    fn initialize(&self, table: &table::Table) -> Option<Self::State>;
    fn declare(&self, state: &Self::State) -> impl Iterator<Item = (u32, Access)>;
    unsafe fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        count: u32,
        table: &'a table::Table,
    ) -> Self::Item<'a>
    where
        Self: 'a;
}

#[derive(Debug, Clone, Copy)]
pub struct Key(pub(crate) ());
#[derive(Debug, Clone, Copy)]
pub struct Row(pub(crate) ());
#[derive(Debug, Clone, Copy)]
pub struct Table(pub(crate) ());
#[derive(Debug, Clone, Copy)]
pub struct Try<I: ?Sized>(pub(crate) I);
#[derive(Debug)]
pub struct Read<T: ?Sized>(pub(crate) PhantomData<T>);
#[derive(Debug)]
pub struct Write<T: ?Sized>(pub(crate) PhantomData<T>);
#[derive(Debug, Clone, Copy)]
pub struct ReadWith(pub(crate) Meta);
#[derive(Debug, Clone, Copy)]
pub struct WriteWith(pub(crate) Meta);

impl<I: Item + ?Sized> Item for &I {
    type Item<'a>
        = I::Item<'a>
    where
        Self: 'a;
    type State = I::State;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        I::initialize(self, table)
    }

    fn declare(&self, state: &Self::State) -> impl Iterator<Item = (u32, Access)> {
        I::declare(self, state)
    }

    unsafe fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        count: u32,
        table: &'a table::Table,
    ) -> Self::Item<'a>
    where
        Self: 'a,
    {
        unsafe { I::get(self, state, count, table) }
    }
}

impl<I: Item + ?Sized> Item for &mut I {
    type Item<'a>
        = I::Item<'a>
    where
        Self: 'a;
    type State = I::State;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        I::initialize(self, table)
    }

    fn declare(&self, state: &Self::State) -> impl Iterator<Item = (u32, Access)> {
        I::declare(self, state)
    }

    unsafe fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        count: u32,
        table: &'a table::Table,
    ) -> Self::Item<'a>
    where
        Self: 'a,
    {
        unsafe { I::get(self, state, count, table) }
    }
}

impl Item for () {
    type Item<'a>
        = ()
    where
        Self: 'a;
    type State = ();

    fn initialize(&self, _: &table::Table) -> Option<Self::State> {
        Some(())
    }

    fn declare(&self, _: &Self::State) -> impl Iterator<Item = (u32, Access)> {
        empty()
    }

    unsafe fn get<'a>(&self, _: &'a mut Self::State, _: u32, _: &'a table::Table) -> Self::Item<'a>
    where
        Self: 'a,
    {
    }
}

impl<I0: Item, I1: Item> Item for (I0, I1) {
    type Item<'a>
        = (I0::Item<'a>, I1::Item<'a>)
    where
        Self: 'a;
    type State = (I0::State, I1::State);

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Some((self.0.initialize(table)?, self.1.initialize(table)?))
    }

    fn declare(&self, state: &Self::State) -> impl Iterator<Item = (u32, Access)> {
        self.0.declare(&state.0).merge(self.1.declare(&state.1))
    }

    unsafe fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        count: u32,
        table: &'a table::Table,
    ) -> Self::Item<'a>
    where
        Self: 'a,
    {
        unsafe {
            (
                self.0.get(&mut state.0, count, table),
                self.1.get(&mut state.1, count, table),
            )
        }
    }
}

unsafe impl<I: Item + ?Sized> Depend for Try<I> {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        self.0.depend()
    }
}

impl<I: Item + ?Sized> Item for Try<I> {
    type Item<'a>
        = Option<I::Item<'a>>
    where
        Self: 'a;
    type State = Option<I::State>;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Some(self.0.initialize(table))
    }

    fn declare(&self, state: &Self::State) -> impl Iterator<Item = (u32, Access)> {
        state
            .as_ref()
            .into_iter()
            .flat_map(|state| self.0.declare(state))
    }

    unsafe fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        count: u32,
        table: &'a table::Table,
    ) -> Self::Item<'a>
    where
        Self: 'a,
    {
        Some(unsafe { self.0.get(state.as_mut()?, count, table) })
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

    fn declare(&self, _: &Self::State) -> impl Iterator<Item = (u32, Access)> {
        empty()
    }

    unsafe fn get<'a>(
        &'a self,
        _: &'a mut Self::State,
        _: u32,
        _: &'a table::Table,
    ) -> Self::Item<'a>
    where
        Self: 'a,
    {
    }
}

unsafe impl Depend for Row {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Read,
            resource: Resource::Table,
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

    fn declare(&self, _: &Self::State) -> impl Iterator<Item = (u32, Access)> {
        empty()
    }

    unsafe fn get<'a>(
        &'a self,
        _: &'a mut Self::State,
        _: u32,
        table: &'a table::Table,
    ) -> Self::Item<'a>
    where
        Self: 'a,
    {
        Rows::new(0..0, table)
    }
}

unsafe impl Depend for Table {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Read,
            resource: Resource::Table,
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

    fn declare(&self, _: &Self::State) -> impl Iterator<Item = (u32, Access)> {
        empty()
    }

    unsafe fn get<'a>(
        &'a self,
        _: &'a mut Self::State,
        _: u32,
        table: &'a table::Table,
    ) -> Self::Item<'a>
    where
        Self: 'a,
    {
        table
    }
}

unsafe impl<T: 'static> Depend for Read<T> {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Read,
            resource: Resource::Column(TypeId::of::<T>()),
        })
    }
}

impl<T: ?Sized> Clone for Read<T> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<T: ?Sized> Copy for Read<T> {}

impl<T: 'static> Item for Read<T> {
    type Item<'a>
        = &'a [T]
    where
        Self: 'a;
    type State = u32;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        table.column(TypeId::of::<T>())
    }

    fn declare(&self, state: &Self::State) -> impl Iterator<Item = (u32, Access)> {
        once((*state, Access::Read))
    }

    unsafe fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        count: u32,
        table: &'a table::Table,
    ) -> Self::Item<'a>
    where
        Self: 'a,
    {
        unsafe { table.columns().get_unchecked(*state as usize).get(count) }
    }
}

unsafe impl<T: 'static> Depend for Write<T> {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Write,
            resource: Resource::Column(TypeId::of::<T>()),
        })
    }
}

impl<T: ?Sized> Clone for Write<T> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<T: ?Sized> Copy for Write<T> {}

impl<T: 'static> Item for Write<T> {
    type Item<'a>
        = &'a mut [T]
    where
        Self: 'a;
    type State = u32;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        table.column(TypeId::of::<T>())
    }

    fn declare(&self, state: &Self::State) -> impl Iterator<Item = (u32, Access)> {
        once((*state, Access::Write))
    }

    unsafe fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        count: u32,
        table: &'a table::Table,
    ) -> Self::Item<'a>
    where
        Self: 'a,
    {
        unsafe {
            table
                .columns()
                .get_unchecked(*state as usize)
                .get_mut(count)
        }
    }
}

unsafe impl Depend for ReadWith {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Read,
            resource: Resource::Column(self.0.identifier()),
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

    fn declare(&self, state: &Self::State) -> impl Iterator<Item = (u32, Access)> {
        once((state.0, Access::Read))
    }

    unsafe fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        count: u32,
        table: &'a table::Table,
    ) -> Self::Item<'a>
    where
        Self: 'a,
    {
        unsafe {
            table
                .columns()
                .get_unchecked(state.0 as usize)
                .get_in(&mut state.1, count)
        };
        &state.1
    }
}

unsafe impl Depend for WriteWith {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Write,
            resource: Resource::Column(self.0.identifier()),
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

    fn declare(&self, state: &Self::State) -> impl Iterator<Item = (u32, Access)> {
        once((state.0, Access::Write))
    }

    unsafe fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        count: u32,
        table: &'a table::Table,
    ) -> Self::Item<'a>
    where
        Self: 'a,
    {
        unsafe {
            table
                .columns()
                .get_unchecked(state.0 as usize)
                .get_in(&mut state.1, count)
        };
        &mut state.1
    }
}
