use crate::v4::{
    Meta, Rows,
    depend::{Access, Depend, Dependency, Resource},
    guard,
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
    type Guard<'a>
    where
        Self: 'a;

    fn initialize(&self, table: &table::Table) -> Option<Self::State>;
    fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        count: u32,
        table: &'a table::Table,
    ) -> Self::Guard<'a>
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

impl<I: Item> Item for &I {
    type Guard<'a>
        = I::Guard<'a>
    where
        Self: 'a;
    type State = I::State;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        I::initialize(self, table)
    }

    fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        count: u32,
        table: &'a table::Table,
    ) -> Self::Guard<'a>
    where
        Self: 'a,
    {
        I::get(self, state, count, table)
    }
}

impl<I: Item> Item for &mut I {
    type Guard<'a>
        = I::Guard<'a>
    where
        Self: 'a;
    type State = I::State;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        I::initialize(self, table)
    }

    fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        count: u32,
        table: &'a table::Table,
    ) -> Self::Guard<'a>
    where
        Self: 'a,
    {
        I::get(self, state, count, table)
    }
}

impl Item for () {
    type Guard<'a>
        = ()
    where
        Self: 'a;
    type State = ();

    fn initialize(&self, _: &table::Table) -> Option<Self::State> {
        Some(())
    }

    fn get<'a>(&self, _: &'a mut Self::State, _: u32, _: &'a table::Table) -> Self::Guard<'a>
    where
        Self: 'a,
    {
    }
}

impl<A0: Item, A1: Item> Item for (A0, A1) {
    type Guard<'a>
        = (A0::Guard<'a>, A1::Guard<'a>)
    where
        Self: 'a;
    type State = (A0::State, A1::State);

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Some((self.0.initialize(table)?, self.1.initialize(table)?))
    }

    fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        count: u32,
        table: &'a table::Table,
    ) -> Self::Guard<'a>
    where
        Self: 'a,
    {
        (
            self.0.get(&mut state.0, count, table),
            self.1.get(&mut state.1, count, table),
        )
    }
}

unsafe impl<I: Item> Depend for Try<I> {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        self.0.depend()
    }
}

impl<I: Item> Item for Try<I> {
    type Guard<'a>
        = Option<I::Guard<'a>>
    where
        Self: 'a;
    type State = Option<I::State>;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Some(self.0.initialize(table))
    }

    fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        count: u32,
        table: &'a table::Table,
    ) -> Self::Guard<'a>
    where
        Self: 'a,
    {
        Some(self.0.get(state.as_mut()?, count, table))
    }
}

unsafe impl Depend for Key {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        empty()
    }
}

// TODO: Implement
impl Item for Key {
    type Guard<'a>
        = ()
    where
        Self: 'a;
    type State = ();

    fn initialize(&self, _: &table::Table) -> Option<Self::State> {
        None
    }

    fn get<'a>(&'a self, _: &'a mut Self::State, _: u32, _: &'a table::Table) -> Self::Guard<'a>
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
    type Guard<'a>
        = guard::Read<'a, [T]>
    where
        Self: 'a;
    type State = u32;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        table.column(TypeId::of::<T>())
    }

    fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        count: u32,
        table: &'a table::Table,
    ) -> Self::Guard<'a>
    where
        Self: 'a,
    {
        unsafe { table.columns().get_unchecked(*state as usize).read(count) }
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
    type Guard<'a>
        = guard::Write<'a, [T]>
    where
        Self: 'a;
    type State = u32;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        table.column(TypeId::of::<T>())
    }

    fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        count: u32,
        table: &'a table::Table,
    ) -> Self::Guard<'a>
    where
        Self: 'a,
    {
        unsafe { table.columns().get_unchecked(*state as usize).write(count) }
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
    type Guard<'a>
        = Rows<'a>
    where
        Self: 'a;
    type State = ();

    fn initialize(&self, _: &table::Table) -> Option<Self::State> {
        Some(())
    }

    fn get<'a>(
        &'a self,
        _: &'a mut Self::State,
        count: u32,
        table: &'a table::Table,
    ) -> Self::Guard<'a>
    where
        Self: 'a,
    {
        Rows::new(0..count, table)
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
    type Guard<'a>
        = &'a table::Table
    where
        Self: 'a;
    type State = ();

    fn initialize(&self, _: &table::Table) -> Option<Self::State> {
        Some(())
    }

    fn get<'a>(&'a self, _: &'a mut Self::State, _: u32, table: &'a table::Table) -> Self::Guard<'a>
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
    type Guard<'a>
        = &'a Slice
    where
        Self: 'a;
    type State = (u32, Slice);

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Some((table.column(self.0.identifier())?, Slice::empty(self.0)))
    }

    fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        count: u32,
        table: &'a table::Table,
    ) -> Self::Guard<'a>
    where
        Self: 'a,
    {
        let column = unsafe { table.columns().get_unchecked(state.0 as usize) };
        unsafe { state.1.set_parts(column.data().cast(), count as _) };
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
    type Guard<'a>
        = &'a mut Slice
    where
        Self: 'a;
    type State = (u32, Slice);

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Some((table.column(self.0.identifier())?, Slice::empty(self.0)))
    }

    fn get<'a>(
        &'a self,
        state: &'a mut Self::State,
        count: u32,
        table: &'a table::Table,
    ) -> Self::Guard<'a>
    where
        Self: 'a,
    {
        let column = unsafe { table.columns().get_unchecked(state.0 as usize) };
        unsafe { state.1.set_parts(column.data().cast(), count as _) };
        &mut state.1
    }
}
