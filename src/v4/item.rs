use crate::v4::{
    Meta,
    depend::{Access, Depend, Dependency, Resource},
    slice::Slice,
    table::{self, Lock},
};
use core::{
    any::TypeId,
    cell::RefCell,
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
    fn declare(&self, state: &Self::State) -> impl Iterator<Item = Lock>;
    unsafe fn get<'a>(&'a self, state: &'a mut Self::State, context: Context<'a>) -> Self::Item<'a>
    where
        Self: 'a;
}

#[derive(Debug, Clone, Copy)]
pub struct Context<'a> {
    table: &'a table::Table,
    remove: &'a RefCell<Vec<u32>>,
    count: u32,
    #[cfg(debug_assertions)]
    locks: &'a [Lock],
}

#[derive(Debug, Clone, Copy)]
pub struct Key(pub(crate) ());
#[derive(Debug, Clone, Copy)]
pub struct Rows(pub(crate) ());
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

impl<'a> Context<'a> {
    #[cfg(not(debug_assertions))]
    pub const fn new(count: u32, table: &'a table::Table, remove: &'a RefCell<Vec<u32>>) -> Self {
        Self {
            table,
            count,
            remove,
        }
    }

    #[cfg(debug_assertions)]
    pub const fn new(
        count: u32,
        table: &'a table::Table,
        remove: &'a RefCell<Vec<u32>>,
        locks: &'a [Lock],
    ) -> Self {
        Self {
            table,
            count,
            remove,
            locks,
        }
    }

    pub const fn table(self) -> &'a table::Table {
        self.table
    }

    pub const fn count(self) -> u32 {
        self.count
    }

    pub unsafe fn rows(self) -> table::Rows<'a> {
        #[cfg(debug_assertions)]
        debug_assert!(self.locks.contains(&Lock::Rows));
        unsafe { self.table.rows(self.count, self.remove) }
    }

    pub unsafe fn column<T: 'static>(self, index: u32) -> &'a [T] {
        #[cfg(debug_assertions)]
        debug_assert!(self.locks.contains(&Lock::Column(index, Access::Read)));
        unsafe {
            self.table
                .columns()
                .get_unchecked(index as usize)
                .get(self.count)
        }
    }

    pub unsafe fn column_in(self, index: u32, slice: &mut Slice) -> &Slice {
        #[cfg(debug_assertions)]
        debug_assert!(self.locks.contains(&Lock::Column(index, Access::Read)));
        unsafe {
            self.table
                .columns()
                .get_unchecked(index as usize)
                .get_in(slice, self.count)
        };
        slice
    }

    pub unsafe fn column_mut<T: 'static>(self, index: u32) -> &'a mut [T] {
        #[cfg(debug_assertions)]
        debug_assert!(self.locks.contains(&Lock::Column(index, Access::Write)));
        unsafe {
            self.table
                .columns()
                .get_unchecked(index as usize)
                .get_mut(self.count)
        }
    }

    pub unsafe fn column_mut_in(self, index: u32, slice: &mut Slice) -> &mut Slice {
        #[cfg(debug_assertions)]
        debug_assert!(self.locks.contains(&Lock::Column(index, Access::Write)));
        unsafe {
            self.table
                .columns()
                .get_unchecked(index as usize)
                .get_in(slice, self.count)
        };
        slice
    }
}

impl<I: Item + ?Sized> Item for &I {
    type Item<'a>
        = I::Item<'a>
    where
        Self: 'a;
    type State = I::State;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        I::initialize(self, table)
    }

    fn declare(&self, state: &Self::State) -> impl Iterator<Item = Lock> {
        I::declare(self, state)
    }

    unsafe fn get<'a>(&'a self, state: &'a mut Self::State, context: Context<'a>) -> Self::Item<'a>
    where
        Self: 'a,
    {
        unsafe { I::get(self, state, context) }
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

    fn declare(&self, state: &Self::State) -> impl Iterator<Item = Lock> {
        I::declare(self, state)
    }

    unsafe fn get<'a>(&'a self, state: &'a mut Self::State, context: Context<'a>) -> Self::Item<'a>
    where
        Self: 'a,
    {
        unsafe { I::get(self, state, context) }
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

    fn declare(&self, _: &Self::State) -> impl Iterator<Item = Lock> {
        empty()
    }

    unsafe fn get<'a>(&self, _: &'a mut Self::State, _: Context<'a>) -> Self::Item<'a>
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

    fn declare(&self, state: &Self::State) -> impl Iterator<Item = Lock> {
        self.0.declare(&state.0).merge(self.1.declare(&state.1))
    }

    unsafe fn get<'a>(&'a self, state: &'a mut Self::State, context: Context<'a>) -> Self::Item<'a>
    where
        Self: 'a,
    {
        unsafe {
            (
                self.0.get(&mut state.0, context),
                self.1.get(&mut state.1, context),
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

    fn declare(&self, state: &Self::State) -> impl Iterator<Item = Lock> {
        state
            .as_ref()
            .into_iter()
            .flat_map(|state| self.0.declare(state))
    }

    unsafe fn get<'a>(&'a self, state: &'a mut Self::State, context: Context<'a>) -> Self::Item<'a>
    where
        Self: 'a,
    {
        Some(unsafe { self.0.get(state.as_mut()?, context) })
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

    fn declare(&self, _: &Self::State) -> impl Iterator<Item = Lock> {
        empty()
    }

    unsafe fn get<'a>(&'a self, _: &'a mut Self::State, _: Context<'a>) -> Self::Item<'a>
    where
        Self: 'a,
    {
    }
}

unsafe impl Depend for Rows {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(Dependency {
            access: Access::Read,
            resource: Resource::Table,
        })
    }
}

impl Item for Rows {
    type Item<'a>
        = table::Rows<'a>
    where
        Self: 'a;
    type State = ();

    fn initialize(&self, _: &table::Table) -> Option<Self::State> {
        Some(())
    }

    fn declare(&self, _: &Self::State) -> impl Iterator<Item = Lock> {
        once(Lock::Rows)
    }

    unsafe fn get<'a>(&'a self, _: &'a mut Self::State, context: Context<'a>) -> Self::Item<'a>
    where
        Self: 'a,
    {
        unsafe { context.rows() }
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

    fn declare(&self, _: &Self::State) -> impl Iterator<Item = Lock> {
        empty()
    }

    unsafe fn get<'a>(&'a self, _: &'a mut Self::State, context: Context<'a>) -> Self::Item<'a>
    where
        Self: 'a,
    {
        context.table()
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

    fn declare(&self, state: &Self::State) -> impl Iterator<Item = Lock> {
        once(Lock::Column(*state, Access::Read))
    }

    unsafe fn get<'a>(&'a self, state: &'a mut Self::State, context: Context<'a>) -> Self::Item<'a>
    where
        Self: 'a,
    {
        unsafe { context.column(*state) }
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

    fn declare(&self, state: &Self::State) -> impl Iterator<Item = Lock> {
        once(Lock::Column(*state, Access::Write))
    }

    unsafe fn get<'a>(&'a self, state: &'a mut Self::State, context: Context<'a>) -> Self::Item<'a>
    where
        Self: 'a,
    {
        unsafe { context.column_mut(*state) }
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

    fn declare(&self, state: &Self::State) -> impl Iterator<Item = Lock> {
        once(Lock::Column(state.0, Access::Read))
    }

    unsafe fn get<'a>(&'a self, state: &'a mut Self::State, context: Context<'a>) -> Self::Item<'a>
    where
        Self: 'a,
    {
        unsafe { context.column_in(state.0, &mut state.1) }
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

    fn declare(&self, state: &Self::State) -> impl Iterator<Item = Lock> {
        once(Lock::Column(state.0, Access::Write))
    }

    unsafe fn get<'a>(&'a self, state: &'a mut Self::State, context: Context<'a>) -> Self::Item<'a>
    where
        Self: 'a,
    {
        unsafe { context.column_mut_in(state.0, &mut state.1) }
    }
}
