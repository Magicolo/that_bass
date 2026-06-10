use crate::v4::{
    Meta,
    depend::{Access, Depend, Dependency},
    key, slice,
    table::{self, Lock},
};
use core::{
    any::{Any, TypeId},
    cell::RefCell,
    iter::{empty, once},
    marker::PhantomData,
};
use itertools::Itertools;

pub trait Item: Depend {
    type State;
    type All<'a>;
    type One<'a>;

    fn initialize(&self, table: &table::Table) -> Option<Self::State>;
    fn declare(&self, state: &Self::State) -> impl Iterator<Item = Lock>;
    unsafe fn all<'a>(&self, state: &Self::State, context: Context<'a>) -> Self::All<'a>;
    unsafe fn one<'a>(&self, state: &Self::State, context: Context<'a>) -> Self::One<'a>;
}

#[derive(Debug, Clone, Copy)]
pub struct Context<'a> {
    table: &'a table::Table,
    remove: &'a RefCell<Vec<u32>>,
    row_or_count: u32,
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
    pub(crate) const fn new(
        row_or_count: u32,
        table: &'a table::Table,
        remove: &'a RefCell<Vec<u32>>,
        #[cfg(debug_assertions)] locks: &'a [Lock],
    ) -> Self {
        Context {
            table,
            row_or_count,
            remove,
            #[cfg(debug_assertions)]
            locks,
        }
    }

    #[inline]
    pub const fn table(self) -> &'a table::Table {
        self.table
    }

    #[inline]
    pub unsafe fn rows(self) -> table::Rows<'a> {
        self.assert_rows();
        unsafe { self.table.rows(self.row_or_count, self.remove) }
    }

    #[inline]
    pub unsafe fn row(self) -> table::Row<'a> {
        self.assert_rows();
        unsafe { self.table.row(self.row_or_count, self.remove) }
    }

    #[inline]
    pub unsafe fn column<T: 'static>(self, index: u32) -> &'a [T] {
        self.assert_read(index);
        unsafe {
            self.table
                .columns()
                .get_unchecked(index as usize)
                .get_all(self.row_or_count)
        }
    }

    #[inline]
    pub unsafe fn column_at<T: 'static>(self, index: u32) -> &'a T {
        self.assert_read(index);
        unsafe {
            self.table
                .columns()
                .get_unchecked(index as usize)
                .get_one(self.row_or_count)
        }
    }

    #[inline]
    pub unsafe fn column_mut<T: 'static>(self, index: u32) -> &'a mut [T] {
        self.assert_write(index);
        unsafe {
            self.table
                .columns()
                .get_unchecked(index as usize)
                .get_all_mut(self.row_or_count)
        }
    }

    #[inline]
    pub unsafe fn column_at_mut<T: 'static>(self, index: u32) -> &'a mut T {
        self.assert_write(index);
        unsafe {
            self.table
                .columns()
                .get_unchecked(index as usize)
                .get_one_mut(self.row_or_count)
        }
    }

    #[inline]
    pub unsafe fn slice_at(self, index: u32) -> &'a dyn Any {
        self.assert_read(index);
        unsafe {
            self.table
                .columns()
                .get_unchecked(index as usize)
                .get_any(self.row_or_count)
        }
    }

    #[inline]
    pub unsafe fn slice(self, index: u32) -> slice::Read<'a> {
        self.assert_read(index);
        unsafe {
            self.table
                .columns()
                .get_unchecked(index as usize)
                .get_slice(self.row_or_count)
        }
    }

    #[inline]
    pub unsafe fn slice_at_mut(self, index: u32) -> &'a dyn Any {
        self.assert_write(index);
        unsafe {
            self.table
                .columns()
                .get_unchecked(index as usize)
                .get_any_mut(self.row_or_count)
        }
    }

    #[inline]
    pub unsafe fn slice_mut(self, index: u32) -> slice::Write<'a> {
        self.assert_write(index);
        unsafe {
            self.table
                .columns()
                .get_unchecked(index as usize)
                .get_slice_mut(self.row_or_count)
        }
    }

    #[inline]
    fn assert_rows(self) {
        #[cfg(debug_assertions)]
        debug_assert!(self.locks.contains(&Lock::Rows));
    }

    #[inline]
    fn assert_read(self, index: u32) {
        #[cfg(debug_assertions)]
        debug_assert!(self.locks.contains(&Lock::Column(index, Access::Read)));
    }

    #[inline]
    fn assert_write(self, index: u32) {
        #[cfg(debug_assertions)]
        debug_assert!(self.locks.contains(&Lock::Column(index, Access::Write)));
    }
}

impl<I: Item + ?Sized> Item for &I {
    type All<'a> = I::All<'a>;
    type One<'a> = I::One<'a>;
    type State = I::State;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        I::initialize(self, table)
    }

    #[inline]
    fn declare(&self, state: &Self::State) -> impl Iterator<Item = Lock> {
        I::declare(self, state)
    }

    #[inline]
    unsafe fn all<'a>(&self, state: &Self::State, context: Context<'a>) -> Self::All<'a> {
        unsafe { I::all(self, state, context) }
    }

    #[inline]
    unsafe fn one<'a>(&self, state: &Self::State, context: Context<'a>) -> Self::One<'a> {
        unsafe { I::one(self, state, context) }
    }
}

impl<I: Item + ?Sized> Item for &mut I {
    type All<'a> = I::All<'a>;
    type One<'a> = I::One<'a>;
    type State = I::State;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        I::initialize(self, table)
    }

    #[inline]
    fn declare(&self, state: &Self::State) -> impl Iterator<Item = Lock> {
        I::declare(self, state)
    }

    #[inline]
    unsafe fn all<'a>(&self, state: &Self::State, context: Context<'a>) -> Self::All<'a> {
        unsafe { I::all(self, state, context) }
    }

    #[inline]
    unsafe fn one<'a>(&self, state: &Self::State, context: Context<'a>) -> Self::One<'a> {
        unsafe { I::one(self, state, context) }
    }
}

impl Item for () {
    type All<'a> = ();
    type One<'a> = ();
    type State = ();

    fn initialize(&self, _: &table::Table) -> Option<Self::State> {
        Some(())
    }

    #[inline]
    fn declare(&self, _: &Self::State) -> impl Iterator<Item = Lock> {
        empty()
    }

    #[inline]
    unsafe fn all<'a>(&self, _: &Self::State, _: Context<'a>) -> Self::All<'a> {}

    #[inline]
    unsafe fn one<'a>(&self, _: &Self::State, _: Context<'a>) -> Self::One<'a> {}
}

impl<I0: Item, I1: Item> Item for (I0, I1) {
    type All<'a> = (I0::All<'a>, I1::All<'a>);
    type One<'a> = (I0::One<'a>, I1::One<'a>);
    type State = (I0::State, I1::State);

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Some((self.0.initialize(table)?, self.1.initialize(table)?))
    }

    #[inline]
    fn declare(&self, state: &Self::State) -> impl Iterator<Item = Lock> {
        self.0.declare(&state.0).merge(self.1.declare(&state.1))
    }

    #[inline]
    unsafe fn all<'a>(&self, state: &Self::State, context: Context<'a>) -> Self::All<'a> {
        unsafe { (self.0.all(&state.0, context), self.1.all(&state.1, context)) }
    }

    #[inline]
    unsafe fn one<'a>(&self, state: &Self::State, context: Context<'a>) -> Self::One<'a> {
        unsafe { (self.0.one(&state.0, context), self.1.one(&state.1, context)) }
    }
}

unsafe impl<I: Item + ?Sized> Depend for Try<I> {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        self.0.depend()
    }
}

impl<I: Item + ?Sized> Item for Try<I> {
    type All<'a> = Option<I::All<'a>>;
    type One<'a> = Option<I::One<'a>>;
    type State = Option<I::State>;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Some(self.0.initialize(table))
    }

    #[inline]
    fn declare(&self, state: &Self::State) -> impl Iterator<Item = Lock> {
        state
            .as_ref()
            .into_iter()
            .flat_map(|state| self.0.declare(state))
    }

    #[inline]
    unsafe fn all<'a>(&self, state: &Self::State, context: Context<'a>) -> Self::All<'a> {
        Some(unsafe { self.0.all(state.as_ref()?, context) })
    }

    #[inline]
    unsafe fn one<'a>(&self, state: &Self::State, context: Context<'a>) -> Self::One<'a> {
        Some(unsafe { self.0.one(state.as_ref()?, context) })
    }
}

unsafe impl Depend for Key {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(Dependency::read(Meta::key()))
    }
}

impl Item for Key {
    type All<'a> = &'a [key::Key];
    type One<'a> = &'a key::Key;
    type State = u32;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        table.column(TypeId::of::<key::Key>())
    }

    #[inline]
    fn declare(&self, state: &Self::State) -> impl Iterator<Item = Lock> {
        once(Lock::Column(*state, Access::Read))
    }

    #[inline]
    unsafe fn all<'a>(&self, state: &Self::State, context: Context<'a>) -> Self::All<'a> {
        unsafe { context.column(*state) }
    }

    #[inline]
    unsafe fn one<'a>(&self, state: &Self::State, context: Context<'a>) -> Self::One<'a> {
        unsafe { context.column_at(*state) }
    }
}

unsafe impl Depend for Rows {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(Dependency::read(Meta::table()))
    }
}

impl Item for Rows {
    type All<'a> = table::Rows<'a>;
    type One<'a> = table::Row<'a>;
    type State = ();

    fn initialize(&self, _: &table::Table) -> Option<Self::State> {
        Some(())
    }

    #[inline]
    fn declare(&self, _: &Self::State) -> impl Iterator<Item = Lock> {
        once(Lock::Rows)
    }

    #[inline]
    unsafe fn all<'a>(&self, _: &Self::State, context: Context<'a>) -> Self::All<'a> {
        unsafe { context.rows() }
    }

    #[inline]
    unsafe fn one<'a>(&self, _: &Self::State, context: Context<'a>) -> Self::One<'a> {
        unsafe { context.row() }
    }
}

unsafe impl Depend for Table {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(Dependency::read(Meta::table()))
    }
}

impl Item for Table {
    type All<'a> = &'a table::Table;
    type One<'a> = &'a table::Table;
    type State = ();

    fn initialize(&self, _: &table::Table) -> Option<Self::State> {
        Some(())
    }

    #[inline]
    fn declare(&self, _: &Self::State) -> impl Iterator<Item = Lock> {
        empty()
    }

    #[inline]
    unsafe fn all<'a>(&self, _: &Self::State, context: Context<'a>) -> Self::All<'a> {
        context.table()
    }

    #[inline]
    unsafe fn one<'a>(&self, _: &Self::State, context: Context<'a>) -> Self::One<'a> {
        context.table()
    }
}

unsafe impl<T: 'static> Depend for Read<T> {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(Dependency::read(Meta::of::<T>()))
    }
}

impl<T: ?Sized> Clone for Read<T> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<T: ?Sized> Copy for Read<T> {}

impl<T: 'static> Item for Read<T> {
    type All<'a> = &'a [T];
    type One<'a> = &'a T;
    type State = u32;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        table.column(TypeId::of::<T>())
    }

    #[inline]
    fn declare(&self, state: &Self::State) -> impl Iterator<Item = Lock> {
        once(Lock::Column(*state, Access::Read))
    }

    #[inline]
    unsafe fn all<'a>(&self, state: &Self::State, context: Context<'a>) -> Self::All<'a> {
        unsafe { context.column(*state) }
    }

    #[inline]
    unsafe fn one<'a>(&self, state: &Self::State, context: Context<'a>) -> Self::One<'a> {
        unsafe { context.column_at(*state) }
    }
}

unsafe impl<T: 'static> Depend for Write<T> {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(Dependency::write(Meta::of::<T>()))
    }
}

impl<T: ?Sized> Clone for Write<T> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<T: ?Sized> Copy for Write<T> {}

impl<T: 'static> Item for Write<T> {
    type All<'a> = &'a mut [T];
    type One<'a> = &'a mut T;
    type State = u32;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        table.column(TypeId::of::<T>())
    }

    #[inline]
    fn declare(&self, state: &Self::State) -> impl Iterator<Item = Lock> {
        once(Lock::Column(*state, Access::Write))
    }

    #[inline]
    unsafe fn all<'a>(&self, state: &Self::State, context: Context<'a>) -> Self::All<'a> {
        unsafe { context.column_mut(*state) }
    }

    #[inline]
    unsafe fn one<'a>(&self, state: &Self::State, context: Context<'a>) -> Self::One<'a> {
        unsafe { context.column_at_mut(*state) }
    }
}

unsafe impl Depend for ReadWith {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(Dependency::read(self.0))
    }
}

impl Item for ReadWith {
    type All<'a> = slice::Read<'a>;
    type One<'a> = &'a dyn Any;
    type State = u32;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Some(table.column(self.0.identifier())?)
    }

    #[inline]
    fn declare(&self, state: &Self::State) -> impl Iterator<Item = Lock> {
        once(Lock::Column(*state, Access::Read))
    }

    #[inline]
    unsafe fn all<'a>(&self, state: &Self::State, context: Context<'a>) -> Self::All<'a> {
        unsafe { context.slice(*state) }
    }

    #[inline]
    unsafe fn one<'a>(&self, state: &Self::State, context: Context<'a>) -> Self::One<'a> {
        unsafe { context.slice_at(*state) }
    }
}

unsafe impl Depend for WriteWith {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        once(Dependency::write(self.0))
    }
}

impl Item for WriteWith {
    type All<'a> = slice::Write<'a>;
    type One<'a> = &'a dyn Any;
    type State = u32;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Some(table.column(self.0.identifier())?)
    }

    #[inline]
    fn declare(&self, state: &Self::State) -> impl Iterator<Item = Lock> {
        once(Lock::Column(*state, Access::Write))
    }

    #[inline]
    unsafe fn all<'a>(&self, state: &Self::State, context: Context<'a>) -> Self::All<'a> {
        unsafe { context.slice_mut(*state) }
    }

    #[inline]
    unsafe fn one<'a>(&self, state: &Self::State, context: Context<'a>) -> Self::One<'a> {
        unsafe { context.slice_at_mut(*state) }
    }
}
