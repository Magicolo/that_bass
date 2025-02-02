use crate::{
    Datum, Error,
    core::{tuple::tuples, utility::get_unchecked},
    key::Key,
    resources::Resources,
    table::Table,
};
use std::{any::TypeId, collections::HashSet, marker::PhantomData, num::NonZeroUsize, sync::Arc};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Access {
    Read(TypeId),
    Write(TypeId),
}

pub struct Read<D>(usize, PhantomData<fn(D)>);
pub struct Write<D>(usize, PhantomData<fn(D)>);

pub struct DeclareContext<'a>(&'a mut HashSet<Access>);
pub struct InitializeContext<'a>(&'a Table);
pub struct ItemContext<'a> {
    table: &'a Table,
    keys: &'a [Key],
    row: usize,
}
pub struct ChunkContext<'a> {
    table: &'a Table,
    keys: &'a [Key],
    count: NonZeroUsize,
}

pub unsafe trait Row: 'static {
    type State;
    type Read: Row;
    type Item<'a>;
    type Chunk<'a>;

    fn declare(context: DeclareContext) -> Result<(), Error>;
    fn initialize(context: InitializeContext) -> Result<Self::State, Error>;
    fn read(state: &Self::State) -> <Self::Read as Row>::State;
    unsafe fn item<'a>(state: &'a Self::State, context: ItemContext<'a>) -> Self::Item<'a>;
    unsafe fn chunk<'a>(state: &'a Self::State, context: ChunkContext<'a>) -> Self::Chunk<'a>;
}

pub(crate) struct ShareAccess<R>(Arc<HashSet<Access>>, PhantomData<fn(R)>);

impl<D> Write<D> {
    pub fn read(&self) -> Read<D> {
        Read(self.0, PhantomData)
    }
}

impl<D> Clone for Write<D> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

impl<D> Clone for Read<D> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

impl DeclareContext<'_> {
    /// Detects violations of rust's invariants.
    pub fn accesses<R: Row>() -> Result<HashSet<Access>, Error> {
        let mut accesses = HashSet::new();
        R::declare(DeclareContext(&mut accesses))?;
        accesses.shrink_to_fit();
        Ok(accesses)
    }

    pub fn own(&mut self) -> DeclareContext<'_> {
        DeclareContext(self.0)
    }

    pub fn read<D: Datum>(&mut self) -> Result<(), Error> {
        self.read_with(TypeId::of::<D>())
    }

    pub fn read_with(&mut self, identifier: TypeId) -> Result<(), Error> {
        if self.0.contains(&Access::Write(identifier)) {
            Err(Error::ReadWriteConflict(identifier))
        } else {
            self.0.insert(Access::Read(identifier));
            Ok(())
        }
    }

    pub fn write<D: Datum>(&mut self) -> Result<(), Error> {
        self.write_with(TypeId::of::<D>())
    }

    pub fn write_with(&mut self, identifier: TypeId) -> Result<(), Error> {
        if self.0.contains(&Access::Read(identifier)) {
            Err(Error::ReadWriteConflict(identifier))
        } else if self.0.insert(Access::Write(identifier)) {
            Ok(())
        } else {
            Err(Error::WriteWriteConflict(identifier))
        }
    }
}

impl<'a> InitializeContext<'a> {
    pub const fn new(table: &'a Table) -> Self {
        Self(table)
    }

    pub fn own(&self) -> Self {
        Self(self.0)
    }

    pub fn read<D: Datum>(&self) -> Result<Read<D>, Error> {
        Ok(Read(self.0.column::<D>()?.0, PhantomData))
    }

    pub fn write<D: Datum>(&self) -> Result<Write<D>, Error> {
        Ok(Write(self.0.column::<D>()?.0, PhantomData))
    }
}

impl<'a> ItemContext<'a> {
    #[inline]
    pub const fn new(table: &'a Table, keys: &'a [Key]) -> Self {
        Self {
            table,
            keys,
            row: 0,
        }
    }

    #[inline]
    pub const fn own(&self) -> Self {
        Self {
            table: self.table,
            keys: self.keys,
            row: self.row,
        }
    }

    #[inline]
    pub fn with(&self, row: usize) -> Self {
        debug_assert!(row < self.table.count());
        Self {
            table: self.table,
            keys: self.keys,
            row,
        }
    }

    #[inline]
    pub const fn table(&self) -> &'a Table {
        self.table
    }

    #[inline]
    pub fn key(&self) -> Key {
        unsafe { *get_unchecked(self.keys, self.row()) }
    }

    #[inline]
    pub const fn row(&self) -> usize {
        self.row
    }

    #[inline]
    pub unsafe fn read<D: Datum>(&self, state: &Read<D>) -> &'a D {
        debug_assert!(self.table.has::<D>());
        get_unchecked(self.table.columns(), state.0).get(self.row)
    }

    #[inline]
    pub unsafe fn write<D: Datum>(&self, state: &Write<D>) -> &'a mut D {
        debug_assert!(self.table.has::<D>());
        get_unchecked(self.table.columns(), state.0).get(self.row)
    }
}

impl<'a> ChunkContext<'a> {
    #[inline]
    pub const fn new(table: &'a Table, keys: &'a [Key], count: NonZeroUsize) -> Self {
        Self { table, keys, count }
    }

    #[inline]
    pub const fn own(&self) -> Self {
        Self {
            table: self.table,
            keys: self.keys,
            count: self.count,
        }
    }

    #[inline]
    pub const fn table(&self) -> &'a Table {
        self.table
    }

    #[inline]
    pub const fn keys(&self) -> &'a [Key] {
        self.keys
    }

    #[inline]
    pub unsafe fn read<D: Datum>(&self, state: &Read<D>) -> &'a [D] {
        get_unchecked(self.table.columns(), state.0).get_all(self.count.get())
    }

    #[inline]
    pub unsafe fn write<D: Datum>(&self, state: &Write<D>) -> &'a mut [D] {
        get_unchecked(self.table.columns(), state.0).get_all(self.count.get())
    }
}

impl<R: Row> ShareAccess<R> {
    pub fn from(resources: &Resources) -> Result<Arc<HashSet<Access>>, Error> {
        let share = resources.try_global(|| {
            let mut accesses = DeclareContext::accesses::<R>()?;
            accesses.shrink_to_fit();
            Ok(Self(Arc::new(accesses), PhantomData))
        })?;
        Ok(share.0.clone())
    }
}

unsafe impl Row for Key {
    type Chunk<'a> = &'a [Key];
    type Item<'a> = Key;
    type Read = Self;
    type State = ();

    fn declare(_: DeclareContext) -> Result<(), Error> {
        Ok(())
    }

    fn initialize(_: InitializeContext) -> Result<Self::State, Error> {
        Ok(())
    }

    fn read(_: &Self::State) -> <Self::Read as Row>::State {}

    #[inline]
    unsafe fn item<'a>(_: &'a Self::State, context: ItemContext<'a>) -> Self::Item<'a> {
        context.key()
    }

    #[inline]
    unsafe fn chunk<'a>(_: &'a Self::State, context: ChunkContext<'a>) -> Self::Chunk<'a> {
        context.keys()
    }
}

unsafe impl Row for Table {
    type Chunk<'a> = &'a Self;
    type Item<'a> = &'a Self;
    type Read = ();
    type State = ();

    fn declare(_: DeclareContext) -> Result<(), Error> {
        Ok(())
    }

    fn initialize(_: InitializeContext) -> Result<Self::State, Error> {
        Ok(())
    }

    fn read(_: &Self::State) -> <Self::Read as Row>::State {}

    #[inline]
    unsafe fn item<'a>(_: &'a Self::State, context: ItemContext<'a>) -> Self::Item<'a> {
        context.table()
    }

    #[inline]
    unsafe fn chunk<'a>(_: &'a Self::State, context: ChunkContext<'a>) -> Self::Chunk<'a> {
        context.table()
    }
}

unsafe impl<D: Datum> Row for &'static D {
    type Chunk<'a> = &'a [D];
    type Item<'a> = &'a D;
    type Read = Self;
    type State = Read<D>;

    fn declare(mut context: DeclareContext) -> Result<(), Error> {
        context.read::<D>()
    }

    fn initialize(context: InitializeContext) -> Result<Self::State, Error> {
        context.read::<D>()
    }

    fn read(state: &Self::State) -> <Self::Read as Row>::State {
        state.clone()
    }

    #[inline]
    unsafe fn item<'a>(state: &'a Self::State, context: ItemContext<'a>) -> Self::Item<'a> {
        context.read(state)
    }

    #[inline]
    unsafe fn chunk<'a>(state: &'a Self::State, context: ChunkContext<'a>) -> Self::Chunk<'a> {
        context.read(state)
    }
}

unsafe impl<D: Datum> Row for &'static mut D {
    type Chunk<'a> = &'a mut [D];
    type Item<'a> = &'a mut D;
    type Read = &'static D;
    type State = Write<D>;

    fn declare(mut context: DeclareContext) -> Result<(), Error> {
        context.write::<D>()
    }

    fn initialize(context: InitializeContext) -> Result<Self::State, Error> {
        context.write::<D>()
    }

    fn read(state: &Self::State) -> <Self::Read as Row>::State {
        state.read()
    }

    #[inline]
    unsafe fn item<'a>(state: &'a Self::State, context: ItemContext<'a>) -> Self::Item<'a> {
        context.write(state)
    }

    #[inline]
    unsafe fn chunk<'a>(state: &'a Self::State, context: ChunkContext<'a>) -> Self::Chunk<'a> {
        context.write(state)
    }
}

unsafe impl<T: 'static> Row for PhantomData<T> {
    type Chunk<'a> = ();
    type Item<'a> = ();
    type Read = Self;
    type State = ();

    fn declare(_: DeclareContext) -> Result<(), Error> {
        Ok(())
    }

    fn initialize(_: InitializeContext) -> Result<Self::State, Error> {
        Ok(())
    }

    fn read(_: &Self::State) -> <Self::Read as Row>::State {}

    #[inline]
    unsafe fn item<'a>(_: &'a Self::State, _: ItemContext<'a>) -> Self::Item<'a> {}

    #[inline]
    unsafe fn chunk<'a>(_: &'a Self::State, _: ChunkContext<'a>) -> Self::Chunk<'a> {}
}

unsafe impl<R: Row> Row for Option<R> {
    type Chunk<'a> = Option<R::Chunk<'a>>;
    type Item<'a> = Option<R::Item<'a>>;
    type Read = Option<R::Read>;
    type State = Option<R::State>;

    fn declare(context: DeclareContext) -> Result<(), Error> {
        R::declare(context)
    }

    fn initialize(context: InitializeContext) -> Result<Self::State, Error> {
        Ok(R::initialize(context).ok())
    }

    fn read(state: &Self::State) -> <Self::Read as Row>::State {
        Some(R::read(state.as_ref()?))
    }

    #[inline]
    unsafe fn item<'a>(state: &'a Self::State, context: ItemContext<'a>) -> Self::Item<'a> {
        Some(R::item(state.as_ref()?, context))
    }

    #[inline]
    unsafe fn chunk<'a>(state: &'a Self::State, context: ChunkContext<'a>) -> Self::Chunk<'a> {
        Some(R::chunk(state.as_ref()?, context))
    }
}

macro_rules! tuple {
    ($n:ident, $c:expr $(, $p:ident, $t:ident, $i:tt)*) => {
        #[allow(clippy::unused_unit)]
        unsafe impl<$($t: Row,)*> Row for ($($t,)*) {
            type State = ($($t::State,)*);
            type Read = ($($t::Read,)*);
            type Item<'a> = ($($t::Item<'a>,)*);
            type Chunk<'a> = ($($t::Chunk<'a>,)*);

            fn declare(mut _context: DeclareContext) -> Result<(), Error> {
                $($t::declare(_context.own())?;)*
                Ok(())
            }
            fn initialize(_context: InitializeContext) -> Result<Self::State, Error> {
                Ok(($($t::initialize(_context.own())?,)*))
            }
            fn read(_state: &Self::State) -> <Self::Read as Row>::State {
                ($($t::read(&_state.$i),)*)
            }
            #[inline]
            unsafe fn item<'a>(_state: &'a Self::State, _context: ItemContext<'a>) -> Self::Item<'a> {
                ($($t::item(&_state.$i, _context.own()),)*)
            }
            #[inline]
            unsafe fn chunk<'a>(
                _state: &'a Self::State,
                _context: ChunkContext<'a>,
            ) -> Self::Chunk<'a> {
                ($($t::chunk(&_state.$i, _context.own()),)*)
            }
        }
    };
}
tuples!(tuple);
