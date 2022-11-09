use crate::{
    core::{tuples, utility::get_unchecked},
    key::Key,
    table::{Column, Table},
    Database, Datum, Error,
};
use std::{any::TypeId, collections::HashSet, marker::PhantomData, sync::Arc};

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
    columns: &'a [Column],
    row: usize,
}
pub struct ChunkContext<'a> {
    table: &'a Table,
    keys: &'a [Key],
    columns: &'a [Column],
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
        let identifier = TypeId::of::<D>();
        if self.0.contains(&Access::Write(identifier)) {
            Err(Error::ReadWriteConflict)
        } else {
            self.0.insert(Access::Read(identifier));
            Ok(())
        }
    }

    pub fn write<D: Datum>(&mut self) -> Result<(), Error> {
        let identifier = TypeId::of::<D>();
        if self.0.contains(&Access::Read(identifier)) {
            Err(Error::ReadWriteConflict)
        } else if self.0.insert(Access::Write(identifier)) {
            Ok(())
        } else {
            Err(Error::WriteWriteConflict)
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
        Ok(Read(self.0.column::<D>()?, PhantomData))
    }

    pub fn write<D: Datum>(&self) -> Result<Write<D>, Error> {
        Ok(Write(self.0.column::<D>()?, PhantomData))
    }
}

impl<'a> ItemContext<'a> {
    #[inline]
    pub const fn new(table: &'a Table, keys: &'a [Key], columns: &'a [Column]) -> Self {
        Self {
            table,
            keys,
            columns,
            row: 0,
        }
    }

    #[inline]
    pub const fn own(&self) -> Self {
        Self {
            table: self.table,
            keys: self.keys,
            columns: self.columns,
            row: self.row,
        }
    }

    #[inline]
    pub const fn with(&self, row: usize) -> Self {
        debug_assert!(row < self.keys.len());
        Self {
            table: self.table,
            keys: self.keys,
            columns: self.columns,
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
        get_unchecked(self.columns, state.0).get(self.row)
    }

    #[inline]
    pub unsafe fn write<D: Datum>(&self, state: &Write<D>) -> &'a mut D {
        debug_assert!(self.table.has::<D>());
        get_unchecked(self.columns, state.0).get(self.row)
    }
}

impl<'a> ChunkContext<'a> {
    #[inline]
    pub const fn new(table: &'a Table, keys: &'a [Key], columns: &'a [Column]) -> Self {
        Self {
            table,
            keys,
            columns,
        }
    }

    #[inline]
    pub const fn own(&self) -> Self {
        Self {
            table: self.table,
            keys: self.keys,
            columns: self.columns,
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
        get_unchecked(self.columns, state.0).get_all(self.keys.len())
    }

    #[inline]
    pub unsafe fn write<D: Datum>(&self, state: &Write<D>) -> &'a mut [D] {
        get_unchecked(self.columns, state.0).get_all(self.keys.len())
    }
}

impl<R: Row> ShareAccess<R> {
    pub fn from(database: &Database) -> Result<Arc<HashSet<Access>>, Error> {
        let share = database.resources().try_global(|| {
            let mut accesses = DeclareContext::accesses::<R>()?;
            accesses.shrink_to_fit();
            Ok(Self(Arc::new(accesses), PhantomData))
        })?;
        let share = share.read();
        Ok(share.0.clone())
    }
}

unsafe impl Row for Key {
    type State = ();
    type Read = Self;
    type Item<'a> = Key;
    type Chunk<'a> = &'a [Key];

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
    type State = ();
    type Read = ();
    type Item<'a> = &'a Self;
    type Chunk<'a> = &'a Self;

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
    type State = Read<D>;
    type Read = Self;
    type Item<'a> = &'a D;
    type Chunk<'a> = &'a [D];

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
    type State = Write<D>;
    type Read = &'static D;
    type Item<'a> = &'a mut D;
    type Chunk<'a> = &'a mut [D];

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

unsafe impl<R: Row> Row for Option<R> {
    type State = Option<R::State>;
    type Read = Option<R::Read>;
    type Item<'a> = Option<R::Item<'a>>;
    type Chunk<'a> = Option<R::Chunk<'a>>;

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
