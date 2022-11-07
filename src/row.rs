use crate::{
    core::utility::get_unchecked,
    key::Key,
    table::{Column, Table},
    Datum, Error,
};
use std::{any::TypeId, collections::HashSet, marker::PhantomData};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Access {
    Read(TypeId),
    Write(TypeId),
}

pub struct Read<D>(usize, PhantomData<fn(D)>);
pub struct Write<D>(usize, PhantomData<fn(D)>);

pub struct DeclareContext<'a>(&'a mut HashSet<Access>);
pub struct InitializeContext<'a>(&'a Table);
pub struct ItemContext<'a>(&'a [Key], &'a [Column], usize);
pub struct ChunkContext<'a>(&'a [Key], &'a [Column]);

pub unsafe trait Row {
    type State;
    type Read: Row;
    type Item<'a>;
    type Chunk<'a>;

    fn declare(context: DeclareContext) -> Result<(), Error>;
    fn initialize(context: InitializeContext) -> Result<Self::State, Error>;
    fn read(state: Self::State) -> <Self::Read as Row>::State;
    unsafe fn item<'a>(state: &'a mut Self::State, context: ItemContext<'a>) -> Self::Item<'a>;
    unsafe fn chunk<'a>(state: &'a mut Self::State, context: ChunkContext<'a>) -> Self::Chunk<'a>;
}

impl<D> Write<D> {
    pub fn read(self) -> Read<D> {
        Read(self.0, PhantomData)
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
    pub const fn new(keys: &'a [Key], columns: &'a [Column]) -> Self {
        Self(keys, columns, 0)
    }

    #[inline]
    pub const fn own(&self) -> Self {
        Self(self.0, self.1, self.2)
    }

    #[inline]
    pub const fn with(&self, index: usize) -> Self {
        debug_assert!(index < self.0.len());
        Self(self.0, self.1, index)
    }

    #[inline]
    pub fn key(&self) -> Key {
        unsafe { *get_unchecked(self.0, self.row()) }
    }

    #[inline]
    pub const fn row(&self) -> usize {
        self.2
    }

    #[inline]
    pub unsafe fn read<D: Datum>(&self, state: &Read<D>) -> &'a D {
        get_unchecked(self.1, state.0).get(self.2)
    }

    #[inline]
    pub unsafe fn write<D: Datum>(&self, state: &Write<D>) -> &'a mut D {
        get_unchecked(self.1, state.0).get(self.2)
    }
}

impl<'a> ChunkContext<'a> {
    #[inline]
    pub const fn new(keys: &'a [Key], columns: &'a [Column]) -> Self {
        Self(keys, columns)
    }

    #[inline]
    pub const fn own(&self) -> Self {
        Self(self.0, self.1)
    }

    #[inline]
    pub const fn key(&self) -> &'a [Key] {
        self.0
    }

    #[inline]
    pub unsafe fn read<D: Datum>(&self, state: &Read<D>) -> &'a [D] {
        get_unchecked(self.1, state.0).get_all(self.0.len())
    }

    #[inline]
    pub unsafe fn write<D: Datum>(&self, state: &Write<D>) -> &'a mut [D] {
        get_unchecked(self.1, state.0).get_all(self.0.len())
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
    fn read(state: Self::State) -> <Self::Read as Row>::State {
        state
    }
    #[inline]
    unsafe fn item<'a>(_: &'a mut Self::State, context: ItemContext<'a>) -> Self::Item<'a> {
        context.key()
    }
    #[inline]
    unsafe fn chunk<'a>(_: &'a mut Self::State, context: ChunkContext<'a>) -> Self::Chunk<'a> {
        context.key()
    }
}

unsafe impl<D: Datum> Row for &D {
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
    fn read(state: Self::State) -> <Self::Read as Row>::State {
        state
    }
    #[inline]
    unsafe fn item<'a>(state: &'a mut Self::State, context: ItemContext<'a>) -> Self::Item<'a> {
        context.read(state)
    }
    #[inline]
    unsafe fn chunk<'a>(state: &'a mut Self::State, context: ChunkContext<'a>) -> Self::Chunk<'a> {
        context.read(state)
    }
}

unsafe impl<'b, D: Datum> Row for &'b mut D {
    type State = Write<D>;
    type Read = &'b D;
    type Item<'a> = &'a mut D;
    type Chunk<'a> = &'a mut [D];

    fn declare(mut context: DeclareContext) -> Result<(), Error> {
        context.write::<D>()
    }
    fn initialize(context: InitializeContext) -> Result<Self::State, Error> {
        context.write::<D>()
    }
    fn read(state: Self::State) -> <Self::Read as Row>::State {
        state.read()
    }
    #[inline]
    unsafe fn item<'a>(state: &'a mut Self::State, context: ItemContext<'a>) -> Self::Item<'a> {
        context.write(state)
    }
    #[inline]
    unsafe fn chunk<'a>(state: &'a mut Self::State, context: ChunkContext<'a>) -> Self::Chunk<'a> {
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
    fn read(state: Self::State) -> <Self::Read as Row>::State {
        Some(R::read(state?))
    }
    #[inline]
    unsafe fn item<'a>(state: &'a mut Self::State, context: ItemContext<'a>) -> Self::Item<'a> {
        Some(R::item(state.as_mut()?, context))
    }
    #[inline]
    unsafe fn chunk<'a>(state: &'a mut Self::State, context: ChunkContext<'a>) -> Self::Chunk<'a> {
        Some(R::chunk(state.as_mut()?, context))
    }
}

unsafe impl Row for () {
    type State = ();
    type Read = ();
    type Item<'a> = ();
    type Chunk<'a> = ();

    fn declare(_: DeclareContext) -> Result<(), Error> {
        Ok(())
    }
    fn initialize(_: InitializeContext) -> Result<Self::State, Error> {
        Ok(())
    }
    fn read(_: Self::State) -> <Self::Read as Row>::State {
        ()
    }
    #[inline]
    unsafe fn item<'a>(_: &'a mut Self::State, _: ItemContext<'a>) -> Self::Item<'a> {
        ()
    }
    #[inline]
    unsafe fn chunk<'a>(_: &'a mut Self::State, _: ChunkContext<'a>) -> Self::Chunk<'a> {
        ()
    }
}

unsafe impl<R1: Row> Row for (R1,) {
    type State = (R1::State,);
    type Read = (R1::Read,);
    type Item<'a> = (R1::Item<'a>,);
    type Chunk<'a> = (R1::Chunk<'a>,);

    fn declare(mut context: DeclareContext) -> Result<(), Error> {
        R1::declare(context.own())?;
        Ok(())
    }
    fn initialize(context: InitializeContext) -> Result<Self::State, Error> {
        Ok((R1::initialize(context.own())?,))
    }
    fn read(state: Self::State) -> <Self::Read as Row>::State {
        (R1::read(state.0),)
    }
    #[inline]
    unsafe fn item<'a>(state: &'a mut Self::State, context: ItemContext<'a>) -> Self::Item<'a> {
        (R1::item(&mut state.0, context.own()),)
    }
    #[inline]
    unsafe fn chunk<'a>(state: &'a mut Self::State, context: ChunkContext<'a>) -> Self::Chunk<'a> {
        (R1::chunk(&mut state.0, context.own()),)
    }
}

unsafe impl<R1: Row, R2: Row> Row for (R1, R2) {
    type State = (R1::State, R2::State);
    type Read = (R1::Read, R2::Read);
    type Item<'a> = (R1::Item<'a>, R2::Item<'a>);
    type Chunk<'a> = (R1::Chunk<'a>, R2::Chunk<'a>);

    fn declare(mut context: DeclareContext) -> Result<(), Error> {
        R1::declare(context.own())?;
        R2::declare(context.own())?;
        Ok(())
    }
    fn initialize(context: InitializeContext) -> Result<Self::State, Error> {
        Ok((
            R1::initialize(context.own())?,
            R2::initialize(context.own())?,
        ))
    }
    fn read(state: Self::State) -> <Self::Read as Row>::State {
        (R1::read(state.0), R2::read(state.1))
    }
    #[inline]
    unsafe fn item<'a>(state: &'a mut Self::State, context: ItemContext<'a>) -> Self::Item<'a> {
        (
            R1::item(&mut state.0, context.own()),
            R2::item(&mut state.1, context.own()),
        )
    }
    #[inline]
    unsafe fn chunk<'a>(state: &'a mut Self::State, context: ChunkContext<'a>) -> Self::Chunk<'a> {
        (
            R1::chunk(&mut state.0, context.own()),
            R2::chunk(&mut state.1, context.own()),
        )
    }
}

unsafe impl<R1: Row, R2: Row, R3: Row> Row for (R1, R2, R3) {
    type State = (R1::State, R2::State, R3::State);
    type Read = (R1::Read, R2::Read, R3::Read);
    type Item<'a> = (R1::Item<'a>, R2::Item<'a>, R3::Item<'a>);
    type Chunk<'a> = (R1::Chunk<'a>, R2::Chunk<'a>, R3::Chunk<'a>);

    fn declare(mut context: DeclareContext) -> Result<(), Error> {
        R1::declare(context.own())?;
        R2::declare(context.own())?;
        R3::declare(context.own())?;
        Ok(())
    }
    fn initialize(context: InitializeContext) -> Result<Self::State, Error> {
        Ok((
            R1::initialize(context.own())?,
            R2::initialize(context.own())?,
            R3::initialize(context.own())?,
        ))
    }
    fn read(state: Self::State) -> <Self::Read as Row>::State {
        (R1::read(state.0), R2::read(state.1), R3::read(state.2))
    }
    #[inline]
    unsafe fn item<'a>(state: &'a mut Self::State, context: ItemContext<'a>) -> Self::Item<'a> {
        (
            R1::item(&mut state.0, context.own()),
            R2::item(&mut state.1, context.own()),
            R3::item(&mut state.2, context.own()),
        )
    }
    #[inline]
    unsafe fn chunk<'a>(state: &'a mut Self::State, context: ChunkContext<'a>) -> Self::Chunk<'a> {
        (
            R1::chunk(&mut state.0, context.own()),
            R2::chunk(&mut state.1, context.own()),
            R3::chunk(&mut state.2, context.own()),
        )
    }
}
