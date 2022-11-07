use crate::{core::utility::get_unchecked, key::Key, Datum, Error};
use std::{
    any::TypeId,
    collections::{HashMap, HashSet},
    marker::PhantomData,
    ptr::NonNull,
    slice::{from_raw_parts, from_raw_parts_mut},
};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Access {
    Read(TypeId),
    Write(TypeId),
}

pub struct Read<D>(usize, PhantomData<fn(D)>);
pub struct Write<D>(usize, PhantomData<fn(D)>);

pub struct DeclareContext<'a>(pub(crate) &'a mut HashSet<Access>);
pub struct InitializeContext<'a>(pub(crate) &'a HashMap<Access, usize>);
pub struct ItemContext<'a, 'b>(pub(crate) &'a [Key], pub(crate) &'b [NonNull<()>], usize);
pub struct ChunkContext<'a, 'b>(pub(crate) &'a [Key], pub(crate) &'b [NonNull<()>]);

pub unsafe trait Row {
    type State;
    type Read: Row;
    type Item<'a>;
    type Chunk<'a>;

    fn declare(context: DeclareContext) -> Result<(), Error>;
    fn initialize(context: InitializeContext) -> Result<Self::State, Error>;
    fn read(state: Self::State) -> <Self::Read as Row>::State;
    fn item<'a>(state: &'a mut Self::State, context: ItemContext<'a, '_>) -> Self::Item<'a>;
    fn chunk<'a>(state: &'a mut Self::State, context: ChunkContext<'a, '_>) -> Self::Chunk<'a>;
}

impl<D> Write<D> {
    pub fn read(self) -> Read<D> {
        Read(self.0, PhantomData)
    }
}

impl DeclareContext<'_> {
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

impl InitializeContext<'_> {
    pub fn own(&self) -> Self {
        Self(self.0)
    }

    pub fn read<D: Datum>(&self) -> Result<Read<D>, Error> {
        match self.0.get(&Access::Read(TypeId::of::<D>())) {
            Some(&index) => Ok(Read(index, PhantomData)),
            None => Err(Error::MissingColumn),
        }
    }

    pub fn write<D: Datum>(&self) -> Result<Write<D>, Error> {
        match self.0.get(&Access::Write(TypeId::of::<D>())) {
            Some(&index) => Ok(Write(index, PhantomData)),
            None => Err(Error::MissingColumn),
        }
    }
}

impl<'a, 'b> ItemContext<'a, 'b> {
    #[inline]
    pub const fn new(keys: &'a [Key], columns: &'b [NonNull<()>]) -> Self {
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
    pub fn read<D: Datum>(&self, state: &Read<D>) -> &'a D {
        let data = unsafe { *get_unchecked(self.1, state.0) };
        unsafe { &*data.as_ptr().cast::<D>().add(self.2) }
    }

    #[inline]
    pub fn write<D: Datum>(&self, state: &Write<D>) -> &'a mut D {
        let data = unsafe { *get_unchecked(self.1, state.0) };
        unsafe { &mut *data.as_ptr().cast::<D>().add(self.2) }
    }
}

impl<'a> ChunkContext<'a, '_> {
    #[inline]
    pub const fn own(&self) -> Self {
        Self(self.0, self.1)
    }

    #[inline]
    pub const fn key(&self) -> &'a [Key] {
        self.0
    }

    #[inline]
    pub fn read<D: Datum>(&self, state: &Read<D>) -> &'a [D] {
        let data = unsafe { *get_unchecked(self.1, state.0) };
        unsafe { from_raw_parts(data.as_ptr().cast::<D>(), self.0.len()) }
    }

    #[inline]
    pub fn write<D: Datum>(&self, state: &Write<D>) -> &'a mut [D] {
        let data = unsafe { *get_unchecked(self.1, state.0) };
        unsafe { from_raw_parts_mut(data.as_ptr().cast::<D>(), self.0.len()) }
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
    fn item<'a>(_: &'a mut Self::State, context: ItemContext<'a, '_>) -> Self::Item<'a> {
        context.key()
    }
    #[inline]
    fn chunk<'a>(_: &'a mut Self::State, context: ChunkContext<'a, '_>) -> Self::Chunk<'a> {
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
    fn item<'a>(state: &'a mut Self::State, context: ItemContext<'a, '_>) -> Self::Item<'a> {
        context.read(state)
    }
    #[inline]
    fn chunk<'a>(state: &'a mut Self::State, context: ChunkContext<'a, '_>) -> Self::Chunk<'a> {
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
    fn item<'a>(state: &'a mut Self::State, context: ItemContext<'a, '_>) -> Self::Item<'a> {
        context.write(state)
    }
    #[inline]
    fn chunk<'a>(state: &'a mut Self::State, context: ChunkContext<'a, '_>) -> Self::Chunk<'a> {
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
    fn item<'a>(state: &'a mut Self::State, context: ItemContext<'a, '_>) -> Self::Item<'a> {
        Some(R::item(state.as_mut()?, context))
    }

    #[inline]
    fn chunk<'a>(state: &'a mut Self::State, context: ChunkContext<'a, '_>) -> Self::Chunk<'a> {
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
    fn item<'a>(_: &'a mut Self::State, _: ItemContext<'a, '_>) -> Self::Item<'a> {
        ()
    }
    #[inline]
    fn chunk<'a>(_: &'a mut Self::State, _: ChunkContext<'a, '_>) -> Self::Chunk<'a> {
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
    fn item<'a>(state: &'a mut Self::State, context: ItemContext<'a, '_>) -> Self::Item<'a> {
        (R1::item(&mut state.0, context.own()),)
    }
    #[inline]
    fn chunk<'a>(state: &'a mut Self::State, context: ChunkContext<'a, '_>) -> Self::Chunk<'a> {
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
    fn item<'a>(state: &'a mut Self::State, context: ItemContext<'a, '_>) -> Self::Item<'a> {
        (
            R1::item(&mut state.0, context.own()),
            R2::item(&mut state.1, context.own()),
        )
    }
    #[inline]
    fn chunk<'a>(state: &'a mut Self::State, context: ChunkContext<'a, '_>) -> Self::Chunk<'a> {
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
    fn item<'a>(state: &'a mut Self::State, context: ItemContext<'a, '_>) -> Self::Item<'a> {
        (
            R1::item(&mut state.0, context.own()),
            R2::item(&mut state.1, context.own()),
            R3::item(&mut state.2, context.own()),
        )
    }
    #[inline]
    fn chunk<'a>(state: &'a mut Self::State, context: ChunkContext<'a, '_>) -> Self::Chunk<'a> {
        (
            R1::chunk(&mut state.0, context.own()),
            R2::chunk(&mut state.1, context.own()),
            R3::chunk(&mut state.2, context.own()),
        )
    }
}
