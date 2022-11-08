use crate::{
    core::utility::get_unchecked,
    table::{Column, Table},
    Datum, Error, Meta,
};
use std::{marker::PhantomData, mem::size_of};

pub struct Apply<D>(usize, PhantomData<fn(D)>);
pub struct DeclareContext<'a>(&'a mut Vec<&'static Meta>);
pub struct InitializeContext<'a>(&'a Table);
pub struct ApplyContext<'a>(&'a [Column], usize);

pub unsafe trait Template: 'static {
    const SIZE: usize;
    type State: Send + Sync;
    fn declare(context: DeclareContext) -> Result<(), Error>;
    fn initialize(context: InitializeContext) -> Result<Self::State, Error>;
    /// SAFETY: All proper column locks have to be held at the time of calling this method. Also, the index carried by
    /// `ApplyContext` must be properly valid in every columns.
    unsafe fn apply(self, state: &Self::State, context: ApplyContext);
}

impl DeclareContext<'_> {
    pub fn metas<T: Template>() -> Result<Vec<&'static Meta>, Error> {
        let mut metas = Vec::new();
        let context = DeclareContext(&mut metas);
        T::declare(context)?;
        Ok(metas)
    }

    pub fn apply<D: Datum>(&mut self) -> Result<(), Error> {
        let meta = D::meta();
        if self.0.contains(&meta) {
            Err(Error::DuplicateMeta)
        } else {
            self.0.push(meta);
            Ok(())
        }
    }

    pub fn own(&mut self) -> DeclareContext {
        DeclareContext(self.0)
    }
}

impl<'a> InitializeContext<'a> {
    pub fn new(table: &'a Table) -> Self {
        Self(table)
    }

    pub fn own(&self) -> Self {
        Self(self.0)
    }

    pub fn apply<D: Datum>(&self) -> Result<Apply<D>, Error> {
        Ok(Apply(self.0.column::<D>()?, PhantomData))
    }
}

impl<'a> ApplyContext<'a> {
    #[inline]
    pub const fn new(columns: &'a [Column]) -> Self {
        Self(columns, 0)
    }

    #[inline]
    pub const fn own(&self) -> Self {
        Self(self.0, self.1)
    }

    #[inline]
    pub const fn with(&self, index: usize) -> Self {
        Self(self.0, index)
    }

    #[inline]
    pub fn apply<D: Datum>(&self, state: &Apply<D>, value: D) {
        unsafe { get_unchecked(self.0, state.0).set(self.1, value) };
    }
}

unsafe impl<D: Datum> Template for D {
    const SIZE: usize = size_of::<D>();
    type State = Apply<D>;

    fn declare(mut context: DeclareContext) -> Result<(), Error> {
        context.apply::<D>()
    }

    fn initialize(context: InitializeContext) -> Result<Self::State, Error> {
        context.apply::<D>()
    }

    #[inline]
    unsafe fn apply(self, state: &Self::State, context: ApplyContext) {
        context.apply(state, self);
    }
}

unsafe impl Template for () {
    const SIZE: usize = 0;
    type State = ();
    fn declare(_: DeclareContext) -> Result<(), Error> {
        Ok(())
    }
    fn initialize(_: InitializeContext) -> Result<Self::State, Error> {
        Ok(())
    }
    #[inline]
    unsafe fn apply(self, _: &Self::State, _: ApplyContext) {}
}

unsafe impl<T1: Template, T2: Template> Template for (T1, T2) {
    const SIZE: usize = T1::SIZE + T2::SIZE;
    type State = (T1::State, T2::State);

    fn declare(mut context: DeclareContext) -> Result<(), Error> {
        T1::declare(context.own())?;
        T2::declare(context)
    }

    fn initialize(context: InitializeContext) -> Result<Self::State, Error> {
        Ok((
            T1::initialize(context.own())?,
            T2::initialize(context.own())?,
        ))
    }

    #[inline]
    unsafe fn apply(self, state: &Self::State, context: ApplyContext) {
        self.0.apply(&state.0, context.own());
        self.1.apply(&state.1, context.own());
    }
}
