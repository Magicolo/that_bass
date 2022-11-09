use crate::{
    core::{tuples, utility::get_unchecked},
    table::{Column, Table},
    Database, Datum, Error, Meta,
};
use std::{marker::PhantomData, sync::Arc};

pub struct Apply<D>(usize, PhantomData<fn(D)>);
pub struct DeclareContext<'a>(&'a mut Vec<&'static Meta>);
pub struct InitializeContext<'a>(&'a Table);
pub struct ApplyContext<'a>(&'a [Column], usize);

pub unsafe trait Template: 'static {
    type State: Send + Sync;
    fn declare(context: DeclareContext) -> Result<(), Error>;
    fn initialize(context: InitializeContext) -> Result<Self::State, Error>;
    /// SAFETY: All proper column locks have to be held at the time of calling this method. Also, the index carried by
    /// `ApplyContext` must be properly valid in every columns.
    unsafe fn apply(self, state: &Self::State, context: ApplyContext);
}

pub(crate) struct ShareMeta<T>(Arc<Box<[&'static Meta]>>, PhantomData<fn(T)>);

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
        Ok(Apply(self.0.column::<D>()?.0, PhantomData))
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

impl<T: Template> ShareMeta<T> {
    pub fn from(database: &Database) -> Result<Arc<Box<[&'static Meta]>>, Error> {
        let share = database.resources().try_global(|| {
            let mut metas = DeclareContext::metas::<T>()?;
            // Must sort here since the order of these metas is used to lock columns in target tables.
            metas.sort_unstable_by_key(|meta| meta.identifier());
            Ok(Self(Arc::new(metas.into_boxed_slice()), PhantomData))
        })?;
        let share = share.read();
        Ok(share.0.clone())
    }
}

unsafe impl<D: Datum> Template for D {
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

macro_rules! tuple {
    ($n:ident, $c:expr $(, $p:ident, $t:ident, $i:tt)*) => {
        unsafe impl<$($t: Template,)*> Template for ($($t,)*) {
            type State = ($($t::State,)*);

            fn declare(mut _context: DeclareContext) -> Result<(), Error> {
                $($t::declare(_context.own())?;)*
                Ok(())
            }

            fn initialize(_context: InitializeContext) -> Result<Self::State, Error> {
                Ok(($($t::initialize(_context.own())?,)*))
            }

            #[inline]
            unsafe fn apply(self, _state: &Self::State, _context: ApplyContext) {
                $(self.$i.apply(&_state.$i, _context.own());)*
            }
        }
    };
}
tuples!(tuple);
