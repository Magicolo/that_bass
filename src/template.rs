use crate::{
    core::{tuple::tuples, utility::get_unchecked},
    key::Key,
    resources::Resources,
    table::Table,
    Datum, Error, Meta,
};
use std::{marker::PhantomData, sync::Arc};

pub struct Apply<D>(usize, PhantomData<fn(D)>);
pub struct DeclareContext<'a>(&'a mut Vec<&'static Meta>);
pub struct InitializeContext<'a>(&'a Table);
pub struct ApplyContext<'a>(&'a Table, &'a [Key], usize);

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
    /// Returned `metas` are ordered by `meta.identifier()` and deduplicated.
    pub fn metas<T: Template>() -> Result<Vec<&'static Meta>, Error> {
        let mut metas = Vec::new();
        let context = DeclareContext(&mut metas);
        T::declare(context)?;
        Ok(metas)
    }

    pub fn apply<D: Datum>(&mut self) -> Result<(), Error> {
        self.apply_with(D::meta())
    }

    pub fn apply_with(&mut self, meta: &'static Meta) -> Result<(), Error> {
        match self
            .0
            .binary_search_by_key(&meta.identifier(), |meta| meta.identifier())
        {
            Ok(_) => Err(Error::DuplicateMeta),
            Err(index) => Ok(self.0.insert(index, meta)),
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
    pub const fn new(table: &'a Table, keys: &'a [Key]) -> Self {
        Self(table, keys, 0)
    }

    #[inline]
    pub const fn own(&self) -> Self {
        Self(self.0, self.1, self.2)
    }

    #[inline]
    pub const fn with(&self, index: usize) -> Self {
        debug_assert!(index < self.1.len());
        Self(self.0, self.1, index)
    }

    #[inline]
    pub fn key(&self) -> Key {
        unsafe { *get_unchecked(self.1, self.2) }
    }

    #[inline]
    pub fn apply<D: Datum>(&self, state: &Apply<D>, value: D) {
        unsafe { get_unchecked(self.0.columns(), state.0).set(self.2, value) };
    }
}

impl<T: Template> ShareMeta<T> {
    pub fn from(resources: &Resources) -> Result<Arc<Box<[&'static Meta]>>, Error> {
        let share = resources.try_global(|| {
            let metas = DeclareContext::metas::<T>()?;
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

unsafe impl<T: 'static> Template for PhantomData<T> {
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

pub struct With<T, F>(F, PhantomData<fn(T)>);

#[inline]
pub const fn with<T, F: FnOnce(Key) -> T>(with: F) -> With<T, F> {
    With(with, PhantomData)
}

unsafe impl<T: Template, F: FnOnce(Key) -> T + 'static> Template for With<T, F> {
    type State = T::State;

    fn declare(context: DeclareContext) -> Result<(), Error> {
        T::declare(context)
    }
    fn initialize(context: InitializeContext) -> Result<Self::State, Error> {
        T::initialize(context)
    }
    #[inline]
    unsafe fn apply(self, state: &Self::State, context: ApplyContext) {
        self.0(context.key()).apply(state, context)
    }
}
