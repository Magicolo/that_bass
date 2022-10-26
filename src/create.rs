use crate::{
    database::Database,
    key::Key,
    table::{Store, Table},
    Datum, Error, Meta,
};
use std::{any::TypeId, collections::HashSet, sync::Arc};

pub unsafe trait Template: 'static {
    type State: Send + Sync;
    fn declare(context: Context) -> Result<(), Error>;
    fn initialize(table: &Table) -> Result<Self::State, Error>;
    unsafe fn apply(self, state: &Self::State, row: usize, stores: &[Store]);
}

pub struct Create<'d, T: Template> {
    database: &'d Database,
    state: Arc<T::State>,
    table: Arc<Table>,
    keys: Vec<Key>,
    templates: Vec<T>,
}

pub struct Context<'a> {
    types: &'a mut HashSet<TypeId>,
    metas: &'a mut Vec<&'static Meta>,
}

impl Database {
    pub fn create<T: Template>(&self) -> Result<Create<T>, Error> {
        struct Shared<T: Template>(Arc<T::State>, Arc<Table>);

        let shared = self.resources().try_global(|| {
            let mut metas = Vec::new();
            let context = Context {
                types: &mut HashSet::new(),
                metas: &mut metas,
            };
            T::declare(context)?;
            let mut types = HashSet::new();
            if metas.iter().all(|meta| types.insert(meta.identifier())) {
                metas.sort_unstable_by_key(|meta| meta.identifier());
                let table = self.tables().find_or_add(metas, 0);
                let state = Arc::new(T::initialize(&table)?);
                Ok(Shared::<T>(state, table))
            } else {
                Err(Error::DuplicateMeta)
            }
        })?;
        let Shared(state, table) = &*shared.read();
        Ok(Create {
            database: self,
            state: state.clone(),
            table: table.clone(),
            keys: Vec::new(),
            templates: Vec::new(),
        })
    }
}

impl<'d, T: Template> Create<'d, T> {
    #[inline]
    pub fn all<I: IntoIterator<Item = T>>(&mut self, templates: I) -> &[Key] {
        let start = self.templates.len();
        self.templates.extend(templates);
        self.keys.resize(self.templates.len(), Key::NULL);
        self.database.keys().reserve(&mut self.keys[start..]);
        &self.keys[start..]
    }

    #[inline]
    pub fn with(&mut self, count: usize, mut with: impl FnMut() -> T) -> &[Key] {
        self.all((0..count).map(|_| with()))
    }

    #[inline]
    pub fn clones(&mut self, count: usize, template: T) -> &[Key]
    where
        T: Clone,
    {
        // TODO: Do not clone the last template.
        self.with(count, || template.clone())
    }

    #[inline]
    pub fn defaults(&mut self, count: usize) -> &[Key]
    where
        T: Default,
    {
        self.with(count, T::default)
    }

    #[inline]
    pub fn all_n<const N: usize>(&mut self, templates: [T; N]) -> [Key; N] {
        let mut keys = [Key::NULL; N];
        self.templates.extend(templates);
        self.database.keys().reserve(&mut keys);
        self.keys.extend_from_slice(&keys);
        keys
    }

    #[inline]
    pub fn one(&mut self, template: T) -> Key {
        self.all_n([template])[0]
    }

    #[inline]
    pub fn with_n<const N: usize>(&mut self, mut with: impl FnMut() -> T) -> [Key; N] {
        self.all_n([(); N].map(|_| with()))
    }

    #[inline]
    pub fn clones_n<const N: usize>(&mut self, template: T) -> [Key; N]
    where
        T: Clone,
    {
        // TODO: Do not clone the last template.
        self.with_n(|| template.clone())
    }

    #[inline]
    pub fn defaults_n<const N: usize>(&mut self) -> [Key; N]
    where
        T: Default,
    {
        self.with_n(T::default)
    }

    /// Resolves the accumulated create operations.
    ///
    /// In order to prevent deadlocks, **do not call this method while using a `Query`** unless you can
    /// guarantee that there are no overlaps in table usage between this `Create` and the `Query`.
    #[inline]
    pub fn resolve(&mut self) {
        apply(
            self.database,
            &self.table,
            &*self.state,
            &self.keys[..self.templates.len()],
            self.templates.drain(..),
        )
    }

    #[inline]
    pub fn clear(&mut self) {
        let keys = self.database.keys();
        keys.release(self.keys[..self.templates.len()].iter().copied());
        self.templates.clear();
    }
}

impl<T: Template> Drop for Create<'_, T> {
    fn drop(&mut self) {
        self.resolve();
    }
}

impl Context<'_> {
    pub fn declare(&mut self, meta: &'static Meta) -> Result<(), Error> {
        if self.types.insert(meta.identifier()) {
            self.metas.push(meta);
            Ok(())
        } else {
            Err(Error::DuplicateMeta)
        }
    }

    pub fn own(&mut self) -> Context {
        Context {
            types: self.types,
            metas: self.metas,
        }
    }
}

unsafe impl<D: Datum> Template for D {
    type State = usize;

    fn declare(mut context: Context) -> Result<(), Error> {
        context.declare(D::meta())
    }

    fn initialize(table: &Table) -> Result<Self::State, Error> {
        table.store::<D>()
    }

    #[inline]
    unsafe fn apply(self, &state: &Self::State, row: usize, stores: &[Store]) {
        unsafe { stores.get_unchecked(state).set(row, self) };
    }
}

unsafe impl Template for () {
    type State = ();
    fn declare(_: Context) -> Result<(), Error> {
        Ok(())
    }
    fn initialize(_: &Table) -> Result<Self::State, Error> {
        Ok(())
    }
    #[inline]
    unsafe fn apply(self, _: &Self::State, _: usize, _: &[Store]) {}
}

unsafe impl<T1: Template, T2: Template> Template for (T1, T2) {
    type State = (T1::State, T2::State);

    fn declare(mut context: Context) -> Result<(), Error> {
        T1::declare(context.own())?;
        T2::declare(context)
    }

    fn initialize(table: &Table) -> Result<Self::State, Error> {
        Ok((T1::initialize(table)?, T2::initialize(table)?))
    }

    #[inline]
    unsafe fn apply(self, state: &Self::State, row: usize, stores: &[Store]) {
        self.0.apply(&state.0, row, stores);
        self.1.apply(&state.1, row, stores);
    }
}

fn apply<T: Template, I: ExactSizeIterator<Item = T>>(
    database: &Database,
    table: &Table,
    state: &T::State,
    keys: &[Key],
    mut templates: I,
) {
    if keys.len() == 0 {
        return;
    }

    debug_assert_eq!(templates.len(), keys.len());
    database.add_to_table(keys, table, |range, stores| {
        for index in range {
            match templates.next() {
                Some(template) => unsafe { template.apply(state, index, stores) },
                None => unreachable!(),
            }
        }
    });
    debug_assert!(templates.next().is_none());
}
