use crate::{
    database::Database,
    key::Key,
    table::{Add, Store, Table},
    Datum, Error, Meta,
};
use std::{any::TypeId, collections::HashSet, marker::PhantomData, sync::Arc};

pub unsafe trait Template: 'static {
    type State: Send + Sync;
    fn declare(context: Context) -> Result<(), Error>;
    fn initialize(table: &Table) -> Result<Self::State, Error>;
    unsafe fn apply(self, state: &Self::State, row: usize, stores: &[Store]);
}

pub struct Create<'a, T: Template> {
    database: &'a Database,
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
        struct Shared<T: Template>(Arc<T::State>, Arc<Table>, PhantomData<fn(T)>);

        let shared = self.resources().global(|| {
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
                Ok(Shared::<T>(state, table, PhantomData))
            } else {
                Err(Error::DuplicateMeta)
            }
        })?;
        let shared = shared.read();
        Ok(Create {
            database: self,
            state: shared.0.clone(),
            table: shared.1.clone(),
            keys: Vec::new(),
            templates: Vec::new(),
        })
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
        unsafe { stores.get_unchecked(state).set_unlocked_at(row, self) };
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

impl<T: Template> Create<'_, T> {
    #[inline]
    pub fn all_n<const N: usize>(&self, templates: [T; N]) -> [Key; N] {
        let mut keys = [Key::NULL; N];
        apply(
            self.database,
            &self.table,
            &self.state,
            templates.into_iter(),
            &mut keys,
        );
        keys
    }

    #[inline]
    pub fn one(&self, template: T) -> Key {
        self.all_n([template])[0]
    }

    #[inline]
    pub fn clones_n<const N: usize>(&self, template: T) -> [Key; N]
    where
        T: Clone,
    {
        // TODO: Do not clone the last template.
        self.all_n([(); N].map(|_| template.clone()))
    }

    #[inline]
    pub fn defaults_n<const N: usize>(&self) -> [Key; N]
    where
        T: Default,
    {
        self.all_n([(); N].map(|_| T::default()))
    }

    #[inline]
    pub fn all<I: IntoIterator<Item = T>>(&mut self, templates: I) -> &[Key] {
        self.templates.extend(templates);
        self.keys.resize(self.templates.len(), Key::NULL);
        apply(
            self.database,
            &self.table,
            &self.state,
            self.templates.drain(..),
            &mut self.keys,
        );
        &self.keys
    }

    #[inline]
    pub fn with(&mut self, count: usize, with: impl FnMut(usize) -> T) -> &[Key] {
        self.all((0..count).map(with))
    }

    #[inline]
    pub fn clones(&mut self, count: usize, template: T) -> &[Key]
    where
        T: Clone,
    {
        // TODO: Do not clone the last template.
        self.with(count, |_| template.clone())
    }

    #[inline]
    pub fn defaults(&mut self, count: usize) -> &[Key]
    where
        T: Default,
    {
        self.with(count, |_| T::default())
    }
}

#[inline]
fn apply<T: Template, I: ExactSizeIterator<Item = T>>(
    database: &Database,
    table: &Table,
    state: &Arc<T::State>,
    templates: I,
    keys: &mut [Key],
) {
    debug_assert_eq!(templates.len(), keys.len());
    database.add_to_table(
        keys,
        table,
        (state, templates),
        |(state, templates), mut row_index, stores| {
            for template in templates {
                unsafe { template.apply(state, row_index as _, stores) };
                row_index += 1;
            }
        },
        |(state, templates), keys, mut row_index, row_count| {
            let state = state.clone();
            let templates = templates.collect::<Vec<_>>();
            Add {
                keys: keys.iter().copied().collect(),
                row_index,
                row_count,
                initialize: Box::new(move |stores| {
                    for template in templates {
                        unsafe { template.apply(&state, row_index as _, stores) }
                        row_index += 1;
                    }
                }),
            }
        },
    );
}
