use parking_lot::RwLock;

use crate::{
    database::{Database, Inner},
    key::Key,
    table::Table,
    utility::FullIterator,
    Datum, Error, Meta,
};
use std::{collections::HashSet, marker::PhantomData, sync::Arc};

pub unsafe trait Template {
    type State;
    fn initialize(context: Context) -> Self::State;
    unsafe fn apply(self, state: &Self::State, row: usize, table: &Table);
}

pub struct Spawn<T: Template>(PhantomData<T>);
pub struct With<T: Template, F: FnMut(Key) -> T>(F, PhantomData<T>);

pub struct Create<'a, T: Template> {
    inner: &'a Inner,
    state: T::State,
    table: Arc<RwLock<Table>>,
    keys: Vec<Key>,
    templates: Vec<T>,
    _marker: PhantomData<T>,
}

pub struct Context<'a> {
    metas: &'a mut Vec<Meta>,
}

impl Database {
    pub fn create<T: Template>(&self) -> Result<Create<T>, Error> {
        let mut metas = Vec::new();
        let context = Context { metas: &mut metas };
        let state = T::initialize(context);
        let inner = self.inner();
        let mut types = HashSet::new();
        if metas.iter().all(|meta| types.insert(meta.identifier)) {
            metas.sort_unstable_by_key(|meta| meta.identifier);
            let table = inner.tables.find_or_add(metas, types, 0);
            Ok(Create {
                inner,
                state,
                table,
                keys: Vec::new(),
                templates: Vec::new(),
                _marker: PhantomData,
            })
        } else {
            Err(Error::DuplicateMeta)
        }
    }
}

impl Context<'_> {
    pub fn declare(&mut self, meta: Meta) {
        self.metas.push(meta);
    }

    pub fn own(&mut self) -> Context {
        Context { metas: self.metas }
    }
}

unsafe impl<D: Datum> Template for D {
    type State = ();

    fn initialize(mut context: Context) -> Self::State {
        context.declare(D::meta());
    }

    #[inline]
    unsafe fn apply(self, _: &Self::State, row: usize, table: &Table) {
        unsafe { table.stores.get_unchecked(0).write_unlocked_at(row, self) };
    }
}

unsafe impl Template for () {
    type State = ();
    #[inline]
    fn initialize(_: Context) -> Self::State {}
    #[inline]
    unsafe fn apply(self, _: &Self::State, _: usize, _: &Table) {}
}

unsafe impl<T1: Template, T2: Template> Template for (T1, T2) {
    type State = (T1::State, T2::State);

    fn initialize(mut context: Context) -> Self::State {
        (T1::initialize(context.own()), T2::initialize(context.own()))
    }

    #[inline]
    unsafe fn apply(self, state: &Self::State, row: usize, table: &Table) {
        self.0.apply(&state.0, row, table);
        self.1.apply(&state.1, row, table);
    }
}

impl<T: Template> Create<'_, T> {
    pub fn all_n<const N: usize>(&self, templates: [T; N]) -> [Key; N] {
        let mut keys = [Key::NULL; N];
        Self::apply(
            &self.inner,
            &self.state,
            &self.table,
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
        self.all_n([(); N].map(|_| template.clone()))
    }

    #[inline]
    pub fn defaults_n<const N: usize>(&self) -> [Key; N]
    where
        T: Default,
    {
        self.all_n([(); N].map(|_| T::default()))
    }

    pub fn all<I: IntoIterator<Item = T>>(&mut self, templates: I) -> &[Key] {
        self.templates.extend(templates);
        self.keys.resize(self.templates.len(), Key::NULL);
        Self::apply(
            &self.inner,
            &self.state,
            &self.table,
            self.templates.drain(..),
            &mut self.keys,
        );
        &self.keys
    }

    #[inline]
    pub fn clones(&mut self, count: usize, template: T) -> &[Key]
    where
        T: Clone,
    {
        self.all((0..count).map(|_| template.clone()))
    }

    #[inline]
    pub fn defaults(&mut self, count: usize) -> &[Key]
    where
        T: Default,
    {
        self.all((0..count).map(|_| T::default()))
    }

    fn apply<I: FullIterator<Item = T>>(
        inner: &Inner,
        state: &T::State,
        table: &RwLock<Table>,
        mut templates: I,
        keys: &mut [Key],
    ) {
        debug_assert_eq!(templates.len(), keys.len());
        inner
            .create(table, keys, |row, table| unsafe {
                for i in 0..row.1 {
                    templates
                        .next()
                        .expect("Expected initialize to be called once per template.")
                        .apply(state, row.0 + i, table)
                }
            })
            .expect("Expected create to succeed.");
        debug_assert!(templates.next().is_none());
    }
}
