use crate::{
    key::Key,
    table::{self, Table},
    template::{ApplyContext, DeclareContext, InitializeContext, Template},
    Database, Error, Listen,
};
use std::{num::NonZeroUsize, sync::Arc};

pub struct Create<'d, T: Template> {
    database: &'d Database,
    state: Arc<T::State>,
    table: Arc<Table>,
    keys: Vec<Key>,
    templates: Vec<T>,
}

struct Share<T: Template>(Arc<T::State>, Arc<Table>);

impl Database {
    pub fn create<T: Template>(&self) -> Result<Create<T>, Error> {
        let (state, table) = Share::<T>::from(self)?;
        Ok(Create {
            database: self,
            state,
            table,
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
    pub fn clones(&mut self, count: usize, template: &T) -> &[Key]
    where
        T: Clone,
    {
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
    pub fn clones_n<const N: usize>(&mut self, template: &T) -> [Key; N]
    where
        T: Clone,
    {
        self.with_n(|| template.clone())
    }

    #[inline]
    pub fn defaults_n<const N: usize>(&mut self) -> [Key; N]
    where
        T: Default,
    {
        self.with_n(T::default)
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (Key, &T)> {
        self.templates
            .iter()
            .zip(&self.keys)
            .map(|pair| (*pair.1, pair.0))
    }

    pub fn drain(&mut self) -> impl ExactSizeIterator<Item = (Key, T)> + '_ {
        let keys = self.database.keys();
        keys.recycle(self.keys[..self.templates.len()].iter().copied());
        self.templates
            .drain(..)
            .zip(&self.keys)
            .map(|pair| (*pair.1, pair.0))
    }

    pub fn clear(&mut self) {
        let keys = self.database.keys();
        keys.recycle(self.keys[..self.templates.len()].iter().copied());
        self.templates.clear();
    }

    /// Resolves the accumulated create operations.
    ///
    /// In order to prevent deadlocks, **do not call this method while using a `Query`** unless you can
    /// guarantee that there are no overlaps in table usage between this `Create` and the `Query`.
    pub fn resolve(&mut self) -> usize {
        let Some(count) = NonZeroUsize::new(self.templates.len()) else {
            return 0;
        };

        let (start, inner) = table::Inner::reserve(self.table.inner.upgradable_read(), count);
        // SAFETY: The range `start..start + count` is reserved to this thread by `table::Inner::reserve` and will not be modified
        // until `table::Inner::commit` is called and as long as a table lock is held.
        let keys = unsafe { &mut **inner.keys.get() };
        keys[start..start + count.get()].copy_from_slice(&self.keys[..count.get()]);
        let context = ApplyContext::new(keys, inner.columns());
        for (i, template) in self.templates.drain(..).enumerate() {
            // SAFETY: This is safe by the guarantees of `T::apply` and by the fact that only the rows in the range
            // `start..start + count` are modified.
            let index = start + i;
            debug_assert!(index < start + count.get());
            // SAFETY: No locks are required because of the guarantee above and `index` is guaranteed to hold no previous
            // valid data.
            unsafe { template.apply(&self.state, context.with(index)) };
        }

        // Initialize table keys.
        self.database
            .keys()
            .initialize(keys, self.table.index(), start..start + count.get());
        // Emit a `create` event before
        self.database
            .listen
            .created(&self.keys[..count.get()], &self.table);
        inner.commit(count);
        count.get()
    }
}

impl<T: Template> Drop for Create<'_, T> {
    fn drop(&mut self) {
        self.clear();
    }
}

impl<T: Template> Share<T> {
    pub fn from(database: &Database) -> Result<(Arc<T::State>, Arc<Table>), Error> {
        let share = database.resources().try_global(|| {
            let metas = DeclareContext::metas::<T>()?;
            let table = database.tables().find_or_add(&metas);
            let context = InitializeContext::new(&table);
            let state = Arc::new(T::initialize(context)?);
            Ok(Share::<T>(state, table))
        })?;
        let share = share.read();
        Ok((share.0.clone(), share.1.clone()))
    }
}

pub(crate) fn is<T: Template>(table: &Table, database: &Database) -> bool {
    match Share::<T>::from(database) {
        Ok(pair) => table.index() == pair.1.index(),
        Err(_) => false,
    }
}

pub(crate) fn has<T: Template>(table: &Table, database: &Database) -> bool {
    match Share::<T>::from(database) {
        Ok(pair) => table.has_all(pair.1.metas().iter().map(|meta| meta.identifier())),
        Err(_) => false,
    }
}
