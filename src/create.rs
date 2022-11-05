use crate::{
    key::Key,
    resources::Global,
    table::{self, Table},
    template::{ApplyContext, DeclareContext, InitializeContext, Template},
    Database, Error,
};
use std::{num::NonZeroUsize, ptr::NonNull, sync::Arc};

pub struct Create<'d, T: Template> {
    database: &'d Database,
    state: Arc<T::State>,
    table: Arc<Table>,
    keys: Vec<Key>,
    templates: Vec<T>,
    pointers: Vec<NonNull<()>>,
}

struct Share<T: Template>(Arc<T::State>, Arc<Table>);

impl<T: Template> Share<T> {
    pub fn from(database: &Database) -> Result<Global<Share<T>>, Error> {
        database.resources().try_global(|| {
            let metas = DeclareContext::metas::<T>()?;
            let table = database.tables().find_or_add(metas, 0);
            let indices = table
                .types()
                .enumerate()
                .map(|pair| (pair.1, pair.0))
                .collect();
            let context = InitializeContext(&indices);
            let state = Arc::new(T::initialize(context)?);
            Ok(Share::<T>(state, table))
        })
    }
}

impl Database {
    pub fn create<T: Template>(&self) -> Result<Create<T>, Error> {
        let share = Share::<T>::from(self)?;
        let share = share.read();
        Ok(Create {
            database: self,
            state: share.0.clone(),
            table: share.1.clone(),
            keys: Vec::new(),
            templates: Vec::new(),
            pointers: Vec::new(),
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
    pub fn clone(&mut self, template: &T) -> Key
    where
        T: Clone,
    {
        self.one(template.clone())
    }

    #[inline]
    pub fn default(&mut self) -> Key
    where
        T: Default,
    {
        self.one(T::default())
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

    /// Resolves the accumulated create operations.
    ///
    /// In order to prevent deadlocks, **do not call this method while using a `Query`** unless you can
    /// guarantee that there are no overlaps in table usage between this `Create` and the `Query`.
    pub fn resolve(&mut self) {
        let Some(count) = NonZeroUsize::new(self.templates.len()) else {
            return;
        };

        let (start, inner) = table::Inner::reserve(self.table.inner.upgradable_read(), count);
        let keys = unsafe { &mut **inner.keys.get() };
        keys[start..start + count.get()].copy_from_slice(&self.keys[..count.get()]);
        if T::SIZE == 0 {
            // No data to initialize.
            self.templates.clear();
        } else {
            // Initialize table rows.
            for store in inner.stores() {
                // SAFETY: Since this row is not yet observable by any thread but this one, no need to take locks.
                self.pointers.push(unsafe { *store.data().data_ptr() });
            }
            let context = ApplyContext(&self.pointers, 0);
            for (i, template) in self.templates.drain(..).enumerate() {
                unsafe { template.apply(&self.state, context.with(start + i)) };
            }
            self.pointers.clear();
        }
        // Initialize table keys.
        self.database
            .keys()
            .initialize(keys, self.table.index(), start..start + count.get());
        inner.commit(count);

        // Sanity checks.
        debug_assert!(self.templates.is_empty());
        debug_assert!(self.pointers.is_empty());
    }

    #[inline]
    pub fn clear(&mut self) {
        let keys = self.database.keys();
        keys.recycle(self.keys[..self.templates.len()].iter().copied());
        self.templates.clear();
    }
}

impl<T: Template> Drop for Create<'_, T> {
    fn drop(&mut self) {
        self.resolve();
    }
}

pub(crate) fn is<T: Template>(table: &Table, database: &Database) -> bool {
    match Share::<T>::from(database) {
        Ok(share) => share.read().1.index() == table.index(),
        Err(_) => false,
    }
}

pub(crate) fn has<T: Template>(table: &Table, database: &Database) -> bool {
    match Share::<T>::from(database) {
        Ok(shared) => shared
            .read()
            .1
            .types()
            .all(|identifier| table.has_with(identifier)),
        Err(_) => false,
    }
}
