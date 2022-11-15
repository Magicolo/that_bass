use crate::{
    event::Listen,
    key::Key,
    resources::Resources,
    table::{Table, Tables},
    template::{ApplyContext, InitializeContext, ShareMeta, Template},
    Database, Error,
};
use parking_lot::RwLockUpgradableReadGuard;
use std::{
    num::NonZeroUsize,
    sync::{atomic::Ordering, Arc},
};

pub struct Create<'d, T: Template, L> {
    database: &'d Database<L>,
    state: Arc<T::State>,
    table: Arc<Table>,
    keys: Vec<Key>,
    templates: Vec<T>,
}

struct Share<T: Template>(Arc<T::State>, Arc<Table>);

impl<L> Database<L> {
    pub fn create<T: Template>(&self) -> Result<Create<T, L>, Error> {
        Share::<T>::from(self.tables(), self.resources()).map(|(state, table)| Create {
            database: self,
            state,
            table,
            keys: Vec::new(),
            templates: Vec::new(),
        })
    }
}

impl<'d, T: Template, L> Create<'d, T, L> {
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
        self.database.keys().reserve(&mut keys);
        self.keys.extend_from_slice(&keys);
        self.templates.extend(templates);
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

    #[inline]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (Key, &T)> {
        self.keys.iter().copied().zip(self.templates.iter())
    }

    pub fn drain(&mut self) -> impl ExactSizeIterator<Item = (Key, T)> + '_ {
        self.database.keys().recycle(self.keys.iter().copied());
        self.keys.drain(..).zip(self.templates.drain(..))
    }

    pub fn clear(&mut self) {
        self.database.keys().recycle(self.keys.drain(..));
        self.templates.clear();
    }
}

impl<T: Template, L: Listen> Create<'_, T, L> {
    /// Resolves the accumulated create operations.
    ///
    /// In order to prevent deadlocks, **do not call this method while using a `Query`** unless you can
    /// guarantee that there are no overlaps in table usage between this `Create` and the `Query`.
    pub fn resolve(&mut self) -> usize {
        debug_assert_eq!(self.keys.len(), self.templates.len());
        let Some(count) = NonZeroUsize::new(self.keys.len()) else {
            return 0;
        };

        // The upgradable lock serves 2 purposes:
        // 1- It can be upgraded to grow the columns if required.
        // 2- It prevents this operation from overlaping with the end of `destroy/remove` operations, which is important since `create`
        // writes to keys and columns without locking.
        let keys = self.table.keys.upgradable_read();
        let (start, keys) = self.table.reserve(keys, count);
        let context = ApplyContext::new(&self.table, &keys);
        for (i, template) in self.templates.drain(..).enumerate() {
            debug_assert!(i < count.get());
            // SAFETY: This is safe by the guarantees of `T::apply` and by the fact that only the rows in the range
            // `start..start + count` are modified.
            // SAFETY: No locks are required because of the guarantee above and the `start + i` row is guaranteed to hold no previous
            // valid data.
            unsafe { template.apply(&self.state, context.with(start + i)) };
        }
        {
            // SAFETY: The range `start..start + count` is reserved to this thread by `table::Inner::reserve` and will not be modified
            // until `table::Inner::commit` is called and as long as a table lock is held.
            let keys = unsafe { &mut *RwLockUpgradableReadGuard::rwlock(&keys).data_ptr() };
            keys[start..start + count.get()].copy_from_slice(&self.keys);
        }

        /*
            Initialize table keys.
            - Calling `initialize` first means that another thread may try to use a `Query::find` and access a `row >= count`. Although
            this is an uncomfortable state of affairs (because it is easy to forget), queries can detect this state and report an
            appropriate error.
            - Calling `fetch_add` first means that another thread may query the created rows and observe that their key in invalid through
            `Database::keys().get()`. This is not acceptable and there doesn't seem to be a good fix.
        */
        self.database
            .keys()
            .initialize(&keys, self.table.index(), start..start + count.get());
        self.table
            .count
            .fetch_add(count.get() as _, Ordering::Release);
        self.database
            .listen
            .on_create(&keys[start..start + count.get()], &self.table);
        drop(keys);
        self.keys.clear();
        debug_assert_eq!(self.keys.len(), 0);
        debug_assert_eq!(self.templates.len(), 0);
        count.get()
    }
}

impl<T: Template, L> Drop for Create<'_, T, L> {
    fn drop(&mut self) {
        self.clear();
    }
}

impl<T: Template> Share<T> {
    pub fn from(
        tables: &Tables,
        resources: &Resources,
    ) -> Result<(Arc<T::State>, Arc<Table>), Error> {
        let metas = ShareMeta::<T>::from(resources)?;
        let share = resources.try_global(|| {
            let table = tables.find_or_add(&metas);
            let context = InitializeContext::new(&table);
            let state = Arc::new(T::initialize(context)?);
            Ok(Share::<T>(state, table))
        })?;
        let share = share.read();
        Ok((share.0.clone(), share.1.clone()))
    }
}

pub(crate) fn is<T: Template>(table: &Table, tables: &Tables, resources: &Resources) -> bool {
    match Share::<T>::from(tables, resources) {
        Ok(pair) => table.index() == pair.1.index(),
        Err(_) => false,
    }
}

pub(crate) fn has<T: Template>(table: &Table, tables: &Tables, resources: &Resources) -> bool {
    match Share::<T>::from(tables, resources) {
        Ok(pair) => table.has_all(pair.1.metas().map(|meta| meta.identifier())),
        Err(_) => false,
    }
}
