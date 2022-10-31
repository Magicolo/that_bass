use crate::{
    database::Database,
    key::Key,
    resources::Global,
    table::Table,
    template::{ApplyContext, DeclareContext, InitializeContext, Template},
    Error,
};
use parking_lot::{RwLockUpgradableReadGuard, RwLockWriteGuard};
use std::{
    ptr::NonNull,
    sync::{atomic::Ordering::*, Arc},
};

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
        let count = self.templates.len();
        if count == 0 {
            return;
        }

        let inner = self.table.inner.upgradable_read();
        // Ensure that the table's capacity.
        let (start, inner) = {
            let (index, _) = {
                let add = Table::recompose_pending(count as _, 0);
                let pending = inner.pending.fetch_add(add, AcqRel);
                Table::decompose_pending(pending)
            };
            // There can not be more than `u32::MAX` keys at a given time.
            assert!(index < u32::MAX - count as u32);

            let capacity = index as usize + count;
            if capacity <= inner.capacity() {
                (index as usize, RwLockUpgradableReadGuard::downgrade(inner))
            } else {
                let mut inner = RwLockUpgradableReadGuard::upgrade(inner);
                inner.grow(capacity);
                (index as usize, RwLockWriteGuard::downgrade(inner))
            }
        };
        let end = start + count;

        // Initialize table keys.
        unsafe { (&mut **inner.keys.get()).get_unchecked_mut(start..end) }
            .copy_from_slice(&self.keys[..count]);

        // Initialize table rows.
        self.pointers.extend(
            inner
                .stores()
                .iter()
                // SAFETY: Since this row is not yet observable by any thread but this one, no need to take locks.
                .map(|store| unsafe { *store.data().data_ptr() }),
        );
        let context = ApplyContext(&self.pointers, 0);
        for (i, template) in self.templates.drain(..).enumerate() {
            unsafe { template.apply(&self.state, context.with(start + i)) };

            let key = unsafe { *self.keys.get_unchecked(i) };
            let slot = unsafe { self.database.keys().get_unchecked(key) };
            slot.initialize(key.generation(), self.table.index(), (start + i) as u32);
        }
        self.pointers.clear();

        // Try to commit the table count.
        let add = Table::recompose_pending(0, count as _);
        let pending = inner.pending.fetch_add(add, AcqRel);
        let (begun, ended) = Table::decompose_pending(pending);
        debug_assert!(begun >= ended);
        if begun == ended + count as u32 {
            inner.count.fetch_max(begun, Relaxed);
        }

        // Sanity checks.
        debug_assert!(self.templates.is_empty());
        debug_assert!(self.pointers.is_empty());
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
