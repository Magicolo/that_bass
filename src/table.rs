use crate::{
    core::{
        iterate::FullIterator,
        slice::{self, Slice},
        utility::{get_unchecked, sorted_contains},
    },
    key::Key,
    Database, Datum, Error, Meta,
};
use parking_lot::{RwLock, RwLockUpgradableReadGuard, RwLockWriteGuard};
use std::{
    any::TypeId,
    num::NonZeroUsize,
    ptr::NonNull,
    slice::from_raw_parts_mut,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

pub struct Table {
    index: u32,
    pub(crate) count: AtomicUsize,
    pub(crate) keys: RwLock<Vec<Key>>,
    /// Columns are ordered consistently between tables.
    pub(crate) columns: Box<[Column]>,
}

pub struct Column {
    meta: &'static Meta,
    data: RwLock<NonNull<()>>,
}

pub(crate) struct State {
    tables: Slice<Arc<Table>>,
}

#[derive(Clone)]
pub struct Tables<'a>(slice::Guard<'a, Arc<Table>>);

impl Database {
    #[inline]
    pub fn tables(&self) -> Tables {
        Tables(self.tables.tables.guard())
    }
}

impl Column {
    pub(crate) fn new(meta: &'static Meta, capacity: usize) -> Self {
        Self {
            data: RwLock::new(unsafe { (meta.new)(capacity) }),
            meta,
        }
    }

    #[inline]
    pub const fn meta(&self) -> &'static Meta {
        self.meta
    }

    #[inline]
    pub(crate) const fn data(&self) -> &RwLock<NonNull<()>> {
        &self.data
    }

    #[inline]
    pub(crate) unsafe fn copy_to(
        source: (&Self, usize),
        target: (&Self, usize),
        count: NonZeroUsize,
    ) {
        debug_assert_eq!(source.0.meta().identifier(), target.0.meta().identifier());
        let &Meta { copy, .. } = source.0.meta();
        copy(
            (*source.0.data.data_ptr(), source.1),
            (*target.0.data.data_ptr(), target.1),
            count,
        );
    }

    pub(crate) unsafe fn grow(&self, old_capacity: usize, new_capacity: NonZeroUsize) {
        debug_assert!(old_capacity < new_capacity.get());
        let &Meta {
            new, free, copy, ..
        } = self.meta();
        let data = &mut *self.data.data_ptr();
        let old_data = *data;
        let new_data = new(new_capacity.get());
        if let Some(old_capacity) = NonZeroUsize::new(old_capacity) {
            copy((old_data, 0), (new_data, 0), old_capacity);
        }
        // A count of 0 is sent to `free` because the values of `old_data` have been moved to `new_data`, so they must not be dropped.
        free(old_data, 0, old_capacity);
        *data = new_data;
    }

    pub(crate) unsafe fn shrink(&mut self, old_capacity: NonZeroUsize, new_capacity: usize) {
        debug_assert!(old_capacity.get() > new_capacity);
        let &Meta {
            new,
            free,
            copy,
            drop,
            ..
        } = self.meta();
        let data = self.data.get_mut();
        let old_data = *data;
        let new_data = new(new_capacity);
        if let Some(new_capacity) = NonZeroUsize::new(new_capacity) {
            copy((old_data, 0), (new_data, 0), new_capacity);
        }
        if let Some(over) = NonZeroUsize::new(old_capacity.get() - new_capacity) {
            drop.1(old_data, new_capacity, over);
        }
        // A count of 0 is sent to `free` because the values of `old_data` have been moved to `new_data`, so they must not be dropped.
        free(old_data, 0, old_capacity.get());
        *data = new_data;
    }

    /// SAFETY: Both the 'source' and 'target' indices must be within the bounds of the column.
    /// The ranges 'source_index..source_index + count' and 'target_index..target_index + count' must not overlap.
    #[inline]
    pub(crate) unsafe fn squash(
        &self,
        source_index: usize,
        target_index: usize,
        count: NonZeroUsize,
    ) {
        let &Meta { copy, drop, .. } = self.meta();
        let data = unsafe { *self.data.data_ptr() };
        drop.1(data, target_index, count);
        copy((data, source_index), (data, target_index), count);
    }

    /// SAFETY: Both the 'source' and 'target' indices must be within the bounds of the column.
    /// The ranges 'source_index..source_index + count' and 'target_index..target_index + count' must not overlap.
    #[inline]
    pub(crate) unsafe fn copy(
        &self,
        source_index: usize,
        target_index: usize,
        count: NonZeroUsize,
    ) {
        let &Meta { copy, .. } = self.meta();
        let data = *self.data.data_ptr();
        copy((data, source_index), (data, target_index), count);
    }

    #[inline]
    pub(crate) unsafe fn drop(&self, index: usize, count: NonZeroUsize) {
        let &Meta { drop, .. } = self.meta();
        let data = unsafe { *self.data.data_ptr() };
        drop.1(data, index, count);
    }

    #[inline]
    pub(crate) unsafe fn free(&mut self, count: usize, capacity: usize) {
        let &Meta { free, .. } = self.meta();
        let data = *self.data.get_mut();
        free(data, count, capacity);
    }

    #[inline]
    pub(crate) unsafe fn get<T: 'static>(&self, index: usize) -> &mut T {
        debug_assert_eq!(TypeId::of::<T>(), self.meta().identifier());
        let data = *self.data.data_ptr();
        &mut *data.as_ptr().cast::<T>().add(index)
    }

    #[inline]
    pub(crate) unsafe fn get_all<T: 'static>(&self, count: usize) -> &mut [T] {
        debug_assert_eq!(TypeId::of::<T>(), self.meta().identifier());
        let data = *self.data.data_ptr();
        from_raw_parts_mut(data.as_ptr().cast::<T>(), count)
    }

    #[inline]
    pub(crate) unsafe fn set<T: 'static>(&self, index: usize, value: T) {
        debug_assert_eq!(TypeId::of::<T>(), self.meta().identifier());
        let data = *self.data.data_ptr();
        data.as_ptr().cast::<T>().add(index).write(value);
    }
}

impl Tables<'_> {
    #[inline]
    pub fn update(&mut self) {
        self.0.update();
    }

    #[inline]
    pub fn len(&mut self) -> usize {
        self.0.get().len()
    }

    #[inline]
    pub fn iter(&mut self) -> impl FullIterator<Item = &Table> {
        self.0.get().iter().map(|table| &**table)
    }

    #[inline]
    pub fn get(&mut self, index: usize) -> Result<&Table, Error> {
        match self.0.get().get(index) {
            Some(table) => Ok(&**table),
            None => Err(Error::MissingTable(index)),
        }
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &Table {
        get_unchecked(self.0.get_weak(), index)
    }

    #[inline]
    pub fn get_shared(&mut self, index: usize) -> Result<Arc<Table>, Error> {
        match self.0.get().get(index) {
            Some(table) => Ok(table.clone()),
            None => Err(Error::MissingTable(index)),
        }
    }

    #[inline]
    pub unsafe fn get_shared_unchecked(&self, index: usize) -> Arc<Table> {
        get_unchecked(self.0.get_weak(), index).clone()
    }

    /// `metas` must be sorted by `meta.identifier()` and must be deduplicated.
    #[inline]
    pub(crate) fn find_or_add(&mut self, metas: &[&'static Meta]) -> Arc<Table> {
        // Verifies that `metas` is sorted and deduplicated.
        debug_assert!(metas
            .windows(2)
            .all(|metas| metas[0].identifier() < metas[1].identifier()));

        let mut index = 0;
        loop {
            let tables = self.0.get();
            for table in &tables[index..] {
                if table.is_all(metas.iter().map(|meta| meta.identifier())) {
                    return table.clone();
                }
            }
            index = tables.len();

            let columns = metas.iter().map(|&meta| Column::new(meta, 0)).collect();
            let table = Arc::new(Table {
                index: index as _,
                count: 0.into(),
                keys: RwLock::new(Vec::new()),
                columns,
            });
            if let Some(_) = self.0.push_if_same(table.clone()) {
                break table;
            }
        }
    }
}

impl State {
    pub fn new() -> Self {
        Self {
            tables: Slice::new(&[]),
        }
    }
}

impl Table {
    #[inline]
    pub const fn index(&self) -> u32 {
        self.index
    }

    #[inline]
    pub fn count(&self) -> usize {
        self.count.load(Ordering::Acquire)
    }

    #[inline]
    pub fn metas(&self) -> impl FullIterator<Item = &'static Meta> + '_ {
        self.columns().iter().map(|column| column.meta())
    }

    #[inline]
    pub fn types(&self) -> impl FullIterator<Item = TypeId> + '_ {
        self.metas().map(Meta::identifier)
    }

    #[inline]
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    #[inline]
    pub fn has<D: Datum>(&self) -> bool {
        self.has_with(TypeId::of::<D>())
    }

    #[inline]
    pub fn has_with(&self, identifier: TypeId) -> bool {
        self.columns()
            .binary_search_by_key(&identifier, |column| column.meta().identifier())
            .is_ok()
    }

    pub fn column<D: Datum>(&self) -> Result<(usize, &Column), Error> {
        self.column_with(TypeId::of::<D>())
    }

    pub fn column_with(&self, identifier: TypeId) -> Result<(usize, &Column), Error> {
        let index = self
            .columns
            .binary_search_by_key(&identifier, |column| column.meta().identifier())
            .map_err(|_| Error::MissingColumn(identifier))?;
        Ok((index, &self.columns[index]))
    }

    pub fn shrink(&mut self) {
        let keys = self.keys.get_mut();
        let Some(old_capacity) = NonZeroUsize::new(keys.len()) else {
            return;
        };
        let new_capacity = *self.count.get_mut() as usize;
        if new_capacity == old_capacity.get() {
            return;
        }

        debug_assert!(new_capacity <= old_capacity.get());
        keys.truncate(new_capacity as _);
        keys.shrink_to_fit();

        let new_capacity = keys.len();
        for column in self.columns.iter_mut() {
            unsafe { column.shrink(old_capacity, new_capacity) };
        }
    }

    pub(crate) fn reserve<'a>(
        &self,
        keys: RwLockUpgradableReadGuard<'a, Vec<Key>>,
        count: NonZeroUsize,
    ) -> (usize, RwLockUpgradableReadGuard<'a, Vec<Key>>) {
        let start = self.count.load(Ordering::Acquire);
        assert!(
            start < u32::MAX as usize - count.get(),
            "There can not be more than `u32::MAX` keys at a given time."
        );

        let old_capacity = keys.len();
        let new_capacity = count.saturating_add(start as _);
        if new_capacity.get() <= old_capacity {
            return (start, keys);
        }

        let mut keys = RwLockUpgradableReadGuard::upgrade(keys);
        keys.resize(new_capacity.get(), Key::NULL);
        let keys = RwLockWriteGuard::downgrade_to_upgradable(keys);
        for column in self.columns.iter() {
            let guard = column.data().write();
            unsafe { column.grow(old_capacity, new_capacity) };
            drop(guard);
        }
        (start, keys)
    }

    /// `types` must be ordered and deduplicated.
    #[inline]
    pub(crate) fn is_all(&self, types: impl ExactSizeIterator<Item = TypeId>) -> bool {
        self.columns().len() == types.len() && self.types().eq(types)
    }

    /// `types` must be ordered and deduplicated.
    #[inline]
    pub(crate) fn has_all(&self, types: impl ExactSizeIterator<Item = TypeId>) -> bool {
        self.columns().len() >= types.len() && sorted_contains(self.types(), types)
    }
}

unsafe impl Send for Table {}
unsafe impl Sync for Table {}

impl Drop for Table {
    fn drop(&mut self) {
        let count = *self.count.get_mut() as usize;
        let capacity = self.keys.get_mut().len();
        debug_assert!(count <= capacity);
        for column in self.columns.iter_mut() {
            unsafe { column.free(count, capacity) };
        }
    }
}
