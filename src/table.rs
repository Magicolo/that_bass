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
    alloc::{alloc, dealloc, Layout, LayoutError},
    any::TypeId,
    num::NonZeroUsize,
    ops::{Deref, DerefMut},
    ptr::{copy_nonoverlapping, NonNull},
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

pub struct Table {
    index: u32,
    pub(crate) count: AtomicUsize,
    pub(crate) keys: RwLock<Keys>,
    /// Columns are ordered consistently between tables.
    pub(crate) columns: Box<[Column]>,
}

pub struct Column {
    meta: &'static Meta,
    data: RwLock<NonNull<u8>>,
}

pub(crate) struct State {
    tables: Slice<Arc<Table>>,
}

pub(crate) struct Keys {
    data: NonNull<Key>,
    capacity: usize,
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
    pub(crate) fn new(meta: &'static Meta) -> Self {
        Self {
            data: RwLock::new(NonNull::dangling()),
            meta,
        }
    }

    #[inline]
    pub const fn meta(&self) -> &'static Meta {
        self.meta
    }

    #[inline]
    pub(crate) const fn data(&self) -> &RwLock<NonNull<u8>> {
        &self.data
    }

    #[inline]
    pub(crate) unsafe fn copy_to(
        source: (&Self, usize),
        target: (&Self, usize),
        count: NonZeroUsize,
    ) -> bool {
        debug_assert_eq!(source.0.meta().identifier(), target.0.meta().identifier());
        let &Meta { copy: Some(copy), .. } = source.0.meta() else { return false; };
        copy(
            (*source.0.data.data_ptr(), source.1),
            (*target.0.data.data_ptr(), target.1),
            count,
        );
        true
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
        if let Some(drop) = drop {
            drop(data, target_index, count);
        }
        if let Some(copy) = copy {
            copy((data, source_index), (data, target_index), count);
        }
    }

    /// SAFETY: Both the 'source' and 'target' indices must be within the bounds of the column.
    /// The ranges 'source_index..source_index + count' and 'target_index..target_index + count' must not overlap.
    #[inline]
    pub(crate) unsafe fn copy(
        &self,
        source_index: usize,
        target_index: usize,
        count: NonZeroUsize,
    ) -> bool {
        let &Meta { copy: Some(copy), .. } = self.meta() else { return false; };
        let data = *self.data.data_ptr();
        copy((data, source_index), (data, target_index), count);
        true
    }

    #[inline]
    pub(crate) unsafe fn drop(&self, index: usize, count: NonZeroUsize) -> bool {
        let &Meta { drop: Some(drop), .. } = self.meta() else { return false; };
        let data = unsafe { *self.data.data_ptr() };
        drop(data, index, count);
        true
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

            let columns = metas.iter().map(|&meta| Column::new(meta)).collect();
            let table = Arc::new(Table {
                index: index as _,
                count: 0.into(),
                keys: RwLock::new(Keys::default()),
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
    pub const fn columns(&self) -> &[Column] {
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

    pub(crate) fn reserve<'a>(
        &self,
        keys: RwLockUpgradableReadGuard<'a, Keys>,
        count: NonZeroUsize,
    ) -> (usize, RwLockUpgradableReadGuard<'a, Keys>) {
        fn next(
            columns: &[Column],
            capacities: (usize, usize),
            layouts: (Layout, Layout),
        ) -> Result<(*mut u8, Layout), LayoutError> {
            match columns.split_first() {
                Some((column, columns)) if column.meta().size() == 0 => {
                    next(columns, capacities, layouts)
                }
                Some((column, columns)) => {
                    let Meta { layout, .. } = column.meta();
                    let old: (Layout, usize) = layouts.0.extend(layout(capacities.0)?)?;
                    let new: (Layout, usize) = layouts.1.extend(layout(capacities.1)?)?;
                    let (target, layout) = next(columns, capacities, (old.0, new.0))?;
                    let mut guard = column.data().write();
                    unsafe {
                        copy_nonoverlapping(
                            guard.as_ptr(),
                            target.add(new.1),
                            old.0.size() - old.1,
                        );
                        *guard = NonNull::new_unchecked(target.add(new.1));
                    }
                    drop(guard);
                    Ok((target, layout))
                }
                None => Ok((unsafe { alloc(layouts.1) }, layouts.0)),
            }
        }

        let start = self.count.load(Ordering::Acquire);
        assert!(
            start < u32::MAX as usize - count.get(),
            "There can not be more than `u32::MAX` keys at a given time."
        );

        let end = count.saturating_add(start as _);
        let old_capacity = keys.capacity;
        let new_capacity = end.get().next_power_of_two();
        if new_capacity <= old_capacity {
            return (start, keys);
        }

        let old_layout = Layout::array::<Key>(old_capacity).unwrap();
        let new_layout = Layout::array::<Key>(new_capacity).unwrap();
        let (target, layout) = next(
            &self.columns,
            (old_capacity, new_capacity),
            (old_layout, new_layout),
        )
        .unwrap();
        let source = keys.data.cast().as_ptr();
        // Copy can happen outside of write lock since `Key` is immutable.
        unsafe { copy_nonoverlapping(source, target, old_layout.size()) };
        let mut keys = RwLockUpgradableReadGuard::upgrade(keys);
        keys.data = unsafe { NonNull::new_unchecked(target).cast() };
        keys.capacity = new_capacity;
        let keys = RwLockWriteGuard::downgrade_to_upgradable(keys);
        if old_capacity > 0 {
            unsafe { dealloc(source, layout) };
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
        let count = *self.count.get_mut();
        let Keys { data, capacity } = *self.keys.get_mut();
        debug_assert!(count <= capacity);
        let Some(capacity) = NonZeroUsize::new(capacity) else { return; };
        debug_assert_ne!(data, NonNull::dangling());

        let mut total = Layout::array::<Key>(capacity.get()).unwrap();
        for column in self.columns.iter_mut() {
            let Meta { layout, .. } = column.meta();
            if let Some(count) = NonZeroUsize::new(count) {
                unsafe { column.drop(0, count) };
            }
            (total, _) = total.extend(layout(capacity.get()).unwrap()).unwrap();
        }
        unsafe { dealloc(data.cast().as_ptr(), total) };
    }
}

impl Deref for Keys {
    type Target = [Key];

    fn deref(&self) -> &Self::Target {
        unsafe { from_raw_parts(self.data.as_ptr(), self.capacity) }
    }
}

impl DerefMut for Keys {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { from_raw_parts_mut(self.data.as_ptr(), self.capacity) }
    }
}

impl Default for Keys {
    fn default() -> Self {
        Self {
            data: NonNull::dangling(),
            capacity: 0,
        }
    }
}

#[test]
fn table_reserve_grows_when_empty() {
    let table = Table {
        index: 0,
        count: 0.into(),
        keys: Keys::default().into(),
        columns: [].into(),
    };
    let keys = table.keys.upgradable_read();
    assert_eq!(keys.data, NonNull::dangling());
    assert_eq!(keys.len(), 0);
    let (index, keys) = table.reserve(keys, NonZeroUsize::MIN);
    assert_eq!(index, 0);
    assert_eq!(keys.len(), 1);
    assert_ne!(keys.data, NonNull::dangling());
    assert_eq!(table.count(), 0);
}

#[test]
fn table_reserve_grows_in_powers_of_2() {
    let table = Table {
        index: 0,
        count: 0.into(),
        keys: Keys::default().into(),
        columns: [].into(),
    };
    let keys = table.keys.upgradable_read();
    let (_, keys) = table.reserve(keys, NonZeroUsize::MIN.saturating_add(2));
    assert_eq!(keys.len(), 4);
    let (_, keys) = table.reserve(keys, NonZeroUsize::MIN.saturating_add(4));
    assert_eq!(keys.len(), 8);
}
