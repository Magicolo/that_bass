use parking_lot::{RwLockReadGuard, RwLockUpgradableReadGuard, RwLockWriteGuard};

use crate::{
    key::{Key, Keys},
    resources::Resources,
    table::{self, Store, Table, TableRead, TableWrite, Tables},
};
use std::{ops::Range, sync::atomic::Ordering::*};

pub struct Database {
    keys: Keys,
    tables: Tables,
    resources: Resources,
}

impl Database {
    pub fn new() -> Self {
        Self {
            keys: Keys::new(),
            tables: Tables::new(),
            resources: Resources::new(),
        }
    }

    #[inline]
    pub const fn keys(&self) -> &Keys {
        &self.keys
    }

    #[inline]
    pub const fn tables(&self) -> &Tables {
        &self.tables
    }

    #[inline]
    pub const fn resources(&self) -> &Resources {
        &self.resources
    }

    #[inline]
    pub(crate) const fn decompose_pending(pending: u64) -> (u32, u32) {
        ((pending >> 32) as u32, pending as u32)
    }

    #[inline]
    pub(crate) const fn recompose_pending(begun: u32, ended: u32) -> u64 {
        ((begun as u64) << 32) | (ended as u64)
    }

    pub(crate) fn add_to_table(
        &self,
        keys: &[Key],
        table: &Table,
        initialize: impl FnOnce(Range<usize>, &[Store]),
    ) {
        let inner = table.inner.upgradable_read();
        let (index, inner) = Self::add_to_reserve(keys.len() as _, inner);
        self.add_to_resolve(keys, index, table, &inner, initialize);
    }

    pub(crate) fn remove_from_table(&self, table: &Table, rows: &[u32], range: Range<u32>) {
        let mut inner = table.inner.write();
        self.remove_from_resolve(table, &mut inner, rows, range);
    }

    /// Can be used to add or remove data associated with a key.
    pub(crate) fn move_to_table(
        &self,
        key: Key,
        target_index: u32,
        mut initialize: impl FnMut(usize, &Store),
    ) -> Option<()> {
        let target_table = self.tables.get(target_index as usize)?;
        let (slot, indices, mut source_write, target_upgrade) = loop {
            let slot = self.keys.get(key).ok()?;
            let indices = slot.indices();
            let source_table = self.tables.get(indices.0 as usize)?;

            // If locks are always taken in order (lower index first), there can not be a deadlock between move operations.
            // TODO: Defer the operation if a local deadlock is detected...
            let (source_write, target_upgrade) = if source_table.index() < target_table.index() {
                let left = source_table.inner.write();
                let right = target_table.inner.upgradable_read();
                (left, right)
            } else if source_table.index() > target_table.index() {
                let right = target_table.inner.upgradable_read();
                let left = source_table.inner.write();
                (left, right)
            } else {
                // No move is needed.
                return Some(());
            };

            // Check the indices while holding the source table lock to ensure that the key wasn't moved.
            if indices == slot.indices() {
                break (slot, indices, source_write, target_upgrade);
            }
        };

        let last_index = Self::remove_from_reserve(&mut source_write, 1);
        let (index, target_read) = Self::add_to_reserve(1, target_upgrade);
        let start = index as usize;

        fn drop_or_squash(source: u32, target: u32, store: &mut Store) {
            if source == target {
                unsafe { store.drop(target as _, 1) };
            } else {
                unsafe { store.squash(source as _, target as _, 1) };
            }
        }

        let mut store_indices = (0, 0);
        loop {
            match (
                source_write.stores.get_mut(store_indices.0),
                target_read.stores.get(store_indices.1),
            ) {
                (Some(source_store), Some(target_store)) => {
                    let source_identifier = source_store.meta().identifier();
                    let target_identifier = target_store.meta().identifier();
                    if source_identifier == target_identifier {
                        store_indices.0 += 1;
                        store_indices.1 += 1;
                        unsafe {
                            Store::copy((source_store, indices.1 as _), (target_store, start), 1);
                        };
                        drop_or_squash(last_index, indices.1, source_store);
                    } else if source_identifier < target_identifier {
                        store_indices.0 += 1;
                        drop_or_squash(last_index, indices.1, source_store);
                    } else {
                        store_indices.1 += 1;
                        initialize(start, target_store);
                    }
                }
                (Some(source_store), None) => {
                    store_indices.0 += 1;
                    drop_or_squash(last_index, indices.1, source_store);
                }
                (None, Some(target_store)) => {
                    store_indices.1 += 1;
                    initialize(start, target_store);
                }
                (None, None) => break,
            }
        }

        if last_index == indices.1 {
            unsafe {
                let keys = &mut *target_read.keys.get();
                *keys.get_unchecked_mut(start) = key;
                slot.update(target_index, start as _);
            }
        } else {
            let source_keys = source_write.keys.get_mut();
            unsafe {
                let target_keys = &mut *target_read.keys.get();
                let last_key = *source_keys.get_unchecked(last_index as usize);
                *source_keys.get_unchecked_mut(indices.1 as usize) = last_key;
                *target_keys.get_unchecked_mut(start) = key;
                slot.update(target_index, start as _);
                self.keys
                    .get_unchecked(last_key)
                    .update(indices.0, indices.1);
            }
        }

        Self::add_to_commit(1, &target_read);
        drop(source_write);
        Some(())
    }

    pub(crate) fn table_read<'a>(&'a self, table: &'a Table) -> TableRead<'a> {
        TableRead::new(self, table, table.inner.read())
    }

    pub(crate) fn table_read_with<'a>(
        &'a self,
        table: &'a Table,
        valid: impl FnOnce(&table::Inner) -> bool,
    ) -> Option<TableRead<'a>> {
        let read = table.inner.read();
        if valid(&read) {
            Some(TableRead::new(self, table, read))
        } else {
            None
        }
    }

    pub(crate) fn table_try_read<'a>(&'a self, table: &'a Table) -> Option<TableRead<'a>> {
        Some(TableRead::new(self, table, table.inner.try_read()?))
    }

    pub(crate) fn table_write<'a>(&'a self, table: &'a Table) -> TableWrite<'a> {
        TableWrite::new(self, table, table.inner.write())
    }

    pub(crate) fn table_write_with<'a>(
        &'a self,
        table: &'a Table,
        valid: impl FnOnce(&mut table::Inner) -> bool,
    ) -> Option<TableWrite<'a>> {
        let mut write = table.inner.write();
        if valid(&mut write) {
            Some(TableWrite::new(self, table, write))
        } else {
            None
        }
    }

    fn add_to_reserve<'d>(
        reserve: u32,
        inner: RwLockUpgradableReadGuard<'d, table::Inner>,
    ) -> (u32, RwLockReadGuard<'d, table::Inner>) {
        let (index, _) = {
            let add = Self::recompose_pending(reserve, 0);
            let pending = inner.pending.fetch_add(add, AcqRel);
            Self::decompose_pending(pending)
        };
        // There can not be more than `u32::MAX` keys at a given time.
        assert!(index < u32::MAX - reserve);

        let capacity = index as usize + reserve as usize;
        if capacity <= inner.capacity() {
            (index, RwLockUpgradableReadGuard::downgrade(inner))
        } else {
            let mut inner = RwLockUpgradableReadGuard::upgrade(inner);
            inner.grow(capacity);
            (index, RwLockWriteGuard::downgrade(inner))
        }
    }

    fn add_to_resolve(
        &self,
        keys: &[Key],
        index: u32,
        table: &Table,
        inner: &table::Inner,
        initialize: impl FnOnce(Range<usize>, &[Store]),
    ) {
        let start = index as usize;
        let end = start + keys.len();
        unsafe { (&mut **inner.keys.get()).get_unchecked_mut(start..end) }.copy_from_slice(keys);
        initialize(start..end, inner.stores());
        for (i, &key) in keys.iter().enumerate() {
            let slot = unsafe { self.keys.get_unchecked(key) };
            slot.initialize(key.generation(), table.index(), index + i as u32);
        }
        Self::add_to_commit(keys.len() as _, inner);
    }

    fn add_to_commit(reserve: u32, inner: &table::Inner) {
        let add = Self::recompose_pending(0, reserve);
        let pending = inner.pending.fetch_add(add, AcqRel);
        let (begun, ended) = Self::decompose_pending(pending);
        debug_assert!(begun >= ended);
        if begun == ended + reserve {
            inner.count.fetch_max(begun, Relaxed);
        }
    }

    fn remove_from_reserve(inner: &mut table::Inner, count: u32) -> u32 {
        let table_count = inner.count.get_mut();
        let table_pending = inner.pending.get_mut();
        let (begun, ended) = Self::decompose_pending(*table_pending);

        // Sanity checks. If this is not the case, there is a bug in the locking logic.
        debug_assert_eq!(begun, ended);
        debug_assert!(begun > 0);
        debug_assert_eq!(begun, *table_count);
        *table_count -= count;
        *table_pending = Self::recompose_pending(begun - 1, ended - 1);
        *table_count
    }

    fn remove_from_resolve(
        &self,
        table: &Table,
        inner: &mut table::Inner,
        rows: &[u32],
        range: Range<u32>,
    ) {
        if rows.len() == 0 {
            return;
        }

        let head = Self::remove_from_reserve(inner, rows.len() as u32) as usize;
        let keys = inner.keys.get_mut();
        let (low, high) = (range.start as usize, range.end as usize);
        if range.len() == rows.len() {
            // The destroy range is contiguous.
            let over = high.saturating_sub(head);
            let end = rows.len() - over;
            if end > 0 {
                // Squash the range at the end of the table on the begining of the removed range.
                let start = head + over;
                for store in inner.stores.iter_mut() {
                    unsafe { store.squash(start, low, end) };
                }

                // Update the keys.
                keys.copy_within(start..start + end, low);
                for i in low..low + end {
                    unsafe { self.keys().get_unchecked(*keys.get_unchecked(i)) }
                        .update(table.index(), i as _);
                }
            }

            if over > 0 {
                for store in inner.stores.iter_mut() {
                    unsafe { store.drop(head, over) };
                }
            }
        } else {
            let mut index = 0;
            let mut last = keys.len();

            // Try to consume the rows in the range `end..keys.len()` since they are guaranteed to be valid.
            let end = high.max(head);
            while last > end {
                match rows.get(index) {
                    Some(&row) => {
                        let row = row as usize;
                        let mut previous = row;
                        let start = index;
                        index += 1;

                        if row < head {
                            last -= 1;

                            // Try to batch contiguous squashes.
                            while last > end {
                                if let Some(&row) = rows.get(index) {
                                    let row = row as usize;
                                    if previous + 1 == row && row < head {
                                        previous = row;
                                        last -= 1;
                                        index += 1;
                                    } else {
                                        break;
                                    }
                                }
                            }

                            let count = index - start;
                            for store in inner.stores.iter_mut() {
                                unsafe { store.squash(last, row, count) };
                            }

                            keys.copy_within(last..last + count, row);
                            for row in row..row + count {
                                unsafe { self.keys().get_unchecked(*keys.get_unchecked(row)) }
                                    .update(table.index(), row as _);
                            }
                        } else {
                            // Try to batch contiguous drops.
                            while let Some(&row) = rows.get(index) {
                                let row = row as usize;
                                if previous + 1 == row {
                                    previous = row;
                                    index += 1;
                                } else {
                                    break;
                                }
                            }

                            let count = index - start;
                            for store in inner.stores.iter_mut() {
                                unsafe { store.drop(row, count) };
                            }
                        }
                    }
                    None => return,
                }
            }

            // Tag keys that are going to be removed such removed keys and valid keys can be differentiated.
            for &row in &rows[index..] {
                *unsafe { keys.get_unchecked_mut(row as usize) } = Key::NULL;
            }

            // Remove that remaining rows the slow way.
            while let Some(&row) = rows.get(index) {
                let row = row as usize;
                let mut previous = row;
                let start = index;
                index += 1;

                if row < head {
                    // Find the next valid row to move.
                    while keys.get(last).copied() == Some(Key::NULL) {
                        last -= 1;
                    }
                    debug_assert!(last >= head);

                    for store in inner.stores.iter_mut() {
                        unsafe { store.squash(last, row, 1) };
                    }

                    unsafe {
                        let key = *keys.get_unchecked_mut(last);
                        *keys.get_unchecked_mut(row) = key;
                        self.keys()
                            .get_unchecked(key)
                            .update(table.index(), row as _);
                    }

                    last -= 1;
                } else {
                    // Try to batch contiguous drops.
                    while let Some(&row) = rows.get(index) {
                        let row = row as usize;
                        if previous + 1 == row {
                            previous = row;
                            index += 1;
                        } else {
                            break;
                        }
                    }

                    let count = index - start;
                    for store in inner.stores.iter_mut() {
                        unsafe { store.drop(row, count) };
                    }
                }
            }
        }
    }
}
