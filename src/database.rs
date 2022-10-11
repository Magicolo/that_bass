use crate::{
    key::{Key, Keys},
    resources::Resources,
    table::{self, Add, Defer, Remove, Store, Table, TableRead, TableUpgrade, TableWrite, Tables},
    Error,
};
use std::sync::atomic::Ordering::*;

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

    pub(crate) fn add_to_table<S>(
        &self,
        keys: &mut [Key],
        table: &Table,
        state: S,
        initialize: impl FnOnce(S, u32, &[Store]),
        defer: impl FnOnce(S, &[Key], u32, usize) -> Add,
    ) {
        if keys.len() == 0 {
            return;
        }

        // All keys can be reserved at once.
        self.keys.reserve(keys);

        // Hold this lock until the operation is fully completed such that no move operations are interleaved.
        let table_upgrade = self.table_upgrade(table);
        match Self::add_to_reserve(keys.len() as _, table_upgrade) {
            Ok((row_index, row_count, table_read)) => {
                self.add_to_resolve(
                    keys,
                    row_index,
                    row_count,
                    table_read.table(),
                    table_read.inner(),
                    |stores| initialize(state, row_index, stores),
                );
                drop(table_read);
            }
            Err((row_index, row_count)) => {
                table.defer(Defer::Add(defer(state, keys, row_index, row_count)))
            }
        }
    }

    pub(crate) fn remove_from_table(&self, key: Key) -> Result<bool, Error> {
        let (table_index, row_index) = self.keys.release(key)?;
        let table = unsafe { self.tables.get_unchecked(table_index as usize) };
        match self.table_try_write(table) {
            Some(mut table_write) => {
                self.remove_from_resolve(&mut table_write, row_index);
                drop(table_write);
                Ok(true)
            }
            None => {
                table.defer(Defer::Remove(Remove { row_index }));
                Ok(false)
            }
        }
    }

    /// Can be used to add or remove data associated with a key.
    pub(crate) fn move_to_table(
        &self,
        key: Key,
        target_index: u32,
        mut initialize: impl FnMut(usize, usize, &Store),
    ) -> Option<()> {
        let target_table = self.tables.get(target_index as usize)?;
        let (slot, indices, mut source_write, target_upgrade) = loop {
            let slot = self.keys.get(key).ok()?;
            let indices = slot.indices();
            let source_table = self.tables.get(indices.0 as usize)?;

            // If locks are always taken in order (lower index first), there can not be a deadlock between move operations.
            // TODO: Defer the operation if a local deadlock is detected...
            let (source_write, target_upgrade) = if source_table.index() < target_table.index() {
                let left = self.table_write(source_table);
                let right = self.table_upgrade(target_table);
                (left, right)
            } else if source_table.index() > target_table.index() {
                let right = self.table_upgrade(target_table);
                let left = self.table_write(source_table);
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

        let last_index = Self::remove_from_reserve(&mut source_write);
        // TODO: Defer
        let (row_index, row_count, target_read) = match Self::add_to_reserve(1, target_upgrade) {
            Ok((row_index, row_count, target_read)) => (row_index, row_count, target_read),
            Err((row_index, row_count)) => return None,
        };

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
                source_write.inner_mut().stores.get_mut(store_indices.0),
                target_read.inner().stores.get(store_indices.1),
            ) {
                (Some(source_store), Some(target_store)) => {
                    let source_identifier = source_store.meta().identifier;
                    let target_identifier = target_store.meta().identifier;
                    if source_identifier == target_identifier {
                        store_indices.0 += 1;
                        store_indices.1 += 1;
                        unsafe {
                            Store::copy(
                                (source_store, indices.1 as _),
                                (target_store, row_index as _),
                                1,
                            );
                        };
                        drop_or_squash(last_index, indices.1, source_store);
                    } else if source_identifier < target_identifier {
                        store_indices.0 += 1;
                        drop_or_squash(last_index, indices.1, source_store);
                    } else {
                        store_indices.1 += 1;
                        initialize(row_index as _, row_count, target_store);
                    }
                }
                (Some(source_store), None) => {
                    store_indices.0 += 1;
                    drop_or_squash(last_index, indices.1, source_store);
                }
                (None, Some(target_store)) => {
                    store_indices.1 += 1;
                    initialize(row_index as _, row_count, target_store);
                }
                (None, None) => break,
            }
        }

        if last_index == indices.1 {
            unsafe {
                let keys = &mut *target_read.inner().keys.get();
                *keys.get_unchecked_mut(row_index as usize) = key;
                slot.update(target_index, row_index);
            }
        } else {
            let source_keys = source_write.inner_mut().keys.get_mut();
            unsafe {
                let target_keys = &mut *target_read.inner().keys.get();
                let last_key = *source_keys.get_unchecked(last_index as usize);
                *source_keys.get_unchecked_mut(indices.1 as usize) = last_key;
                *target_keys.get_unchecked_mut(row_index as usize) = key;
                slot.update(target_index, row_index);
                self.keys
                    .get_unchecked(last_key)
                    .update(indices.0, indices.1);
            }
        }

        Self::add_to_commit(1, target_read.inner());
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

    pub(crate) fn table_try_read_with<'a>(
        &'a self,
        table: &'a Table,
        valid: impl FnOnce(&table::Inner) -> bool,
    ) -> Option<TableRead<'a>> {
        let read = table.inner.try_read()?;
        if valid(&read) {
            Some(TableRead::new(self, table, read))
        } else {
            None
        }
    }

    pub(crate) fn table_upgrade<'a>(&'a self, table: &'a Table) -> TableUpgrade<'a> {
        TableUpgrade::new(self, table, table.inner.upgradable_read())
    }

    pub(crate) fn table_try_upgrade<'a>(&'a self, table: &'a Table) -> Option<TableUpgrade<'a>> {
        Some(TableUpgrade::new(
            self,
            table,
            table.inner.try_upgradable_read()?,
        ))
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

    pub(crate) fn table_try_write<'a>(&'a self, table: &'a Table) -> Option<TableWrite<'a>> {
        Some(TableWrite::new(self, table, table.inner.try_write()?))
    }

    pub(crate) fn resolve_table(&self, table_write: &mut TableWrite) {
        while let Some(defer) = table_write.table().defer.write().pop_front() {
            match defer {
                Defer::Add(Add {
                    keys,
                    row_index,
                    row_count,
                    initialize,
                }) => {
                    table_write.inner_mut().grow(row_count);
                    self.add_to_resolve(
                        &keys,
                        row_index,
                        row_count,
                        table_write.table(),
                        table_write.inner(),
                        initialize,
                    )
                }
                Defer::Remove(Remove { row_index }) => {
                    self.remove_from_resolve(table_write, row_index)
                }
            }
        }
    }

    #[inline]
    const fn decompose_pending(pending: u64) -> (u32, u32) {
        ((pending >> 32) as u32, pending as u32)
    }

    #[inline]
    const fn recompose_pending(begun: u32, ended: u32) -> u64 {
        ((begun as u64) << 32) | (ended as u64)
    }

    fn add_to_reserve(
        reserve: u32,
        table_upgrade: TableUpgrade,
    ) -> Result<(u32, usize, TableRead), (u32, usize)> {
        let (row_index, _) = {
            let add = Self::recompose_pending(reserve, 0);
            let pending = table_upgrade.inner().pending.fetch_add(add, AcqRel);
            Self::decompose_pending(pending)
        };
        let row_count = row_index as usize + reserve as usize;

        // There can not be more than `u32::MAX` keys at a given time.
        assert!(row_count < u32::MAX as _);
        let table_read = if row_count > table_upgrade.capacity() {
            // TODO: Prevent local deadlocks.
            let mut table_write = table_upgrade.upgrade();
            table_write.inner_mut().grow(row_count);
            table_write.downgrade()
            // match table_upgrade.try_upgrade() {
            //     Ok(mut table_write) => {
            //         table_write.inner_mut().grow(row_count);
            //         table_write.downgrade()
            //     }
            //     Err(table_upgrade) => {
            //         // Do not run `TableUpgrade::drop` since it just failed to upgrade its lock.
            //         drop(table_upgrade.guard());
            //         return Err((row_index, row_count));
            //     }
            // }
        } else {
            table_upgrade.downgrade()
        };
        Ok((row_index, row_count, table_read))
    }

    fn add_to_resolve(
        &self,
        keys: &[Key],
        row_index: u32,
        row_count: usize,
        table: &Table,
        inner: &table::Inner,
        initialize: impl FnOnce(&[Store]),
    ) {
        unsafe {
            let table_keys = &mut *inner.keys.get();
            table_keys.get_unchecked_mut(row_index as usize..row_count as usize)
        }
        .copy_from_slice(keys);
        initialize(inner.stores());

        // Initialize the slot only after the table row has been fully initialized and while holding the `table_read`
        // lock to ensure that no keys can be observed in an uninitialized state either through the `Keys` or the table.
        for &key in keys.iter() {
            let slot = unsafe { self.keys.get_unchecked(key) };
            slot.initialize(key.generation(), table.index(), row_index);
        }

        Self::add_to_commit(keys.len() as _, inner);
    }

    fn add_to_commit(reserve: u32, inner: &table::Inner) {
        let pending = inner.pending.fetch_add(reserve as u64, AcqRel);
        let (begun, ended) = Self::decompose_pending(pending);
        debug_assert!(begun >= ended);
        if begun == ended + reserve {
            inner.count.fetch_max(begun, Relaxed);
        }
    }

    fn remove_from_reserve(table_write: &mut TableWrite) -> u32 {
        let inner = table_write.inner_mut();
        let table_count = inner.count.get_mut();
        let table_pending = inner.pending.get_mut();
        let (begun, ended) = Self::decompose_pending(*table_pending);

        // Sanity checks. If this is not the case, there is a bug in the locking logic.
        debug_assert_eq!(begun, ended);
        debug_assert!(begun > 0);
        debug_assert_eq!(begun, *table_count);
        *table_count -= 1;
        *table_pending = Self::recompose_pending(begun - 1, ended - 1);
        *table_count
    }

    fn remove_from_resolve(&self, table_write: &mut TableWrite, row_index: u32) {
        let last_index = Self::remove_from_reserve(table_write);
        if row_index == last_index {
            for store in table_write.inner_mut().stores.iter_mut() {
                unsafe { store.drop(row_index as _, 1) };
            }
        } else {
            for store in table_write.inner_mut().stores.iter_mut() {
                unsafe { store.squash(last_index as _, row_index as _, 1) };
            }

            let keys = table_write.inner_mut().keys.get_mut();
            unsafe {
                let last_key = *keys.get_unchecked(last_index as usize);
                *keys.get_unchecked_mut(row_index as usize) = last_key;
                self.keys
                    .get_unchecked(last_key)
                    .update(table_write.table().index(), row_index);
            }
        }
    }
}
