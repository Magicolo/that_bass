use crate::{
    key::{Key, Keys},
    resources::Resources,
    table::{self, Defer, Store, Table, Tables},
    utility::FullIterator,
    Error,
};
use parking_lot::{RwLockReadGuard, RwLockUpgradableReadGuard, RwLockWriteGuard};
use std::{
    mem::{replace, ManuallyDrop},
    sync::atomic::Ordering::*,
};

pub struct Database {
    pub(crate) keys: Keys,
    pub(crate) tables: Tables,
    pub(crate) resources: Resources,
}

pub struct TableRead<'a>(
    &'a Database,
    &'a Table,
    Option<RwLockReadGuard<'a, table::Inner>>,
);

pub struct TableUpgrade<'a>(
    &'a Database,
    &'a Table,
    Option<RwLockUpgradableReadGuard<'a, table::Inner>>,
);

pub struct TableWrite<'a>(
    &'a Database,
    &'a Table,
    Option<RwLockWriteGuard<'a, table::Inner>>,
);

impl Database {
    pub fn new() -> Self {
        Self {
            keys: Keys::new(),
            tables: Tables::new(),
            resources: Resources::new(),
        }
    }

    #[inline]
    pub fn tables(&self) -> impl FullIterator<Item = &Table> {
        self.tables.into_iter()
    }

    pub(crate) fn add_to_table<S, D: 'static>(
        &self,
        keys: &mut [Key],
        table: &Table,
        mut state: S,
        mut initialize: impl FnMut(&mut S, (usize, usize), &[Store]),
        mut state_defer: impl FnMut(&mut S, usize) -> D,
        mut initialize_defer: impl FnMut(&mut D, usize, &[Store]) + Copy + 'static,
    ) {
        if keys.len() == 0 {
            return;
        }

        // All keys can be reserved at once.
        self.keys.reserve(keys);

        // Hold this lock until the operation is fully complete such that no move operation are interleaved.
        let table_upgrade = self.table_upgrade(table);
        match Self::add_to_reserve(keys.len() as _, table_upgrade) {
            Ok((row_index, row_count, table_read)) => {
                self.add_to_resolve(
                    keys,
                    row_index,
                    row_count,
                    table_read.table(),
                    table_read.inner(),
                    |row, stores| initialize(&mut state, row, stores),
                );
                drop(table_read);
            }
            Err((row_index, row_count)) => {
                let mut state = state_defer(&mut state, keys.len());
                table.defer(Defer::Add {
                    keys: keys.iter().copied().collect(),
                    row_index,
                    row_count,
                    initialize: Box::new(move |row_index, stores| {
                        initialize_defer(&mut state, row_index, stores)
                    }),
                });
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
                table.defer(Defer::Remove { row_index });
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
        loop {
            let slot = self.keys.get(key).ok()?;
            let source_indices = slot.indices();
            if source_indices.0 == target_index {
                // No move is needed.
                break Some(());
            }
            let source_table = self.tables.get(source_indices.0 as usize)?;

            // Note that 2 very synchronized threads with their `source_table` and `target_table` swapped may
            // defeat this scheme for taking 2 write locks without dead locking. It is assumed that it doesn't
            // really happen in practice.
            let source_write = self.table_write(source_table);
            let (mut source_write, target_upgrade) = match self.table_try_upgrade(target_table) {
                Some(target_read) => (source_write, target_read),
                None => {
                    drop(source_write);
                    let target_read = self.table_upgrade(target_table);
                    match self.table_try_write(source_table) {
                        Some(source_write) => (source_write, target_read),
                        None => continue,
                    }
                }
            };
            if source_indices != slot.indices() {
                continue;
            }

            let last_index = Self::remove_from_reserve(&mut source_write);
            // TODO: Defer
            let (row_index, row_count, target_read) = match Self::add_to_reserve(1, target_upgrade)
            {
                Ok((row_index, row_count, target_read)) => (row_index, row_count, target_read),
                Err((row_index, row_count)) => return None,
            };

            let mut store_indices = (0, 0);

            fn drop_or_squash(source: u32, target: u32, store: &mut Store) {
                if source == target {
                    unsafe { store.drop(target as _, 1) };
                } else {
                    unsafe { store.squash(source as _, target as _, 1) };
                }
            }

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
                                    (source_store, source_indices.1 as _),
                                    (target_store, row_index as _),
                                    1,
                                );
                            };
                            drop_or_squash(last_index, source_indices.1, source_store);
                        } else if source_identifier < target_identifier {
                            store_indices.0 += 1;
                            drop_or_squash(last_index, source_indices.1, source_store);
                        } else {
                            store_indices.1 += 1;
                            initialize(row_index as _, row_count, target_store);
                        }
                    }
                    (Some(source_store), None) => {
                        store_indices.0 += 1;
                        drop_or_squash(last_index, source_indices.1, source_store);
                    }
                    (None, Some(target_store)) => {
                        store_indices.1 += 1;
                        initialize(row_index as _, row_count, target_store);
                    }
                    (None, None) => break,
                }
            }

            if last_index == source_indices.1 {
                unsafe {
                    let keys = &mut *target_read.inner().keys.get();
                    *keys.get_unchecked_mut(row_index as usize) = key;
                    self.keys.get_unchecked(key).update(target_index, row_index);
                }
            } else {
                let source_keys = source_write.inner_mut().keys.get_mut();
                unsafe {
                    let last_key = *source_keys.get_unchecked(last_index as usize);
                    let source_key = source_keys.get_unchecked_mut(source_indices.1 as usize);
                    let source_key = replace(source_key, last_key);

                    let target_keys = &mut *target_read.inner().keys.get();
                    *target_keys.get_unchecked_mut(row_index as usize) = source_key;
                    self.keys
                        .get_unchecked(source_key)
                        .update(target_index, row_index);
                    self.keys
                        .get_unchecked(last_key)
                        .update(source_indices.0, source_indices.1);
                }
            }

            Self::add_to_commit(1, target_read.inner());
            drop(source_write);
            break Some(());
        }
    }

    pub(crate) fn table_read<'a>(&'a self, table: &'a Table) -> TableRead<'a> {
        TableRead::new(self, table, table.inner.read())
    }

    pub(crate) fn table_try_read<'a>(&'a self, table: &'a Table) -> Option<TableRead<'a>> {
        Some(TableRead::new(self, table, table.inner.try_read()?))
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

    pub(crate) fn table_try_write<'a>(&'a self, table: &'a Table) -> Option<TableWrite<'a>> {
        Some(TableWrite::new(self, table, table.inner.try_write()?))
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
        let table_read = if row_count > table_upgrade.inner().capacity() {
            match table_upgrade.try_upgrade() {
                Ok(mut table_write) => {
                    table_write.inner_mut().grow(row_count);
                    table_write.downgrade()
                }
                Err(table_upgrade) => {
                    // Do not run `TableUpgrade::drop` since it just failed to upgrade its lock.
                    table_upgrade.forget();
                    return Err((row_index, row_count));
                }
            }
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
        mut initialize: impl FnMut((usize, usize), &[Store]),
    ) {
        unsafe {
            let table_keys = &mut *inner.keys.get();
            table_keys.get_unchecked_mut(row_index as usize..row_count as usize)
        }
        .copy_from_slice(keys);
        initialize((row_index as _, keys.len()), inner.stores());

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
        // TODO: If `try_write` fails, defer the destroy.
        // - Somehow, queries must be prevented from observing this key in the table...
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

impl<'a> TableRead<'a> {
    #[inline]
    pub(crate) fn new(
        database: &'a Database,
        table: &'a Table,
        guard: RwLockReadGuard<'a, table::Inner>,
    ) -> Self {
        Self(database, table, Some(guard))
    }

    #[inline]
    pub const fn database(&self) -> &'a Database {
        self.0
    }

    #[inline]
    pub const fn table(&self) -> &'a Table {
        self.1
    }

    #[inline]
    pub fn keys(&self) -> &[Key] {
        self.inner().keys()
    }

    #[inline]
    pub fn stores(&self) -> &[Store] {
        self.inner().stores()
    }

    #[inline]
    pub(crate) fn inner(&self) -> &table::Inner {
        unsafe { self.2.as_deref().unwrap_unchecked() }
    }

    #[inline]
    pub(crate) fn forget(self) {
        let mut guard = ManuallyDrop::new(self);
        drop(guard.2.take());
    }
}

impl<'a> TableUpgrade<'a> {
    #[inline]
    pub(crate) fn new(
        database: &'a Database,
        table: &'a Table,
        guard: RwLockUpgradableReadGuard<'a, table::Inner>,
    ) -> Self {
        Self(database, table, Some(guard))
    }

    #[inline]
    pub const fn database(&self) -> &'a Database {
        self.0
    }

    #[inline]
    pub const fn table(&self) -> &'a Table {
        self.1
    }

    #[inline]
    pub fn keys(&self) -> &[Key] {
        self.inner().keys()
    }

    #[inline]
    pub fn stores(&self) -> &[Store] {
        self.inner().stores()
    }

    #[inline]
    pub(crate) fn inner(&self) -> &table::Inner {
        unsafe { self.2.as_deref().unwrap_unchecked() }
    }

    #[inline]
    pub fn upgrade(self) -> TableWrite<'a> {
        let mut guard = ManuallyDrop::new(self);
        TableWrite::new(
            guard.0,
            guard.1,
            RwLockUpgradableReadGuard::upgrade(unsafe { guard.2.take().unwrap_unchecked() }),
        )
    }

    #[inline]
    pub fn try_upgrade(self) -> Result<TableWrite<'a>, Self> {
        let mut guard = ManuallyDrop::new(self);
        match RwLockUpgradableReadGuard::try_upgrade(unsafe { guard.2.take().unwrap_unchecked() }) {
            Ok(write) => Ok(TableWrite::new(guard.0, guard.1, write)),
            Err(upgrade) => Err(Self::new(guard.0, guard.1, upgrade)),
        }
    }

    #[inline]
    pub fn downgrade(self) -> TableRead<'a> {
        let mut guard = ManuallyDrop::new(self);
        TableRead::new(
            guard.0,
            guard.1,
            RwLockUpgradableReadGuard::downgrade(unsafe { guard.2.take().unwrap_unchecked() }),
        )
    }

    #[inline]
    pub(crate) fn forget(self) {
        let mut guard = ManuallyDrop::new(self);
        drop(guard.2.take());
    }
}

impl<'a> TableWrite<'a> {
    #[inline]
    pub(crate) fn new(
        database: &'a Database,
        table: &'a Table,
        guard: RwLockWriteGuard<'a, table::Inner>,
    ) -> Self {
        let mut guard = Self(database, table, Some(guard));
        guard.resolve();
        guard
    }

    #[inline]
    pub const fn database(&self) -> &'a Database {
        self.0
    }

    #[inline]
    pub const fn table(&self) -> &'a Table {
        self.1
    }

    #[inline]
    pub fn keys(&self) -> &[Key] {
        self.inner().keys()
    }

    #[inline]
    pub fn stores(&self) -> &[Store] {
        self.inner().stores()
    }

    #[inline]
    pub(crate) fn inner(&self) -> &table::Inner {
        unsafe { self.2.as_deref().unwrap_unchecked() }
    }

    #[inline]
    pub(crate) fn inner_mut(&mut self) -> &mut table::Inner {
        unsafe { self.2.as_deref_mut().unwrap_unchecked() }
    }

    #[inline]
    pub fn downgrade(self) -> TableRead<'a> {
        let mut guard = ManuallyDrop::new(self);
        TableRead::new(
            guard.0,
            guard.1,
            RwLockWriteGuard::downgrade(unsafe { guard.2.take().unwrap_unchecked() }),
        )
    }

    pub(crate) fn resolve(&mut self) {
        while let Some(defer) = self.1.defer.write().pop_front() {
            match defer {
                Defer::Add {
                    keys,
                    row_index,
                    row_count,
                    mut initialize,
                } => {
                    self.inner_mut().grow(row_count);
                    self.database().add_to_resolve(
                        &keys,
                        row_index,
                        row_count,
                        self.table(),
                        self.inner(),
                        |row, stores| initialize(row.0, stores),
                    )
                }
                Defer::Remove { row_index } => self.0.remove_from_resolve(self, row_index),
            }
        }
    }

    #[inline]
    pub(crate) fn forget(self) {
        let mut guard = ManuallyDrop::new(self);
        drop(guard.2.take());
    }
}

impl<'a> Drop for TableRead<'a> {
    #[inline]
    fn drop(&mut self) {
        if match self.table().defer.try_read() {
            Some(defer) if defer.len() > 0 => true,
            None => true,
            _ => false,
        } {
            drop(self.2.take());
            self.0.table_try_write(self.1);
        }
    }
}

impl<'a> Drop for TableUpgrade<'a> {
    #[inline]
    fn drop(&mut self) {
        if match self.table().defer.try_read() {
            Some(defer) if defer.len() > 0 => true,
            None => true,
            _ => false,
        } {
            if let Some(upgrade) = self.2.take() {
                if let Ok(write) = RwLockUpgradableReadGuard::try_upgrade(upgrade) {
                    TableWrite::new(self.0, self.1, write);
                }
            }
        }
    }
}

impl<'a> Drop for TableWrite<'a> {
    #[inline]
    fn drop(&mut self) {
        self.resolve();
    }
}
