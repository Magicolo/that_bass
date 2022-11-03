use crate::{
    key::{Key, Slot},
    table::{self, Store, Table},
    template::{ApplyContext, DeclareContext, InitializeContext, Template},
    Database, Error, Meta,
};
use parking_lot::RwLockWriteGuard;
use std::{
    collections::HashMap, marker::PhantomData, mem::replace, num::NonZeroUsize, ptr::NonNull,
    sync::Arc,
};

pub struct Add<'d, T: Template> {
    database: &'d Database,
    keys: HashMap<Key, (usize, u32)>,
    pending: Vec<(Key, &'d Slot, T, u32)>,
    sorted: HashMap<u32, State<'d, T>>,
    metas: Arc<Vec<&'static Meta>>,
    pointers: Vec<NonNull<()>>,
}

struct State<'d, T: Template> {
    source: Arc<Table>,
    target: Arc<Table>,
    inner: Arc<Inner<T>>,
    // TODO: Merge `rows` and `templates`? How does this affect `slot.update/initialize`?
    rows: Vec<(Key, &'d Slot, u32)>,
    templates: Vec<T>,
}

struct Inner<T: Template> {
    state: T::State,
    add: Vec<usize>,
    // TODO: A `Vec<usize>` should suffice here where its indices map to its values `source_store -> target_store`.
    copy: Vec<(usize, usize)>,
}

impl Database {
    pub fn add<T: Template>(&self) -> Result<Add<T>, Error> {
        struct Share<T: Template>(Arc<Vec<&'static Meta>>, PhantomData<fn(T)>);

        let share = self.resources().try_global(|| {
            let mut metas = DeclareContext::metas::<T>()?;
            // Must sort here since the order of these metas is used to lock stores in target tables.
            metas.sort_unstable_by_key(|meta| meta.identifier());
            Ok(Share::<T>(Arc::new(metas), PhantomData))
        })?;
        let share = share.read();
        Ok(Add {
            database: self,
            keys: HashMap::new(),
            pending: Vec::new(),
            sorted: HashMap::new(),
            metas: share.0.clone(),
            pointers: Vec::new(),
        })
    }
}

impl<'d, T: Template> Add<'d, T> {
    #[inline]
    pub fn one(&mut self, key: Key, template: T) -> bool {
        match self.database.keys().get(key) {
            Ok((slot, table)) => Self::sort(
                key,
                slot,
                template,
                table,
                &self.metas,
                &mut self.sorted,
                self.database,
            )
            .is_ok(),
            Err(_) => false,
        }
    }

    #[inline]
    pub fn all<I: IntoIterator<Item = (Key, T)>>(&mut self, templates: I) -> usize {
        templates
            .into_iter()
            .filter_map(|(key, template)| self.one(key, template).then_some(()))
            .count()
    }

    pub fn resolve(&mut self) -> usize {
        self.resolve_sorted();
        while self.pending.len() > 0 {
            for (key, slot, template, table) in self.pending.drain(..) {
                let _ = Self::sort(
                    key,
                    slot,
                    template,
                    table,
                    &self.metas,
                    &mut self.sorted,
                    self.database,
                );
            }
            self.resolve_sorted();
        }
        let count = self.keys.len();
        self.keys.clear();
        count
    }

    fn resolve_sorted(&mut self) {
        // TODO: Maybe do not iterate over all pairs?
        for state in self.sorted.values_mut() {
            if state.rows.len() == 0 {
                continue;
            }

            // If locks are always taken in order (lower index first), there can not be a deadlock between move operations.
            let (mut source, target, low, high) = if state.source.index() < state.target.index() {
                let (source, low, high) = Self::filter(
                    &state.source,
                    &mut self.keys,
                    &mut state.rows,
                    &mut state.templates,
                    &mut self.pending,
                );
                if state.rows.len() == 0 {
                    // Happens if all keys from this table have been moved or destroyed between here and the sorting.
                    // - One less lock to take if all rows have been filtered.
                    continue;
                }
                let target = state.target.inner.upgradable_read();
                (source, target, low, high)
            } else if state.source.index() > state.target.index() {
                let target = state.target.inner.upgradable_read();
                let (source, low, high) = Self::filter(
                    &state.source,
                    &mut self.keys,
                    &mut state.rows,
                    &mut state.templates,
                    &mut self.pending,
                );
                if state.rows.len() == 0 {
                    // Happens if all keys from this table have been moved or destroyed between here and the sorting.
                    continue;
                }
                (source, target, low, high)
            } else {
                // The keys do not need to be moved, simply write the row data.
                let (inner, ..) = Self::filter(
                    &state.source,
                    &mut self.keys,
                    &mut state.rows,
                    &mut state.templates,
                    &mut self.pending,
                );
                if state.rows.len() > 0 && T::SIZE > 0 {
                    lock(&state.inner.add, &mut self.pointers, &inner, |pointers| {
                        for (i, template) in state.templates.drain(..).enumerate() {
                            let &(.., row) = unsafe { state.rows.get_unchecked(i) };
                            debug_assert!(row < u32::MAX);
                            let context = ApplyContext(pointers, row as _);
                            unsafe { template.apply(&state.inner.state, context) };
                        }
                        state.rows.clear();
                    });
                } else {
                    state.rows.clear();
                    state.templates.clear();
                }

                // Sanity checks.
                debug_assert!(state.rows.is_empty());
                debug_assert!(state.templates.is_empty());
                debug_assert!(self.pointers.is_empty());
                continue;
            };

            let count = state.rows.len();
            let (start, target) = table::Inner::reserve(target, count);
            // Move data from source to target.
            debug_assert!(low <= high);
            let source = &mut *source;
            let range = low..high + 1;
            let head = source.release(count);
            let keys = (source.keys.get_mut(), unsafe { &mut *target.keys.get() });
            let (low, high) = (range.start as usize, range.end as usize);

            if range.len() == count {
                // Fast path. The move range is contiguous. Copy everything from source to target at once.
                for &indices in state.inner.copy.iter() {
                    let source = unsafe { source.stores.get_unchecked_mut(indices.0) };
                    let target = unsafe { target.stores.get_unchecked(indices.1) };
                    let count = unsafe { NonZeroUsize::new_unchecked(count) };
                    unsafe {
                        Store::copy_to((source, low), (target, start), count);
                    };
                }
                keys.1[start..start + count].copy_from_slice(&keys.0[low..low + count]);

                // Swap remove without dropping.
                let over = high.saturating_sub(head);
                let end = count - over;
                if let Some(end) = NonZeroUsize::new(end) {
                    let start = head + over;
                    // Copy the range at the end of the table on the beginning of the removed range.
                    for store in source.stores.iter_mut() {
                        unsafe { store.copy(start, low, end) };
                    }

                    // Update the keys.
                    keys.0.copy_within(start..start + end.get(), low);
                    for i in low..low + end.get() {
                        unsafe { self.database.keys().get_unchecked(*keys.0.get_unchecked(i)) }
                            .update(i as _);
                    }
                }
            } else {
                // Range is not contiguous; use the slow path.
                let mut cursor = head;
                for (i, &(.., row)) in state.rows.iter().enumerate() {
                    let row = row as usize;

                    for &indices in state.inner.copy.iter() {
                        let source = unsafe { source.stores.get_unchecked_mut(indices.0) };
                        let target = unsafe { target.stores.get_unchecked(indices.1) };
                        let count = unsafe { NonZeroUsize::new_unchecked(1) };
                        unsafe {
                            Store::copy_to((source, row), (target, start + i), count);
                        };
                    }

                    // Tag keys that are going to be removed such that removed keys and valid keys can be differentiated.
                    let key = replace(unsafe { keys.0.get_unchecked_mut(row as usize) }, Key::NULL);
                    unsafe { *keys.1.get_unchecked_mut(start + i) = key };

                    if row < head {
                        // Find the next valid row to move.
                        while unsafe { *keys.0.get_unchecked(cursor) } == Key::NULL {
                            cursor += 1;
                        }
                        debug_assert!(cursor < head + count);

                        for store in source.stores.iter_mut() {
                            unsafe { store.squash(cursor, row, NonZeroUsize::new_unchecked(1)) };
                        }

                        let key = unsafe { *keys.0.get_unchecked_mut(cursor) };
                        unsafe { *keys.0.get_unchecked_mut(row) = key };
                        let slot = unsafe { self.database.keys().get_unchecked(key) };
                        slot.update(row as _);
                        cursor += 1;
                    }
                }
            }

            // Initialize missing data `T` in target.
            if T::SIZE == 0 {
                state.templates.clear();
            } else {
                for &index in state.inner.add.iter() {
                    // SAFETY: Since this row is not yet observable by any thread but this one, bypass locks.
                    let store = unsafe { target.stores.get_unchecked(index) };
                    self.pointers.push(unsafe { *store.data().data_ptr() });
                }

                let context = ApplyContext(&self.pointers, 0);
                for (i, template) in state.templates.drain(..).enumerate() {
                    unsafe { template.apply(&state.inner.state, context.with(start + i)) };
                }
                self.pointers.clear();
            }

            target.commit(count);
            // Slots must be updated after the table `commit` to prevent a `query::find` to be able to observe a row which
            // has an index greater than the `table.count()`. As long as the slots remain in the source table, all accesses
            // to these keys will block at the table access and will correct their table index after they acquire the source
            // table lock.
            for (i, (key, slot, ..)) in state.rows.drain(..).enumerate() {
                slot.initialize(key.generation(), state.target.index(), (start + i) as u32);
            }
            // Keep the `source` and `target` locks until all table operations are fully completed.
            drop(source);
            drop(target);

            // Sanity checks.
            debug_assert!(state.rows.is_empty());
            debug_assert!(state.templates.is_empty());
            debug_assert!(self.pointers.is_empty());
        }
    }

    fn sort(
        key: Key,
        slot: &'d Slot,
        template: T,
        table: u32,
        metas: &Vec<&'static Meta>,
        sorted: &mut HashMap<u32, State<'d, T>>,
        database: &'d Database,
    ) -> Result<(), Error> {
        struct Share<T: Template> {
            source: Arc<Table>,
            target: Arc<Table>,
            inner: Arc<Inner<T>>,
        }

        match sorted.get_mut(&table) {
            Some(state) => {
                state.rows.push((key, slot, u32::MAX));
                state.templates.push(template);
                Ok(())
            }
            None => {
                let share = database.resources().try_global_with(table, || {
                    let source = database.tables().get_shared(table as usize)?;
                    let mut target_metas = metas.clone();
                    target_metas.extend(source.inner.read().stores().iter().map(Store::meta));
                    let target = database.tables().find_or_add(target_metas, 0);
                    let map = metas
                        .iter()
                        .enumerate()
                        .map(|(i, meta)| (meta.identifier(), i))
                        .collect();
                    let state = T::initialize(InitializeContext(&map))?;

                    let mut copy = Vec::new();
                    for (source, identifier) in source.types().enumerate() {
                        copy.push((source, target.store_with(identifier)?));
                    }

                    let mut add = Vec::new();
                    for meta in metas.iter() {
                        add.push(target.store_with(meta.identifier())?);
                    }

                    debug_assert_eq!(source.types().len(), copy.len());
                    debug_assert_eq!(target.types().len(), copy.len() + add.len());

                    Ok(Share::<T> {
                        source,
                        target,
                        inner: Arc::new(Inner { state, add, copy }),
                    })
                })?;
                let share = share.read();
                sorted.insert(
                    table,
                    State {
                        source: share.source.clone(),
                        target: share.target.clone(),
                        inner: share.inner.clone(),
                        rows: vec![(key, slot, u32::MAX)],
                        templates: vec![template],
                    },
                );
                Ok(())
            }
        }
    }

    fn filter<'a>(
        table: &'a Table,
        keys: &mut HashMap<Key, (usize, u32)>,
        rows: &mut Vec<(Key, &'d Slot, u32)>,
        templates: &mut Vec<T>,
        pending: &mut Vec<(Key, &'d Slot, T, u32)>,
    ) -> (RwLockWriteGuard<'a, table::Inner>, u32, u32) {
        let mut low = u32::MAX;
        let mut high = 0;
        let mut index = 0;
        let inner = table.inner.write();
        while let Some((key, slot, row)) = rows.get_mut(index) {
            if let Ok(table_index) = slot.table(key.generation()) {
                if table_index == table.index() {
                    // Duplicates must only be checked here where the key would be guaranteed to be added.
                    // - This way, a proper count of added can be collected.
                    // - The removal algorithm also assumes that there is no duplicate rows.
                    match keys.insert(*key, (index, table_index)) {
                        Some(pair) if pair.1 == table_index => {
                            // If the key was already seen, keep only the more recent template.
                            templates.swap(pair.0, index);
                            rows.swap_remove(index);
                            templates.swap_remove(index);
                        }
                        _ => {
                            // It is possible that the key has already been processed in another table and moved to this one.
                            // - If this is the case, it needs to be reprocessed.
                            *row = slot.row();
                            low = low.min(*row);
                            high = high.max(*row);
                            index += 1;
                        }
                    }
                } else {
                    let (key, slot, _) = rows.swap_remove(index);
                    let template = templates.swap_remove(index);
                    pending.push((key, slot, template, table_index));
                    continue;
                }
            } else {
                rows.swap_remove(index);
                templates.swap_remove(index);
            }
        }
        (inner, low, high)
    }
}

impl<T: Template> Drop for Add<'_, T> {
    fn drop(&mut self) {
        self.resolve();
    }
}

#[inline]
fn lock<T>(
    indices: &[usize],
    pointers: &mut Vec<NonNull<()>>,
    inner: &table::Inner,
    with: impl FnOnce(&[NonNull<()>]) -> T,
) -> T {
    match indices.split_first() {
        Some((&index, rest)) => {
            let store = unsafe { inner.stores.get_unchecked(index) };
            if store.meta().size == 0 {
                pointers.push(unsafe { *store.data().data_ptr() });
                lock(rest, pointers, inner, with)
            } else {
                let guard = store.data().write();
                pointers.push(*guard);
                let state = lock(rest, pointers, inner, with);
                drop(guard);
                state
            }
        }
        None => {
            let state = with(pointers);
            pointers.clear();
            state
        }
    }
}
