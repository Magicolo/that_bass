use crate::{
    key::{Key, Slot},
    table::{self, Store, Table},
    template::{ApplyContext, DeclareContext, InitializeContext, Template},
    Database, Error, Meta,
};
use std::{
    collections::HashMap, marker::PhantomData, num::NonZeroUsize, ops::Deref, ptr::NonNull,
    sync::Arc,
};

pub struct Add<'d, T: Template> {
    database: &'d Database,
    pending: Vec<(Key, &'d Slot, T)>,
    sorted: HashMap<u32, State<'d, T>>,
    metas: Arc<Vec<&'static Meta>>,
    pointers: Vec<NonNull<()>>,
}

struct State<'d, T: Template> {
    source: Arc<Table>,
    target: Arc<Table>,
    inner: Arc<Inner<T>>,
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
            Ok(slot) => {
                self.pending.push((key, slot, template));
                true
            }
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

    pub fn resolve(&mut self) -> Result<(), Error> {
        struct Share<T: Template> {
            source: Arc<Table>,
            target: Arc<Table>,
            inner: Arc<Inner<T>>,
        }

        while self.pending.len() > 0 {
            for (key, slot, template) in self.pending.drain(..) {
                let (table, row) = slot.indices();
                // Slot has been destroyed.
                if table == u32::MAX {
                    continue;
                }

                match self.sorted.get_mut(&table) {
                    Some(state) => {
                        state.rows.push((key, slot, row));
                        state.templates.push(template);
                    }
                    None => {
                        let share = self.database.resources().try_global_with(table, || {
                            let source = self.database.tables().get_shared(table as usize)?;
                            let mut metas = self.metas.deref().clone();
                            metas.extend(source.inner.read().stores().iter().map(Store::meta));
                            let target = self.database.tables().find_or_add(metas, 0);
                            let map = self
                                .metas
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
                            for meta in self.metas.iter() {
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
                        self.sorted.insert(
                            table,
                            State {
                                source: share.source.clone(),
                                target: share.target.clone(),
                                inner: share.inner.clone(),
                                rows: vec![(key, slot, row)],
                                templates: vec![template],
                            },
                        );
                    }
                }
            }

            // TODO: Maybe do not iterate over all pairs?
            for state in self.sorted.values_mut() {
                if state.rows.len() == 0 {
                    continue;
                }

                // If locks are always taken in order (lower index first), there can not be a deadlock between move operations.
                let (mut source, target) = if state.source.index() < state.target.index() {
                    let source = state.source.inner.write();
                    let target = state.target.inner.upgradable_read();
                    (source, target)
                } else if state.source.index() > state.target.index() {
                    let target = state.target.inner.upgradable_read();
                    let source = state.source.inner.write();
                    (source, target)
                } else {
                    // The key does not need to be moved, simply write the row data.
                    let inner = state.source.inner.read();
                    let (count, _, _) = filter(
                        &mut state.rows,
                        &mut state.templates,
                        &mut self.pending,
                        state.source.index(),
                    );
                    if count > 0 && T::SIZE > 0 {
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

                let (count, low, high) = filter(
                    &mut state.rows,
                    &mut state.templates,
                    &mut self.pending,
                    state.source.index(),
                );
                if count == 0 {
                    // Happens if all keys from this table have been moved or destroyed between here and the sorting.
                    continue;
                }

                let (start, target) = table::Inner::reserve(target, count);
                // Move data from source to target.
                debug_assert!(low <= high);
                let source = &mut *source;
                let range = low..high + 1;
                let head = source.release(count);
                let keys = (source.keys.get_mut(), unsafe { &mut *target.keys.get() });
                let (low, high) = (range.start as usize, range.end as usize);

                if range.len() == count {
                    // Fast path. The move range is contiguous. Copy everything from source to target.
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
                                .update(state.source.index(), i as _);
                        }
                    }
                } else {
                    // TODO: Range is not contiguous...
                }

                // Initialize missing data `T` in target.
                if T::SIZE == 0 {
                    state.templates.clear();
                } else {
                    for &index in state.inner.add.iter() {
                        // SAFETY: Since this row is not yet observable by any thread but this one, no need to take locks.
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
                for (i, (_, slot, ..)) in state.rows.drain(..).enumerate() {
                    slot.update(state.target.index(), (start + i) as u32);
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

        Ok(())
    }
}

/// Requeues keys that have been moved or destroyed. For accurate results, call this while holding the `table` lock.
#[inline]
fn filter<'d, T>(
    rows: &mut Vec<(Key, &'d Slot, u32)>,
    templates: &mut Vec<T>,
    pending: &mut Vec<(Key, &'d Slot, T)>,
    table: u32,
) -> (usize, u32, u32) {
    let mut low = u32::MAX;
    let mut high = 0;
    let mut index = 0;
    while let Some((_, slot, row)) = rows.get_mut(index) {
        let indices = slot.indices();
        if indices.0 == table {
            *row = indices.1;
            low = low.min(*row);
            high = high.max(*row);
            index += 1;
        } else {
            let (key, slot, _) = rows.swap_remove(index);
            let template = templates.swap_remove(index);
            pending.push((key, slot, template));
        }
    }
    (rows.len(), low, high)
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
