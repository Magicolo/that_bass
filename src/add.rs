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
    templates: Vec<(Key, &'d Slot, T)>,
}

struct Inner<T: Template> {
    state: T::State,
    add: Vec<usize>,
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
                let table = slot.table();
                match self.sorted.get_mut(&table) {
                    Some(state) => state.templates.push((key, slot, template)),
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
                                templates: vec![(key, slot, template)],
                            },
                        );
                    }
                }
            }

            // TODO: Maybe do not iterate over all pairs?
            for state in self.sorted.values_mut() {
                if state.templates.len() == 0 {
                    continue;
                }

                let source_index = state.source.index();
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
                    if filter(&mut state.templates, &mut self.pending, source_index) == 0
                        || T::SIZE == 0
                    {
                        state.templates.clear();
                    } else {
                        lock(&state.inner.add, &mut self.pointers, &inner, |pointers| {
                            for (_, slot, template) in state.templates.drain(..) {
                                let context = ApplyContext(pointers, slot.row() as _);
                                unsafe { template.apply(&state.inner.state, context) };
                            }
                        });
                    }

                    // Sanity checks.
                    debug_assert!(state.templates.is_empty());
                    debug_assert!(self.pointers.is_empty());
                    continue;
                };

                let count = filter(&mut state.templates, &mut self.pending, source_index);
                if count == 0 {
                    // Happens if all keys from this table have been moved or destroyed between here and the sorting.
                    continue;
                }

                // TODO: Remove from source.
                {}

                let (start, target) = table::Inner::reserve(target, count);
                // Move data from source to target.
                for &pair in state.inner.copy.iter() {
                    let source = unsafe { source.stores.get_unchecked_mut(pair.0) };
                    let target = unsafe { target.stores.get_unchecked(pair.1) };
                    // TODO: Batch operations?
                    for (i, (_, slot, _)) in state.templates.iter().enumerate() {
                        unsafe {
                            Store::copy(
                                (source, slot.row() as _),
                                (target, start + i),
                                NonZeroUsize::new_unchecked(1),
                            );
                        };
                    }
                }

                if T::SIZE == 0 {
                    // No data to initialize.
                    for (i, (key, slot, _)) in state.templates.drain(..).enumerate() {
                        unsafe { *(&mut **target.keys.get()).get_unchecked_mut(start + i) = key };
                        slot.update(state.target.index(), (start + i) as u32);
                    }
                } else {
                    // Initialize table rows.
                    self.pointers
                        .extend(state.inner.add.iter().map(|&i| unsafe {
                            // SAFETY: Since this row is not yet observable by any thread but this one, no need to take locks.
                            *target.stores.get_unchecked(i).data().data_ptr()
                        }));
                    let context = ApplyContext(&self.pointers, 0);
                    for (i, (key, slot, template)) in state.templates.drain(..).enumerate() {
                        unsafe { *(&mut **target.keys.get()).get_unchecked_mut(start + i) = key };
                        unsafe { template.apply(&state.inner.state, context.with(start + i)) };
                        slot.update(state.target.index(), (start + i) as u32);
                    }
                    self.pointers.clear();
                }
                table::Inner::commit(target, count);

                // Sanity checks.
                debug_assert!(state.templates.is_empty());
                debug_assert!(self.pointers.is_empty());
            }
        }

        Ok(())
    }
}

/// Requeues keys that have been moved or destroyed. For accurate results, call this while holding a `state.target.inner` lock.
#[inline]
fn filter<'d, T>(
    sources: &mut Vec<(Key, &'d Slot, T)>,
    targets: &mut Vec<(Key, &'d Slot, T)>,
    table: u32,
) -> usize {
    let drain = sources
        // Moved keys may be requeued.
        .drain_filter(|(_, slot, _)| slot.table() != table)
        // Destroyed keys are removed.
        .filter(|(key, slot, _)| key.generation() == slot.generation());
    targets.extend(drain);
    sources.len()
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
