use crate::{
    database::Database,
    key::{Key, Slot},
    table::{self, Store, Table},
    template::{ApplyContext, DeclareContext, InitializeContext, Template},
    Error, Meta,
};
use std::{
    any::TypeId, collections::HashMap, marker::PhantomData, ops::Deref, ptr::NonNull, sync::Arc,
};

pub struct Add<'d, T: Template> {
    database: &'d Database,
    // TODO: Is the map really necessary? Use a vec?
    pending: HashMap<Key, (&'d Slot, T)>, // The `HashMap` ensures that there is no duplicate key (the most recent `T` is kept).
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

struct Inner<T: Template>(T::State, Vec<(usize, TypeId)>);

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
            pending: HashMap::new(),
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
                self.pending.insert(key, (slot, template));
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
            for (key, (slot, template)) in self.pending.drain() {
                let table = slot.table();
                match self.sorted.get_mut(&table) {
                    Some(state) => state.templates.push((key, slot, template)),
                    None => {
                        let share = self.database.resources().try_global_with(table, || {
                            let source = self.database.tables().get_shared(table as usize)?;
                            let mut metas = self.metas.deref().clone();
                            metas.extend(source.inner.read().stores().iter().map(Store::meta));
                            let target = self.database.tables().find_or_add(metas, 0);

                            let mut indices = Vec::with_capacity(self.metas.len());
                            for meta in self.metas.iter() {
                                let identifier = meta.identifier();
                                indices.push((target.store_with(identifier)?, identifier));
                            }
                            let map = indices
                                .iter()
                                .enumerate()
                                .map(|(i, &(_, identifier))| (identifier, i))
                                .collect();
                            let state = T::initialize(InitializeContext(&map))?;
                            Ok(Share::<T> {
                                source,
                                target,
                                inner: Arc::new(Inner(state, indices)),
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
                } else if state.source.index() == state.target.index() {
                    // The key does not need to be moved, simply write the row data.
                    let inner = state.source.inner.read();
                    lock(&state.inner.1, &mut self.pointers, &inner, |pointers| {
                        for (key, slot, template) in state.templates.drain(..) {
                            let (table, row) = slot.indices();
                            if table == state.source.index() {
                                let context = ApplyContext(pointers, row as _);
                                unsafe { template.apply(&state.inner.0, context) };
                            } else {
                                self.pending.insert(key, (slot, template));
                            }
                        }
                    });

                    // Sanity checks.
                    debug_assert!(state.templates.is_empty());
                    debug_assert!(self.pointers.is_empty());
                } else {

                    // TODO!
                    // let inner = state.source.inner.write();
                    // self.database.move_to_table(
                    //     &state.keys,
                    //     &state.source,
                    //     &state.target,
                    //     |range, stores| {
                    //         // TODO
                    //     },
                    // );
                    // state.keys.clear();
                }
            }
        }

        Ok(())
    }
}

#[inline]
fn lock<T>(
    indices: &[(usize, TypeId)],
    pointers: &mut Vec<NonNull<()>>,
    inner: &table::Inner,
    with: impl FnOnce(&[NonNull<()>]) -> T,
) -> T {
    match indices.split_first() {
        Some((&(index, identifier), rest)) => {
            let store = unsafe { inner.stores.get_unchecked(index) };
            debug_assert_eq!(identifier, store.meta().identifier());

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
