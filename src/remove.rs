use crate::{
    key::{Key, Slot},
    table::Table,
    template::{DeclareContext, Template},
    Database, Error,
};
use std::{any::TypeId, collections::HashMap, ptr::NonNull, sync::Arc};

pub struct Remove<'d> {
    database: &'d Database,
    keys: HashMap<Key, u32>,
    pending: Vec<(Key, &'d Slot, u32)>,
    sorted: HashMap<(u32, TypeId), State<'d>>,
    pointers: Vec<NonNull<()>>,
}

struct State<'d> {
    source: Arc<Table>,
    target: Arc<Table>,
    inner: Arc<Inner>,
    // TODO: Merge `rows` and `templates`? How does this affect `slot.update/initialize`?
    rows: Vec<(Key, &'d Slot, u32)>,
}

struct Inner {
    copy: Vec<(usize, usize)>,
    remove: Vec<usize>,
}

impl Database {
    pub fn remove(&self) -> Result<Remove, Error> {
        Ok(Remove {
            database: self,
            keys: HashMap::new(),
            pending: Vec::new(),
            sorted: HashMap::new(),
            pointers: Vec::new(),
        })
    }
}

impl<'d> Remove<'d> {
    #[inline]
    pub fn one<T: Template>(&mut self, key: Key) -> bool {
        match self.database.keys().get(key) {
            Ok((slot, table)) => {
                Self::sort::<T>(key, slot, table, &mut self.sorted, self.database).is_ok()
            }
            Err(_) => false,
        }
    }

    #[inline]
    pub fn all<T: Template>(&mut self, keys: impl IntoIterator<Item = Key>) -> usize {
        keys.into_iter()
            .filter_map(|key| self.one::<T>(key).then_some(()))
            .count()
    }

    pub fn resolve(&mut self) -> usize {
        todo!()
    }

    fn sort<T: Template>(
        key: Key,
        slot: &'d Slot,
        table: u32,
        sorted: &mut HashMap<(u32, TypeId), State<'d>>,
        database: &'d Database,
    ) -> Result<(), Error> {
        struct Share {
            source: Arc<Table>,
            target: Arc<Table>,
            inner: Arc<Inner>,
        }

        match sorted.get_mut(&(table, TypeId::of::<T>())) {
            Some(state) => {
                state.rows.push((key, slot, u32::MAX));
                Ok(())
            }
            None => {
                let share = database.resources().try_global_with(table, || {
                    let metas = DeclareContext::metas::<T>()?;
                    let source = database.tables().get_shared(table as usize)?;
                    let mut target_metas = Vec::new();
                    for store in source.inner.read().stores() {
                        if metas.contains(&store.meta()) {
                            continue;
                        } else {
                            target_metas.push(store.meta());
                        }
                    }
                    let target = database.tables().find_or_add(target_metas);

                    let mut copy = Vec::new();
                    for (target, identifier) in target.types().enumerate() {
                        copy.push((source.store_with(identifier)?, target));
                    }

                    let mut remove = Vec::new();
                    for meta in metas.iter() {
                        remove.push(source.store_with(meta.identifier())?);
                    }

                    debug_assert_eq!(source.types().len(), copy.len() + remove.len());
                    debug_assert_eq!(target.types().len(), copy.len());

                    Ok(Share {
                        source,
                        target,
                        inner: Arc::new(Inner { copy, remove }),
                    })
                })?;
                let share = share.read();
                sorted.insert(
                    (table, TypeId::of::<T>()),
                    State {
                        source: share.source.clone(),
                        target: share.target.clone(),
                        inner: share.inner.clone(),
                        rows: vec![(key, slot, u32::MAX)],
                    },
                );
                Ok(())
            }
        }
    }
}

impl Drop for Remove<'_> {
    fn drop(&mut self) {
        self.resolve();
    }
}
