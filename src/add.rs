use crate::{
    filter::Filter,
    key::{Key, Slot},
    resources::Global,
    row::{self, Row},
    table::{self, Store, Table},
    template::{ApplyContext, DeclareContext, InitializeContext, Template},
    try_each_swap, Database, Error, Meta,
};
use parking_lot::{RwLockReadGuard, RwLockUpgradableReadGuard, RwLockWriteGuard};
use std::{collections::HashMap, marker::PhantomData, num::NonZeroUsize, ptr::NonNull, sync::Arc};

pub struct AddItem<'a, T: Template>(u32, &'a mut Vec<(u32, T)>);
pub struct AddChunk<'a, T: Template>(State<'a, T>);

/// Adds template `T` to accumulated add operations.
pub struct Add<'d, T: Template> {
    database: &'d Database,
    keys: HashMap<Key, u32>,
    pending: Vec<(Key, &'d Slot, T, u32)>,
    sorted: HashMap<u32, State<'d, T>>,
    metas: Arc<Vec<&'static Meta>>,
    pointers: Vec<NonNull<()>>,
}

/// Adds template `T` to all keys in tables that satisfy the filter `F`.
pub struct AddAll<'d, T: Template, F: Filter> {
    database: &'d Database,
    index: usize,
    states: Vec<StateAll<T>>,
    metas: Arc<Vec<&'static Meta>>,
    pointers: Vec<NonNull<()>>,
    _marker: PhantomData<fn(F)>,
}

struct State<'d, T: Template> {
    source: Arc<Table>,
    target: Arc<Table>,
    inner: Arc<Inner<T>>,
    // TODO: Merge `rows` and `templates`? How does this affect `slot.update/initialize`?
    rows: Vec<(Key, &'d Slot, u32)>,
    templates: Vec<T>,
}

struct StateAll<T: Template> {
    source: Arc<Table>,
    target: Arc<Table>,
    inner: Arc<Inner<T>>,
}

struct StateRow<T: Template> {
    source: Arc<Table>,
    target: Arc<Table>,
    inner: Arc<Inner<T>>,
    rows: Vec<(u32, T)>,
}

struct Inner<T: Template> {
    state: T::State,
    add: Vec<usize>,
    // TODO: A `Vec<usize>` should suffice here where its indices map to its values `source_store -> target_store`.
    copy: Vec<(usize, usize)>,
}

struct ShareMetas<T: Template>(Arc<Vec<&'static Meta>>, PhantomData<fn(T)>);

struct ShareTable<T: Template> {
    source: Arc<Table>,
    target: Arc<Table>,
    inner: Arc<Inner<T>>,
}

impl Database {
    pub fn add<T: Template>(&self) -> Result<Add<T>, Error> {
        let share = ShareMetas::<T>::from(self)?;
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

    pub fn add_all<T: Template, F: Filter>(&self) -> Result<AddAll<T, F>, Error> {
        let share = ShareMetas::<T>::from(self)?;
        let share = share.read();
        Ok(AddAll {
            database: self,
            index: 0,
            states: Vec::new(),
            metas: share.0.clone(),
            pointers: Vec::new(),
            _marker: PhantomData,
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
            let (mut source, target, low, high, count) = if state.source.index()
                < state.target.index()
            {
                let source = state.source.inner.write();
                let (low, high) = Self::filter(
                    &state.source,
                    &mut self.keys,
                    &mut state.rows,
                    &mut state.templates,
                    &mut self.pending,
                );
                let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                    // Happens if all keys from this table have been moved or destroyed between here and the sorting.
                    continue;
                };
                let target = state.target.inner.upgradable_read();
                (source, target, low, high, count)
            } else if state.source.index() > state.target.index() {
                let target = state.target.inner.upgradable_read();
                let source = state.source.inner.write();
                let (low, high) = Self::filter(
                    &state.source,
                    &mut self.keys,
                    &mut state.rows,
                    &mut state.templates,
                    &mut self.pending,
                );
                let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                    // Happens if all keys from this table have been moved or destroyed between here and the sorting.
                    continue;
                };
                (source, target, low, high, count)
            } else {
                // The keys do not need to be moved, simply write the row data.
                let inner = state.source.inner.read();
                Self::filter(
                    &state.source,
                    &mut self.keys,
                    &mut state.rows,
                    &mut state.templates,
                    &mut self.pending,
                );
                if state.rows.len() > 0 && T::SIZE > 0 {
                    lock(&state.inner.add, &mut self.pointers, &inner, |pointers| {
                        let context = ApplyContext(pointers, 0);
                        for (i, template) in state.templates.drain(..).enumerate() {
                            let &(.., row) = unsafe { state.rows.get_unchecked(i) };
                            debug_assert!(row < u32::MAX);
                            unsafe { template.apply(&state.inner.state, context.with(row as _)) };
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

            let (start, target) = table::Inner::reserve(target, count);
            // Move data from source to target.
            let source = &mut *source;
            let range = low..high + 1;
            let head = source.release(count);
            let keys = (source.keys.get_mut(), unsafe { &mut *target.keys.get() });
            let (low, high) = (range.start as usize, range.end as usize);

            if range.len() == count.get() {
                // Fast path. The move range is contiguous. Copy everything from source to target at once.
                copy(
                    (low, keys.0, &mut source.stores),
                    (start, keys.1, &target.stores),
                    count,
                    &state.inner.copy,
                );

                // Swap remove without dropping.
                let over = high.saturating_sub(head);
                let end = count.get() - over;
                if let Some(end) = NonZeroUsize::new(end) {
                    let start = head + over;
                    // Copy the range at the end of the table on the beginning of the removed range.
                    for store in source.stores.iter_mut() {
                        unsafe { store.copy(start, low, end) };
                    }

                    // Update the keys.
                    keys.0.copy_within(start..start + end.get(), low);
                    self.database.keys().update(keys.0, low..low + end.get());
                }
            } else {
                // Range is not contiguous; use the slow path.
                for (i, &(.., row)) in state.rows.iter().enumerate() {
                    copy(
                        (row as usize, keys.0, &mut source.stores),
                        (start + i, keys.1, &target.stores),
                        NonZeroUsize::MIN,
                        &state.inner.copy,
                    );
                    // Tag keys that are going to be removed such that removed keys and valid keys can be differentiated.
                    unsafe { *keys.0.get_unchecked_mut(row as usize) = Key::NULL };
                }

                let mut cursor = head;
                for &(.., row) in state.rows.iter() {
                    let row = row as usize;
                    if row < head {
                        // Find the next valid row to move.
                        while unsafe { *keys.0.get_unchecked(cursor) } == Key::NULL {
                            cursor += 1;
                        }
                        debug_assert!(cursor < head + count.get());

                        for store in source.stores.iter_mut() {
                            unsafe { store.squash(cursor, row, NonZeroUsize::MIN) };
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
        match sorted.get_mut(&table) {
            Some(state) => {
                state.rows.push((key, slot, u32::MAX));
                state.templates.push(template);
                Ok(())
            }
            None => {
                let share = ShareTable::from(table, metas, database)?;
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

    /// Call this while holding a lock on `table`.
    fn filter<'a>(
        table: &'a Table,
        keys: &mut HashMap<Key, u32>,
        rows: &mut Vec<(Key, &'d Slot, u32)>,
        templates: &mut Vec<T>,
        pending: &mut Vec<(Key, &'d Slot, T, u32)>,
    ) -> (u32, u32) {
        let mut low = u32::MAX;
        let mut high = 0;
        let mut index = rows.len();

        // Iterate in reverse to prevent the `ABBAA` problem where `A2` is considered the latest `A` in place of `A3`. This happened
        // with the previous template swapping algorithm.
        while index > 0 {
            index -= 1;

            let (key, slot, row) = unsafe { rows.get_unchecked_mut(index) };
            if let Ok(table_index) = slot.table(key.generation()) {
                if table_index == table.index() {
                    // Duplicates must only be checked here where the key would be guaranteed to be added.
                    // - This way, a proper count of added can be collected.
                    // - The removal algorithm also assumes that there is no duplicate rows.
                    match keys.insert(*key, table_index) {
                        Some(table) if table == table_index => {
                            // If the key was already seen, discard the earlier template.
                            rows.swap_remove(index);
                            templates.swap_remove(index);
                        }
                        _ => {
                            // It is possible that the key has already been processed in another table and moved to this one.
                            // - If this is the case, it needs to be reprocessed.
                            *row = slot.row();
                            low = low.min(*row);
                            high = high.max(*row);
                        }
                    }
                } else {
                    let (key, slot, _) = rows.swap_remove(index);
                    let template = templates.swap_remove(index);
                    pending.push((key, slot, template, table_index));
                }
            } else {
                rows.swap_remove(index);
                templates.swap_remove(index);
            }
        }

        debug_assert!(low <= high);
        (low, high)
    }
}

impl<T: Template> Drop for Add<'_, T> {
    fn drop(&mut self) {
        self.resolve();
    }
}

unsafe impl<T: Template> Row for Add<'_, T> {
    type State = Vec<(u32, T)>;
    type Read = ();
    type Item<'a> = AddItem<'a, T>;
    type Chunk<'a> = AddChunk<'a, T>;

    fn declare(mut context: row::DeclareContext) -> Result<(), Error> {
        for meta in DeclareContext::metas::<T>()? {
            context.add_with(meta.identifier())?;
        }
        Ok(())
    }

    fn initialize(_: row::InitializeContext) -> Result<Self::State, Error> {
        Ok(Vec::new())
    }

    fn read(_: Self::State) -> <Self::Read as Row>::State {}

    #[inline]
    fn item<'a>(state: &'a mut Self::State, context: row::ItemContext<'a, '_>) -> Self::Item<'a> {
        AddItem(context.row() as _, state)
    }

    #[inline]
    fn chunk<'a>(
        state: &'a mut Self::State,
        context: row::ChunkContext<'a, '_>,
    ) -> Self::Chunk<'a> {
        todo!()
    }
}

impl<T: Template> AddItem<'_, T> {
    #[inline]
    pub fn one(self, template: T) {
        self.1.push((self.0, template));
    }

    #[inline]
    pub fn default(self)
    where
        T: Default,
    {
        self.one(T::default())
    }
}

impl<'d, T: Template, F: Filter> AddAll<'d, T, F> {
    #[inline]
    pub fn resolve(&mut self, set: bool) -> usize
    where
        T: Default,
    {
        self.resolve_with(set, T::default)
    }

    pub fn resolve_with<G: FnMut() -> T>(&mut self, set: bool, with: G) -> usize {
        while let Ok(table) = self.database.tables().get(self.index) {
            self.index += 1;
            if let Ok(share) = ShareTable::from(table.index(), &self.metas, self.database) {
                let share = share.read();
                self.states.push(StateAll {
                    source: share.source.clone(),
                    target: share.target.clone(),
                    inner: share.inner.clone(),
                });
            }
        }

        let (sum, ..) = try_each_swap(
            &mut self.states,
            (0, with, &mut self.pointers),
            |(sum, with, pointers), state| {
                *sum += if state.source.index() < state.target.index() {
                    let source = state.source.inner.try_write()?;
                    let target = state.target.inner.try_upgradable_read()?;
                    Self::resolve_tables(source, target, state, pointers, self.database, with)
                } else if state.source.index() > state.target.index() {
                    let target = state.target.inner.try_upgradable_read()?;
                    let source = state.source.inner.try_write()?;
                    Self::resolve_tables(source, target, state, pointers, self.database, with)
                } else if T::SIZE > 0 && set {
                    let inner = state.source.inner.try_read()?;
                    Self::resolve_table(inner, state, pointers, with)
                } else {
                    0
                };
                Some(())
            },
            |(sum, with, pointers), state| {
                *sum += if state.source.index() < state.target.index() {
                    let source = state.source.inner.write();
                    let target = state.target.inner.upgradable_read();
                    Self::resolve_tables(source, target, state, pointers, self.database, with)
                } else if state.source.index() > state.target.index() {
                    let target = state.target.inner.upgradable_read();
                    let source = state.source.inner.write();
                    Self::resolve_tables(source, target, state, pointers, self.database, with)
                } else if T::SIZE > 0 && set {
                    let inner = state.source.inner.read();
                    Self::resolve_table(inner, state, pointers, with)
                } else {
                    0
                };
            },
        );
        sum
    }

    fn resolve_tables(
        mut source: RwLockWriteGuard<table::Inner>,
        target: RwLockUpgradableReadGuard<table::Inner>,
        state: &StateAll<T>,
        pointers: &mut Vec<NonNull<()>>,
        database: &Database,
        with: &mut impl FnMut() -> T,
    ) -> usize {
        let Some(count) = NonZeroUsize::new(*source.count.get_mut() as _) else {
            return 0;
        };
        source.release(count);

        let (start, target) = table::Inner::reserve(target, count);
        let (source, target) = (&mut *source, &*target);
        let keys = (source.keys.get_mut(), unsafe { &mut *target.keys.get() });
        copy(
            (0, keys.0, &mut source.stores),
            (start, keys.1, &target.stores),
            count,
            &state.inner.copy,
        );

        if T::SIZE > 0 {
            for &index in state.inner.add.iter() {
                // SAFETY: Since this row is not yet observable by any thread but this one, bypass locks.
                pointers.push(unsafe { *target.stores.get_unchecked(index).data().data_ptr() });
            }

            let context = ApplyContext(pointers, 0);
            for i in 0..count.get() {
                unsafe { with().apply(&state.inner.state, context.with(start + i)) };
            }
            pointers.clear();
        }

        target.commit(count);
        // Slots must be updated after the table `commit` to prevent a `query::find` to be able to observe a row which
        // has an index greater than the `table.count()`. As long as the slots remain in the source table, all accesses
        // to these keys will block at the table access and will correct their table index after they acquire the source
        // table lock.
        database
            .keys()
            .initialize(keys.1, state.target.index(), start..start + count.get());
        count.get()
        // Keep the `source` and `target` locks until all table operations are fully completed.
    }

    fn resolve_table(
        inner: RwLockReadGuard<table::Inner>,
        state: &StateAll<T>,
        pointers: &mut Vec<NonNull<()>>,
        with: &mut impl FnMut() -> T,
    ) -> usize {
        let Some(count) = NonZeroUsize::new(inner.count() as _) else {
            return 0;
        };

        lock(&state.inner.add, pointers, &inner, |pointers| {
            let context = ApplyContext(&pointers, 0);
            for i in 0..count.get() {
                unsafe { with().apply(&state.inner.state, context.with(i)) };
            }
        });
        count.get()
    }
}

impl<T: Template> ShareMetas<T> {
    pub fn from(database: &Database) -> Result<Global<Self>, Error> {
        database.resources().try_global(|| {
            let mut metas = DeclareContext::metas::<T>()?;
            // Must sort here since the order of these metas is used to lock stores in target tables.
            metas.sort_unstable_by_key(|meta| meta.identifier());
            Ok(ShareMetas::<T>(Arc::new(metas), PhantomData))
        })
    }
}

impl<T: Template> ShareTable<T> {
    pub fn from(
        table: u32,
        metas: &Vec<&'static Meta>,
        database: &Database,
    ) -> Result<Global<Self>, Error> {
        database.resources().try_global_with(table, || {
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

            Ok(ShareTable::<T> {
                source,
                target,
                inner: Arc::new(Inner { state, add, copy }),
            })
        })
    }
}

#[inline]
fn copy(
    source: (usize, &[Key], &mut [Store]),
    target: (usize, &mut [Key], &[Store]),
    count: NonZeroUsize,
    indices: &[(usize, usize)],
) {
    target.1[target.0..target.0 + count.get()]
        .copy_from_slice(&source.1[source.0..source.0 + count.get()]);
    for &indices in indices {
        let source = (unsafe { source.2.get_unchecked_mut(indices.0) }, source.0);
        let target = (unsafe { target.2.get_unchecked(indices.1) }, target.0);
        unsafe { Store::copy_to(source, target, count) };
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
