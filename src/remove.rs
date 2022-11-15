use parking_lot::{RwLockUpgradableReadGuard, RwLockWriteGuard};

use crate::{
    core::utility::{fold_swap, get_unchecked, get_unchecked_mut, ONE},
    event::Listen,
    filter::Filter,
    key::{Key, Slot},
    resources::Resources,
    table::{Column, Table, Tables},
    template::{ShareMeta, Template},
    Database, Error,
};
use std::{
    collections::HashSet,
    marker::PhantomData,
    num::NonZeroUsize,
    sync::{atomic::Ordering, Arc},
};

pub struct Remove<'d, T: Template, F, L> {
    database: &'d Database<L>,
    keys: HashSet<Key>, // A `HashSet` is used because the move algorithm assumes that rows will be unique.
    indices: Vec<usize>, // May be reordered (ex: by `fold_swap`).
    states: Vec<Result<State<'d>, u32>>, // Must remain sorted by `state.source.index()` for `binary_search` to work.
    pending: Vec<(Key, &'d Slot, u32)>,
    moves: Vec<(usize, usize, NonZeroUsize)>,
    copies: Vec<(usize, usize, NonZeroUsize)>,
    filter: F,
    _marker: PhantomData<fn(T)>,
}

/// Removes template `T` to all keys in tables that satisfy the filter `F`.
pub struct RemoveAll<'d, T: Template, F, L> {
    database: &'d Database<L>,
    index: usize,
    states: Vec<StateAll>,
    filter: F,
    _marker: PhantomData<fn(T)>,
}

type Rows<'d> = Vec<(Key, &'d Slot, u32)>;

struct State<'d> {
    source: Arc<Table>,
    target: Arc<Table>,
    rows: Rows<'d>,
}

struct StateAll {
    source: Arc<Table>,
    target: Arc<Table>,
}

struct ShareTable<T> {
    source: Arc<Table>,
    target: Arc<Table>,
    _marker: PhantomData<fn(T)>,
}

impl<L> Database<L> {
    pub fn remove<T: Template>(&self) -> Result<Remove<T, (), L>, Error> {
        // Validate metas here, but there is no need to store them.
        ShareMeta::<T>::from(self.resources()).map(|_| Remove {
            database: self,
            keys: HashSet::new(),
            pending: Vec::new(),
            states: Vec::new(),
            indices: Vec::new(),
            copies: Vec::new(),
            moves: Vec::new(),
            filter: (),
            _marker: PhantomData,
        })
    }

    pub fn remove_all<T: Template>(&self) -> Result<RemoveAll<T, (), L>, Error> {
        // Validate metas here, but there is no need to store them.
        ShareMeta::<T>::from(self.resources()).map(|_| RemoveAll {
            database: self,
            index: 0,
            states: Vec::new(),
            filter: (),
            _marker: PhantomData,
        })
    }
}

impl<'d, T: Template, F, L> Remove<'d, T, F, L> {
    // Note: `one` and `all` methods should minimize the amount of work they do since they are meant to be called inside a query,
    // thus while potentially holding many locks. Therefore, all validation is moved to `resolve` instead.
    #[inline]
    pub fn one(&mut self, key: Key) {
        self.keys.insert(key);
    }

    #[inline]
    pub fn all(&mut self, keys: impl IntoIterator<Item = Key>) {
        self.keys.extend(keys);
    }

    pub fn filter<G: Filter + Default>(self) -> Remove<'d, T, (F, G), L> {
        self.filter_with(G::default())
    }

    pub fn filter_with<G: Filter>(mut self, filter: G) -> Remove<'d, T, (F, G), L> {
        for state in self.states.iter_mut() {
            let index = match state {
                Ok(state) if filter.filter(&state.source, self.database.into()) => None,
                Ok(state) => Some(state.source.index()),
                Err(_) => None,
            };
            if let Some(index) = index {
                *state = Err(index);
            }
        }
        Remove {
            database: self.database,
            keys: self.keys,
            pending: self.pending,
            filter: (self.filter, filter),
            indices: self.indices,
            states: self.states,
            copies: self.copies,
            moves: self.moves,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = Key> + '_ {
        self.keys.iter().copied()
    }

    pub fn drain(&mut self) -> impl ExactSizeIterator<Item = Key> + '_ {
        debug_assert_eq!(self.pending.len(), 0);
        debug_assert_eq!(self.indices.len(), 0);
        self.keys.drain()
    }

    pub fn clear(&mut self) {
        debug_assert_eq!(self.pending.len(), 0);
        debug_assert_eq!(self.indices.len(), 0);
        self.keys.clear();
    }
}

impl<'d, T: Template, F: Filter, L: Listen> Remove<'d, T, F, L> {
    pub fn resolve(&mut self) -> usize {
        for (key, result) in self.database.keys().get_all(self.keys.iter().copied()) {
            if let Ok((slot, table)) = result {
                Self::sort(
                    key,
                    slot,
                    table,
                    &mut self.indices,
                    &mut self.states,
                    &self.filter,
                    &self.database.inner,
                );
            }
        }

        let mut sum = 0;
        loop {
            sum += self.resolve_sorted();
            self.indices.clear();
            if self.pending.len() == 0 {
                break;
            }
            for (key, slot, table) in self.pending.drain(..) {
                Self::sort(
                    key,
                    slot,
                    table,
                    &mut self.indices,
                    &mut self.states,
                    &self.filter,
                    &self.database.inner,
                );
            }
        }
        self.keys.clear();
        debug_assert_eq!(self.keys.len(), 0);
        debug_assert_eq!(self.moves.len(), 0);
        debug_assert_eq!(self.copies.len(), 0);
        debug_assert_eq!(self.indices.len(), 0);
        sum
    }

    fn resolve_sorted(&mut self) -> usize {
        fold_swap(
            &mut self.indices,
            0,
            (
                &mut self.states,
                &mut self.pending,
                &mut self.moves,
                &mut self.copies,
            ),
            |sum, (states, pending, moves, copies), index| {
                let result = unsafe { get_unchecked_mut(states, *index) };
                let state = unsafe { result.as_mut().unwrap_unchecked() };
                debug_assert_ne!(state.source.index(), state.target.index());
                debug_assert!(state.rows.len() > 0);
                let source = state.source.keys.try_upgradable_read().ok_or(sum)?;
                let target = state.target.keys.try_upgradable_read().ok_or(sum)?;
                let (low, high) = Self::retain(&state.source, &mut state.rows, pending);
                let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                    // Happens if all keys from this table have been moved or destroyed between here and the sorting.
                    return Ok(sum);
                };
                move_to(
                    self.database,
                    &self.keys,
                    moves,
                    copies,
                    (&state.source, source),
                    (&state.target, target),
                    (low, high, count),
                    &mut state.rows,
                );
                Ok(sum + count.get())
            },
            |sum, (states, pending, moves, copies), index| {
                let result = unsafe { get_unchecked_mut(states, *index) };
                let state = unsafe { result.as_mut().unwrap_unchecked() };
                debug_assert_ne!(state.source.index(), state.target.index());
                debug_assert!(state.rows.len() > 0);
                let (source, target, low, high, count) =
                    // If locks are always taken in order (lower index first), there can not be a deadlock between move operations.
                    if state.source.index() < state.target.index() {
                        let source = state.source.keys.upgradable_read();
                        let (low, high) = Self::retain(&state.source, &mut state.rows, pending);
                        let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                            // Happens if all keys from this table have been moved or destroyed between here and the sorting.
                            return sum;
                        };
                        let target = state.target.keys.upgradable_read();
                        (source, target, low, high, count)
                    } else {
                        let target = state.target.keys.upgradable_read();
                        let source = state.source.keys.upgradable_read();
                        let (low, high) = Self::retain(&state.source, &mut state.rows, pending);
                        let Some(count) = NonZeroUsize::new(state.rows.len()) else {
                            // Happens if all keys from this table have been moved or destroyed between here and the sorting.
                            return sum ;
                        };
                        (source, target, low, high, count)
                    };
                move_to(
                    self.database,
                    &self.keys,
                    moves,
                    copies,
                    (&state.source, source),
                    (&state.target, target),
                    (low, high, count),
                    &mut state.rows,
                );
                sum + count.get()
            },
        )
    }

    fn sort(
        key: Key,
        slot: &'d Slot,
        table: u32,
        indices: &mut Vec<usize>,
        states: &mut Vec<Result<State<'d>, u32>>,
        filter: &F,
        database: &'d crate::Inner,
    ) {
        let index = match states.binary_search_by_key(&table, |result| match result {
            Ok(state) => state.source.index(),
            Err(index) => *index,
        }) {
            Ok(index) => index,
            Err(index) => {
                let result =
                    match ShareTable::<T>::from(table, &database.tables, &database.resources) {
                        Ok((source, target)) if filter.filter(&source, database.into()) => {
                            Ok(State {
                                source,
                                target,
                                rows: Vec::new(),
                            })
                        }
                        _ => Err(table),
                    };
                for i in indices.iter_mut().filter(|i| **i >= index) {
                    *i += 1;
                }
                states.insert(index, result);
                index
            }
        };
        if let Ok(state) = unsafe { get_unchecked_mut(states, index) } {
            if state.rows.len() == 0 {
                indices.push(index);
            }
            state.rows.push((key, slot, u32::MAX));
        }
    }

    /// Call this while holding a lock on `table`.
    fn retain<'a>(
        table: &'a Table,
        rows: &mut Rows<'d>,
        pending: &mut Vec<(Key, &'d Slot, u32)>,
    ) -> (u32, u32) {
        let mut low = u32::MAX;
        let mut high = 0;
        for i in (0..rows.len()).rev() {
            let (key, slot, row) = unsafe { get_unchecked_mut(rows, i) };
            if let Ok(new_table) = slot.table(*key) {
                if new_table == table.index() {
                    *row = slot.row();
                    low = low.min(*row);
                    high = high.max(*row);
                } else {
                    let (key, slot, _) = rows.swap_remove(i);
                    pending.push((key, slot, new_table));
                }
            } else {
                rows.swap_remove(i);
            }
        }
        debug_assert_eq!(low <= high, rows.len() > 0);
        (low, high)
    }
}

impl<'d, T: Template, F, L> RemoveAll<'d, T, F, L> {
    pub fn filter<G: Filter + Default>(self) -> RemoveAll<'d, T, (F, G), L> {
        self.filter_with(G::default())
    }

    pub fn filter_with<G: Filter>(mut self, filter: G) -> RemoveAll<'d, T, (F, G), L> {
        self.states
            .retain(|state| filter.filter(&state.source, self.database.into()));
        RemoveAll {
            database: self.database,
            index: self.index,
            states: self.states,
            filter: (self.filter, filter),
            _marker: PhantomData,
        }
    }
}

impl<'d, T: Template, F: Filter, L: Listen> RemoveAll<'d, T, F, L> {
    pub fn resolve(&mut self) -> usize {
        while let Ok(table) = self.database.tables().get(self.index) {
            self.index += 1;
            match ShareTable::<T>::from(
                table.index(),
                self.database.tables(),
                self.database.resources(),
            ) {
                Ok((source, target)) if self.filter.filter(table, self.database.into()) => {
                    self.states.push(StateAll { source, target })
                }
                _ => {}
            }
        }

        fold_swap(
            &mut self.states,
            0,
            (),
            |sum, _, state| {
                debug_assert_ne!(state.source.index(), state.target.index());
                let source = state.source.keys.try_write().ok_or(sum)?;
                let target = state.target.keys.try_upgradable_read().ok_or(sum)?;
                Ok(sum + Self::resolve_tables(source, target, state, self.database))
            },
            |sum, _, state| {
                debug_assert_ne!(state.source.index(), state.target.index());
                if state.source.index() < state.target.index() {
                    let source = state.source.keys.write();
                    let target = state.target.keys.upgradable_read();
                    sum + Self::resolve_tables(source, target, state, self.database)
                } else {
                    let target = state.target.keys.upgradable_read();
                    let source = state.source.keys.write();
                    sum + Self::resolve_tables(source, target, state, self.database)
                }
            },
        )
    }

    fn resolve_tables(
        mut source: RwLockWriteGuard<Vec<Key>>,
        target: RwLockUpgradableReadGuard<Vec<Key>>,
        state: &StateAll,
        database: &Database<impl Listen>,
    ) -> usize {
        let count = state.source.count.swap(0, Ordering::AcqRel);
        let Some(count) = NonZeroUsize::new(count) else {
            return 0;
        };
        let (start, target) = state.target.reserve(target, count);
        let target_keys = unsafe { &mut *RwLockUpgradableReadGuard::rwlock(&target).data_ptr() };
        target_keys[start..start + count.get()].copy_from_slice(&source[..count.get()]);
        resolve_copy_move(
            (&mut source, &state.source),
            &state.target,
            &[(0, start, count)],
            &[],
            &database.inner,
        );
        state
            .target
            .count
            .fetch_add(count.get() as _, Ordering::Release);
        // Slots must be updated after the table `fetch_add` to prevent a `query::find` to be able to observe a row which
        // has an index greater than the `table.count()`. As long as the slots remain in the source table, all accesses
        // to these keys will block at the table access and will correct their table index after they acquire the source
        // table lock.
        database
            .keys()
            .initialize(&target, state.target.index(), start..start + count.get());
        drop(source);
        // Although `source` has been dropped, coherence with be maintained since the `target` lock prevent the keys
        // moving again before `on_remove` is done.
        database.listen.on_remove(
            &target[start..start + count.get()],
            &state.source,
            &state.target,
        );
        count.get()
    }
}

impl<T: Template> ShareTable<T> {
    pub fn from(
        table: u32,
        tables: &Tables,
        resources: &Resources,
    ) -> Result<(Arc<Table>, Arc<Table>), Error> {
        let share = resources.try_global_with(table, || {
            let metas = ShareMeta::<T>::from(resources)?;
            let source = tables.get_shared(table as usize)?;
            let target = {
                let mut targets = Vec::new();
                for meta in source.metas() {
                    match metas.binary_search_by_key(&meta.identifier(), |meta| meta.identifier()) {
                        Ok(_) => {}
                        Err(_) => targets.push(meta),
                    }
                }
                tables.find_or_add(&targets)
            };
            if source.index() == target.index() {
                return Err(Error::TablesMustDiffer(source.index() as _));
            }

            let mut copy = Vec::new();
            for target in target.metas().enumerate() {
                let source = source.column_with(target.1.identifier())?;
                debug_assert_eq!(source.1.meta().identifier(), target.1.identifier());
                if source.1.meta().size > 0 {
                    copy.push((source.0, target.0));
                }
            }

            let mut drop = Vec::new();
            for meta in metas.iter() {
                if let Ok((index, column)) = source.column_with(meta.identifier()) {
                    if column.meta().drop.0() {
                        drop.push(index);
                    }
                }
            }

            Ok(Self {
                source,
                target,
                _marker: PhantomData,
            })
        })?;
        let share = share.read();
        Ok((share.source.clone(), share.target.clone()))
    }
}

fn move_to<'d, 'a>(
    database: &Database<impl Listen>,
    set: &HashSet<Key>,
    moves: &mut Vec<(usize, usize, NonZeroUsize)>,
    copies: &mut Vec<(usize, usize, NonZeroUsize)>,
    (source_table, source_keys): (&Table, RwLockUpgradableReadGuard<'a, Vec<Key>>),
    (target_table, target_keys): (&Table, RwLockUpgradableReadGuard<'a, Vec<Key>>),
    (low, high, count): (u32, u32, NonZeroUsize),
    rows: &mut Rows<'d>,
) {
    let (start, target_keys) = target_table.reserve(target_keys, count);
    // Move data from source to target.
    let range = low..high + 1;
    let head = source_table.count.load(Ordering::Acquire) - count.get();
    let (low, high) = (range.start as usize, range.end as usize);

    if range.len() == count.get() {
        // Fast path. The move range is contiguous. Copy everything from source to target at once.
        copies.push((low, start, count));

        let over = high.saturating_sub(head);
        if let Some(end) = NonZeroUsize::new(count.get() - over) {
            moves.push((head + over, low, end));
        }
    } else {
        // Range is not contiguous; use the slow path.
        let mut cursor = head;
        for (i, &(.., row)) in rows.iter().enumerate() {
            let row = row as usize;
            copies.push((row, start + i, ONE));

            if row < head {
                // Find the next valid row to move.
                while set.contains(unsafe { get_unchecked(&source_keys, cursor) }) {
                    cursor += 1;
                }
                debug_assert!(cursor < head + count.get());
                moves.push((cursor, row, ONE));
                cursor += 1;
            }
        }
    }

    // Target keys can be copied over without requiring the `source_keys` write lock since the range `start..start + count` is
    // reserved to this operation.
    {
        let target_keys =
            unsafe { &mut *RwLockUpgradableReadGuard::rwlock(&target_keys).data_ptr() };
        for &(source, target, count) in copies.iter() {
            debug_assert!(target >= start);
            target_keys[target..target + count.get()]
                .copy_from_slice(&source_keys[source..source + count.get()]);
        }
    }

    let mut source_keys = RwLockUpgradableReadGuard::upgrade(source_keys);
    resolve_copy_move(
        (&mut source_keys, source_table),
        target_table,
        copies,
        moves,
        &database.inner,
    );
    source_table.count.fetch_sub(count.get(), Ordering::Release);
    target_table.count.fetch_add(count.get(), Ordering::Release);
    // Slots must be updated after the table `fetch_add` to prevent a `query::find` to be able to observe a row which
    // has an index greater than the `table.count()`. As long as the slots remain in the source table, all accesses
    // to these keys will block at the table access and will correct their table index after they acquire the source
    // table lock.
    for (i, (key, slot, _)) in rows.drain(..).enumerate() {
        slot.initialize(key.generation(), target_table.index(), (start + i) as u32);
    }

    drop(source_keys);
    // Although `source_keys` has been dropped, coherence will be maintained since the `target` lock prevents the keys from
    // moving again before `emit` is done.
    database.listen.on_remove(
        &target_keys[start..start + count.get()],
        source_table,
        target_table,
    );
    drop(target_keys);
    copies.clear();
    moves.clear();
}

/// SAFETY: A write lock over `source_table.keys` must be held when calling this.
/// - Since a write lock is held over the `source_keys`, it is guaranteed that there is no reader/writer in
/// the columns, so there is no need to take column locks.
fn resolve_copy_move(
    (source_keys, source_table): (&mut [Key], &Table),
    target_table: &Table,
    copies: &[(usize, usize, NonZeroUsize)],
    moves: &[(usize, usize, NonZeroUsize)],
    database: &crate::Inner,
) {
    for &(source, target, count) in moves.iter() {
        source_keys.copy_within(source..source + count.get(), target);
        database
            .keys
            .update(&source_keys, target..target + count.get());
    }

    let mut index = 0;
    for source_column in source_table.columns() {
        let copy = source_column.meta().size > 0;
        // SAFETY: Since a write lock is held over the `source_keys`, it is guaranteed that there is no reader/writer in
        // this `source_column`, so there is no need to take a column lock for its operations.
        match target_table.columns().get(index) {
            Some(target_column)
                if source_column.meta().identifier() == target_column.meta().identifier() =>
            {
                index += 1;
                if copy {
                    for &(source, target, count) in copies {
                        unsafe {
                            Column::copy_to(
                                (&source_column, source),
                                (&target_column, target),
                                count,
                            )
                        };
                    }
                }
            }
            _ if source_column.meta().drop.0() => {
                for &(index, _, count) in copies {
                    unsafe { source_column.drop(index, count) }
                }
            }
            _ => {}
        }
        if copy {
            for &(source, target, count) in moves {
                unsafe { source_column.copy(source, target, count) };
            }
        }
    }
    debug_assert_eq!(index, target_table.columns().len());
}
