use crate::{
    bits::Bits,
    database::Database,
    key::Key,
    resources::Local,
    table::{Store, Table, TableRead},
    Datum, Error,
};
use parking_lot::{MappedRwLockReadGuard, MappedRwLockWriteGuard};
use std::{
    any::TypeId,
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    marker::PhantomData,
    mem::swap,
    ops::{Deref, DerefMut, Range},
};

pub struct Query<'a, I: Item, F: Filter = ()> {
    database: &'a Database,
    reads: HashSet<TypeId>,
    writes: HashSet<TypeId>,
    index: usize,
    indices: HashMap<u32, u32>,
    states: Vec<(I::State, TableState<'a>)>,
    done: VecDeque<u32>,
    pending: VecDeque<u32>,
    filter: F,
}

pub struct Items<'a, 'b, I: Item> {
    range: Range<u32>,
    guard: Option<Guard<'a, 'b, <I::State as Lock<'a>>::Guard>>,
    guards: Guards<'a, 'b, I>,
}

pub struct Chunks<'a, 'b, I: Item> {
    guard: Option<Guard<'a, 'b, <I::State as Lock<'a>>::Guard>>,
    guards: Guards<'a, 'b, I>,
}

struct Guards<'a, 'b, I: Item> {
    count: usize,
    database: &'a Database,
    done: &'b mut VecDeque<u32>,
    pending: &'b mut VecDeque<u32>,
    states: &'b Vec<(I::State, TableState<'a>)>,
}

pub struct Context<'a> {
    read: &'a mut HashSet<TypeId>,
    write: &'a mut HashSet<TypeId>,
}

pub trait Filter {
    fn filter(&mut self, table: &Table) -> bool;
}

pub struct Not<F: Filter>(F);
pub struct Has<D: Datum>(PhantomData<D>);

pub unsafe trait Item {
    type State: for<'a> Lock<'a> + 'static;
    fn declare(context: Context) -> Result<(), Error>;
    fn initialize(table: &Table) -> Result<Self::State, Error>;
}

pub trait Lock<'a> {
    type Guard;
    type Chunk;
    type Item;

    fn try_lock(&self, keys: &[Key], stores: &[Store]) -> Option<Self::Guard>;
    fn lock(&self, keys: &[Key], stores: &[Store]) -> Self::Guard;
    unsafe fn chunk(guard: &mut Self::Guard) -> Self::Chunk;
    unsafe fn chunk_unlocked(&self, keys: &[Key], stores: &[Store]) -> Self::Chunk;
    unsafe fn item(guard: &mut Self::Guard, index: usize) -> Self::Item;
    unsafe fn item_unlocked(&self, keys: &[Key], stores: &[Store], index: usize) -> Self::Item;
}

pub struct Read<T>(usize, PhantomData<T>);
pub struct Write<T>(usize, PhantomData<T>);
pub struct Guard<'a, 'b, T>(T, TableGuard<'a, 'b>);
pub enum UnsafeGuard<'a, 'b, I: Item> {
    Safe(Guard<'a, 'b, <I::State as Lock<'a>>::Item>),
    Unsafe(<I::State as Lock<'a>>::Item),
}
enum GuardError<'a, 'b, I: Item> {
    Invalid,
    WouldBlock,
    WouldDeadlock(&'b I::State, TableRead<'a>),
}

struct TableGuard<'a, 'b> {
    table_state: &'b TableState<'a>,
    table_read: TableRead<'a>,
}

#[derive(Default)]
struct TableLocks {
    readers: usize,
    locks: Bits,
    conflicts: Vec<(HashSet<TypeId>, HashSet<TypeId>, Bits)>,
    free: BinaryHeap<usize>,
}

struct TableState<'a> {
    table: &'a Table,
    locks_index: usize,
    table_locks: Local<TableLocks>,
}

impl Context<'_> {
    pub fn own(&mut self) -> Context {
        Context {
            read: self.read,
            write: self.write,
        }
    }

    pub fn read<T: 'static>(&mut self) -> Result<(), Error> {
        let identifier = TypeId::of::<T>();
        if self.write.contains(&identifier) {
            Err(Error::ReadWriteConflict)
        } else {
            self.read.insert(identifier);
            Ok(())
        }
    }

    pub fn write<T: 'static>(&mut self) -> Result<(), Error> {
        let identifier = TypeId::of::<T>();
        if self.read.contains(&identifier) {
            Err(Error::ReadWriteConflict)
        } else if self.write.insert(identifier) {
            Ok(())
        } else {
            Err(Error::WriteWriteConflict)
        }
    }
}

impl<T> Deref for Guard<'_, '_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Guard<'_, '_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, I: Item> Deref for UnsafeGuard<'a, '_, I> {
    type Target = <I::State as Lock<'a>>::Item;

    fn deref(&self) -> &Self::Target {
        match self {
            UnsafeGuard::Safe(item) => item,
            UnsafeGuard::Unsafe(item) => item,
        }
    }
}

impl<'a, I: Item> DerefMut for UnsafeGuard<'a, '_, I> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            UnsafeGuard::Safe(item) => item,
            UnsafeGuard::Unsafe(item) => item,
        }
    }
}

impl<'a, I: Item, F: Filter> Query<'a, I, F> {
    #[inline]
    pub fn item(&mut self, key: Key) -> Result<Guard<<I::State as Lock<'a>>::Item>, Error> {
        self.with(key, Ok, |_, _| None)?
    }

    /// SAFETY: The unsafety only applies if an `Ok(UnsafeGuard::Unsafe)` if returned by this method. If that is the case, it means
    /// that the calling thread already holds that a lock conflicted with the retrieval of this `Item`. Thus, the safety requirements
    /// are as follows:
    /// - The caller must garantee that no mutable reference is aliased. This will be true if already retrieved items have types that don't
    /// conflict (_ref-mut, mut-ref or mut-mut_) with the types of this `Item` AND/OR already retrieved items don't have the same key
    /// as this `Item`.
    /// - The conflicting lock (most likely held by another query `Guard` or `Iterator`) must be held for at least as long as the
    /// `UnsafeGuard` is alive.
    #[inline]
    pub unsafe fn item_unchecked<'b>(
        &'b mut self,
        key: Key,
    ) -> Result<UnsafeGuard<'a, 'b, I>, Error> {
        self.with(key, UnsafeGuard::Safe, |state, table| {
            Some(UnsafeGuard::Unsafe(state.item_unlocked(
                table.keys(),
                table.stores(),
                0,
            )))
        })
    }

    #[inline]
    pub fn item_with<T>(
        &mut self,
        key: Key,
        with: impl FnOnce(<I::State as Lock>::Item) -> T,
    ) -> Result<T, Error> {
        self.with(key, |guard| Ok(with(guard.0)), |_, _| None)?
    }

    #[inline]
    pub fn items(&mut self) -> Items<I> {
        Items {
            range: 0..0,
            guard: None,
            guards: self.guards(),
        }
    }

    #[inline]
    pub fn items_with(&mut self, mut each: impl FnMut(<I::State as Lock>::Item)) {
        for mut guard in self.guards() {
            for i in 0..guard.1.table_read.count() {
                each(unsafe { I::State::item(&mut guard, i as usize) });
            }
            drop(guard);
        }
    }

    #[inline]
    pub fn chunks(&mut self) -> Chunks<I> {
        Chunks {
            guard: None,
            guards: self.guards(),
        }
    }

    #[inline]
    pub fn chunks_with(&mut self, mut each: impl FnMut(<I::State as Lock>::Chunk)) {
        for mut guard in self.guards() {
            each(unsafe { I::State::chunk(&mut guard) });
            drop(guard);
        }
    }

    /// Ensure that all tables have been filtered or initialized.
    fn update(&mut self) -> Result<(), Error> {
        while let Some(table) = self.database.tables().get(self.index) {
            if self.filter.filter(&table) {
                if let Ok(state) = I::initialize(&table) {
                    let table_locks = self
                        .database
                        .resources()
                        .local_with(table.index(), || Ok(TableLocks::default()))?;
                    let locks_index = {
                        let mut locks = table_locks.borrow_mut();
                        let new_index = match locks.free.pop() {
                            Some(index) => {
                                let conflicts = &mut locks.conflicts[index];
                                conflicts.0.clear();
                                conflicts.0.extend(&self.reads);
                                conflicts.1.clear();
                                conflicts.1.extend(&self.writes);
                                index
                            }
                            None => {
                                let index = locks.conflicts.len();
                                locks.conflicts.push((
                                    self.reads.clone(),
                                    self.writes.clone(),
                                    Bits::new(),
                                ));
                                index
                            }
                        };

                        for old_index in 0..locks.conflicts.len() {
                            let new_conflicts = &locks.conflicts[new_index];
                            let old_conflicts = &locks.conflicts[old_index];
                            let allow = new_conflicts.0.is_disjoint(&old_conflicts.1)
                                && new_conflicts.1.is_disjoint(&old_conflicts.0)
                                && new_conflicts.1.is_disjoint(&old_conflicts.1);
                            // Set the conflict even if allow is `true` since indices are reused.
                            locks.conflicts[old_index].2.set(new_index, !allow);
                            locks.conflicts[new_index].2.set(old_index, !allow);
                        }
                        new_index
                    };

                    let index = self.states.len() as _;
                    self.done.push_back(index);
                    self.indices.insert(table.index(), index);
                    self.states.push((
                        state,
                        TableState {
                            table,
                            locks_index,
                            table_locks,
                        },
                    ));
                }
            }
            self.index += 1;
        }

        Ok(())
    }

    fn with<'b, T>(
        &'b mut self,
        key: Key,
        with: impl FnOnce(Guard<'a, 'b, <I::State as Lock<'a>>::Item>) -> T,
        deadlock: impl FnOnce(&'b I::State, TableRead<'a>) -> Option<T>,
    ) -> Result<T, Error> {
        self.update()?;
        loop {
            let slot = self.database.keys().get(key)?;
            let (table_index, store_index) = slot.indices();
            let &state_index = self
                .indices
                .get(&table_index)
                .ok_or(Error::KeyNotInQuery(key))?;
            match Self::guard(self.database, &self.states, state_index, true, || {
                // If this is not the case, it means that the `key` has just been moved.
                slot.indices() == (table_index, store_index)
            }) {
                Ok(mut guard) => {
                    let item = unsafe { I::State::item(&mut guard.0, store_index as _) };
                    break Ok(with(Guard(item, guard.1)));
                }
                Err(GuardError::Invalid | GuardError::WouldBlock) => continue,
                Err(GuardError::WouldDeadlock(state, table)) => {
                    break deadlock(state, table).ok_or(Error::WouldDeadlock)
                }
            }
        }
    }

    #[inline]
    fn guards(&mut self) -> Guards<'a, '_, I> {
        self.update().unwrap();
        swap(&mut self.done, &mut self.pending);
        Guards {
            count: self.pending.len(),
            database: self.database,
            done: &mut self.done,
            pending: &mut self.pending,
            states: &self.states,
        }
    }

    fn guard<'b>(
        database: &'a Database,
        states: &'b Vec<(I::State, TableState<'a>)>,
        index: u32,
        block: bool,
        mut valid: impl FnMut() -> bool,
    ) -> Result<Guard<'a, 'b, <I::State as Lock<'a>>::Guard>, GuardError<'a, 'b, I>> {
        debug_assert!((index as usize) < states.len());
        let state = unsafe { states.get_unchecked(index as usize) };
        let table_read = if block {
            database.table_read(state.1.table)
        } else {
            database
                .table_try_read(state.1.table)
                .ok_or(GuardError::WouldBlock)?
        };
        if !valid() {
            return Err(GuardError::Invalid);
        }

        match state.0.try_lock(table_read.keys(), table_read.stores()) {
            Some(guard) => Ok(Guard(guard, TableGuard::new(&state.1, table_read))),
            None if block && state.1.table_locks.borrow().allows(state.1.locks_index) => {
                let guard = state.0.lock(table_read.keys(), table_read.stores());
                Ok(Guard(guard, TableGuard::new(&state.1, table_read)))
            }
            // Resolve to taking a write lock over the table only if no other strategy works.
            None if block && state.1.table_locks.borrow().readers == 0 => {
                drop(table_read);

                // Since the read lock failed to make progress, escalate to a write lock.
                let table_write = database.table_write(state.1.table);
                if !valid() {
                    return Err(GuardError::Invalid);
                }
                let guard = state.0.lock(table_write.keys(), table_write.stores());
                let table_read = table_write.downgrade();
                Ok(Guard(guard, TableGuard::new(&state.1, table_read)))
            }
            None if block => Err(GuardError::WouldDeadlock(&state.0, table_read)),
            None => Err(GuardError::WouldBlock),
        }
    }
}

impl<'a, 'b, I: Item> Iterator for Guards<'a, 'b, I> {
    type Item = Guard<'a, 'b, <I::State as Lock<'a>>::Guard>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(state_index) = self.pending.pop_front() {
            let block = self.count == 0;
            match Query::<I, ()>::guard(self.database, &self.states, state_index, block, || true) {
                Ok(guard) => {
                    self.done.push_back(state_index);
                    self.count = self.pending.len();
                    return Some(guard);
                }
                Err(GuardError::WouldBlock) => {
                    self.count = self.count.saturating_sub(1);
                    self.pending.push_back(state_index);
                }
                Err(GuardError::WouldDeadlock(_, _)) => return None,
                Err(GuardError::Invalid) => unreachable!(),
            }
        }

        None
    }
}

impl<'a, I: Item> ExactSizeIterator for Guards<'a, '_, I> {
    #[inline]
    fn len(&self) -> usize {
        self.pending.len()
    }
}

impl<I: Item> Drop for Guards<'_, '_, I> {
    #[inline]
    fn drop(&mut self) {
        self.done.extend(self.pending.drain(..));
    }
}

impl<'a, I: Item> Iterator for Items<'a, '_, I> {
    type Item = <I::State as Lock<'a>>::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(index) = self.range.next() {
            let guard = unsafe { self.guard.as_mut().unwrap_unchecked() };
            let item = unsafe { I::State::item(guard, index as _) };
            return Some(item);
        }

        while let Some(mut guard) = self.guards.next() {
            let count = guard.1.table_read.count();
            if count == 0 {
                continue;
            }

            let item = unsafe { I::State::item(&mut guard, 0) };
            self.range = 1..count;
            self.guard = Some(guard);
            return Some(item);
        }

        None
    }
}

impl<'a, I: Item> Iterator for Chunks<'a, '_, I> {
    type Item = <I::State as Lock<'a>>::Chunk;

    fn next(&mut self) -> Option<Self::Item> {
        self.guard = self.guards.next();
        Some(unsafe { I::State::chunk(self.guard.as_mut()?) })
    }
}

impl<I: Item> ExactSizeIterator for Chunks<'_, '_, I> {
    #[inline]
    fn len(&self) -> usize {
        self.guards.len()
    }
}

unsafe impl<D: Datum> Item for &D {
    type State = Read<D>;

    fn declare(mut context: Context) -> Result<(), Error> {
        context.read::<D>()
    }

    fn initialize(table: &Table) -> Result<Self::State, Error> {
        Ok(Read(table.store::<D>()?, PhantomData))
    }
}

impl<'a, D: Datum> Lock<'a> for Read<D> {
    type Guard = MappedRwLockReadGuard<'a, [D]>;
    type Chunk = &'a [D];
    type Item = &'a D;

    #[inline]
    fn try_lock(&self, keys: &[Key], stores: &[Store]) -> Option<Self::Guard> {
        debug_assert!(self.0 < stores.len());
        let store = unsafe { &*stores.as_ptr().add(self.0) };
        unsafe { store.try_read(.., keys.len()) }
    }

    #[inline]
    fn lock(&self, keys: &[Key], stores: &[Store]) -> Self::Guard {
        debug_assert!(self.0 < stores.len());
        let store = unsafe { &*stores.as_ptr().add(self.0) };
        unsafe { store.read(.., keys.len()) }
    }

    #[inline]
    unsafe fn chunk(guard: &mut Self::Guard) -> Self::Chunk {
        // TODO: Fix this...
        &*(guard as *mut Self::Guard)
    }

    #[inline]
    unsafe fn chunk_unlocked(&self, keys: &[Key], stores: &[Store]) -> Self::Chunk {
        let store = &*stores.as_ptr().add(self.0);
        store.get_unlocked(.., keys.len())
    }

    #[inline]
    unsafe fn item(guard: &mut Self::Guard, index: usize) -> Self::Item {
        Self::chunk(guard).get_unchecked(index)
    }

    #[inline]
    unsafe fn item_unlocked(&self, _: &[Key], stores: &[Store], index: usize) -> Self::Item {
        let store = &*stores.as_ptr().add(self.0);
        store.get_unlocked_at(index)
    }
}

unsafe impl<D: Datum> Item for &mut D {
    type State = Write<D>;

    fn declare(mut context: Context) -> Result<(), Error> {
        context.write::<D>()
    }

    fn initialize(table: &Table) -> Result<Self::State, Error> {
        Ok(Write(table.store::<D>()?, PhantomData))
    }
}

impl<'a, D: Datum> Lock<'a> for Write<D> {
    type Guard = MappedRwLockWriteGuard<'a, [D]>;
    type Chunk = &'a mut [D];
    type Item = &'a mut D;

    #[inline]
    fn try_lock(&self, keys: &[Key], stores: &[Store]) -> Option<Self::Guard> {
        debug_assert!(self.0 < stores.len());
        let store = unsafe { &*stores.as_ptr().add(self.0) };
        unsafe { store.try_write(.., keys.len()) }
    }

    #[inline]
    fn lock(&self, keys: &[Key], stores: &[Store]) -> Self::Guard {
        debug_assert!(self.0 < stores.len());
        let store = unsafe { &*stores.as_ptr().add(self.0) };
        unsafe { store.write(.., keys.len()) }
    }

    #[inline]
    unsafe fn chunk(guard: &mut Self::Guard) -> Self::Chunk {
        &mut *(guard as *mut Self::Guard)
    }

    #[inline]
    unsafe fn chunk_unlocked(&self, keys: &[Key], stores: &[Store]) -> Self::Chunk {
        let store = &*stores.as_ptr().add(self.0);
        store.get_unlocked(.., keys.len())
    }

    #[inline]
    unsafe fn item(guard: &mut Self::Guard, index: usize) -> Self::Item {
        Self::chunk(guard).get_unchecked_mut(index)
    }

    #[inline]
    unsafe fn item_unlocked(&self, _: &[Key], stores: &[Store], index: usize) -> Self::Item {
        let store = &*stores.as_ptr().add(self.0);
        store.get_unlocked_at(index)
    }
}

unsafe impl Item for () {
    type State = ();

    fn declare(_: Context) -> Result<(), Error> {
        Ok(())
    }

    fn initialize(_: &Table) -> Result<Self::State, Error> {
        Ok(())
    }
}

impl<'a> Lock<'a> for () {
    type Guard = ();
    type Chunk = ();
    type Item = ();

    #[inline]
    fn try_lock(&self, _: &[Key], _: &[Store]) -> Option<Self::Guard> {
        Some(())
    }
    #[inline]
    fn lock(&self, _: &[Key], _: &[Store]) -> Self::Guard {}
    #[inline]
    unsafe fn chunk(_: &mut Self::Guard) -> Self::Chunk {}
    #[inline]
    unsafe fn chunk_unlocked(&self, _: &[Key], _: &[Store]) -> Self::Chunk {}
    #[inline]
    unsafe fn item(_: &mut Self::Guard, _: usize) -> Self::Item {}
    #[inline]
    unsafe fn item_unlocked(&self, _: &[Key], _: &[Store], _: usize) -> Self::Item {}
}

unsafe impl<I1: Item, I2: Item> Item for (I1, I2) {
    type State = (I1::State, I2::State);

    fn declare(mut context: Context) -> Result<(), Error> {
        I1::declare(context.own())?;
        I2::declare(context.own())?;
        Ok(())
    }

    fn initialize(table: &Table) -> Result<Self::State, Error> {
        Ok((I1::initialize(table)?, I2::initialize(table)?))
    }
}

unsafe impl<I1: Item, I2: Item, I3: Item> Item for (I1, I2, I3) {
    type State = (I1::State, I2::State, I3::State);

    fn declare(mut context: Context) -> Result<(), Error> {
        I1::declare(context.own())?;
        I2::declare(context.own())?;
        I3::declare(context.own())?;
        Ok(())
    }

    fn initialize(table: &Table) -> Result<Self::State, Error> {
        Ok((
            I1::initialize(table)?,
            I2::initialize(table)?,
            I3::initialize(table)?,
        ))
    }
}

impl<'a, L1: Lock<'a>, L2: Lock<'a>> Lock<'a> for (L1, L2) {
    type Guard = (L1::Guard, L2::Guard);
    type Chunk = (L1::Chunk, L2::Chunk);
    type Item = (L1::Item, L2::Item);

    #[inline]
    fn try_lock(&self, keys: &[Key], stores: &[Store]) -> Option<Self::Guard> {
        Some((
            self.0.try_lock(keys, stores)?,
            self.1.try_lock(keys, stores)?,
        ))
    }

    #[inline]
    fn lock(&self, keys: &[Key], stores: &[Store]) -> Self::Guard {
        (self.0.lock(keys, stores), self.1.lock(keys, stores))
    }

    #[inline]
    unsafe fn chunk(guard: &mut Self::Guard) -> Self::Chunk {
        (L1::chunk(&mut guard.0), L2::chunk(&mut guard.1))
    }

    #[inline]
    unsafe fn chunk_unlocked(&self, keys: &[Key], stores: &[Store]) -> Self::Chunk {
        (
            self.0.chunk_unlocked(keys, stores),
            self.1.chunk_unlocked(keys, stores),
        )
    }

    #[inline]
    unsafe fn item(guard: &mut Self::Guard, index: usize) -> Self::Item {
        (L1::item(&mut guard.0, index), L2::item(&mut guard.1, index))
    }

    #[inline]
    unsafe fn item_unlocked(&self, keys: &[Key], stores: &[Store], index: usize) -> Self::Item {
        (
            self.0.item_unlocked(keys, stores, index),
            self.1.item_unlocked(keys, stores, index),
        )
    }
}

impl<'a, L1: Lock<'a>, L2: Lock<'a>, L3: Lock<'a>> Lock<'a> for (L1, L2, L3) {
    type Guard = (L1::Guard, L2::Guard, L3::Guard);
    type Chunk = (L1::Chunk, L2::Chunk, L3::Chunk);
    type Item = (L1::Item, L2::Item, L3::Item);

    #[inline]
    fn try_lock(&self, keys: &[Key], stores: &[Store]) -> Option<Self::Guard> {
        Some((
            self.0.try_lock(keys, stores)?,
            self.1.try_lock(keys, stores)?,
            self.2.try_lock(keys, stores)?,
        ))
    }

    #[inline]
    fn lock(&self, keys: &[Key], stores: &[Store]) -> Self::Guard {
        (
            self.0.lock(keys, stores),
            self.1.lock(keys, stores),
            self.2.lock(keys, stores),
        )
    }

    #[inline]
    unsafe fn chunk(guard: &mut Self::Guard) -> Self::Chunk {
        (
            L1::chunk(&mut guard.0),
            L2::chunk(&mut guard.1),
            L3::chunk(&mut guard.2),
        )
    }

    #[inline]
    unsafe fn chunk_unlocked(&self, keys: &[Key], stores: &[Store]) -> Self::Chunk {
        (
            self.0.chunk_unlocked(keys, stores),
            self.1.chunk_unlocked(keys, stores),
            self.2.chunk_unlocked(keys, stores),
        )
    }

    #[inline]
    unsafe fn item(guard: &mut Self::Guard, index: usize) -> Self::Item {
        (
            L1::item(&mut guard.0, index),
            L2::item(&mut guard.1, index),
            L3::item(&mut guard.2, index),
        )
    }

    #[inline]
    unsafe fn item_unlocked(&self, keys: &[Key], stores: &[Store], index: usize) -> Self::Item {
        (
            self.0.item_unlocked(keys, stores, index),
            self.1.item_unlocked(keys, stores, index),
            self.2.item_unlocked(keys, stores, index),
        )
    }
}

impl Filter for () {
    fn filter(&mut self, table: &Table) -> bool {
        true.filter(table)
    }
}

impl Filter for bool {
    fn filter(&mut self, _: &Table) -> bool {
        *self
    }
}

impl<F: Filter> Filter for Not<F> {
    fn filter(&mut self, table: &Table) -> bool {
        !self.0.filter(table)
    }
}

impl<D: Datum> Filter for Has<D> {
    fn filter(&mut self, table: &Table) -> bool {
        table.has(TypeId::of::<D>())
    }
}

impl<'a, 'b> TableGuard<'a, 'b> {
    #[inline]
    pub fn new(table_state: &'b TableState<'a>, table_read: TableRead<'a>) -> Self {
        let mut locks = table_state.table_locks.borrow_mut();
        debug_assert!(!locks.locks.has(table_state.locks_index as _));
        locks.readers += 1;
        locks.locks.set(table_state.locks_index as _, true);
        Self {
            table_state,
            table_read,
        }
    }
}

impl Drop for TableGuard<'_, '_> {
    #[inline]
    fn drop(&mut self) {
        let mut locks = self.table_state.table_locks.borrow_mut();
        debug_assert!(locks.readers > 0);
        debug_assert!(locks.locks.has(self.table_state.locks_index));
        locks.readers -= 1;
        locks.locks.set(self.table_state.locks_index, false);
    }
}

impl TableLocks {
    pub fn allows(&self, index: usize) -> bool {
        match self.conflicts.get(index) {
            Some((_, _, conflicts)) => self.locks.has_none(conflicts),
            None => false,
        }
    }
}

impl Drop for TableState<'_> {
    fn drop(&mut self) {
        let mut locks = self.table_locks.borrow_mut();
        if self.locks_index == locks.conflicts.len() - 1 {
            locks.conflicts.pop();
            while let Some(&index) = locks.free.peek() {
                if index == locks.conflicts.len() - 1 {
                    locks.conflicts.pop();
                    locks.free.pop();
                } else {
                    break;
                }
            }
        } else {
            locks.free.push(self.locks_index);
        }
    }
}

impl Database {
    pub fn query<I: Item>(&self) -> Result<Query<I>, Error> {
        self.query_with(())
    }

    pub fn query_with<I: Item, F: Filter>(&self, filter: F) -> Result<Query<I, F>, Error> {
        let mut reads = HashSet::new();
        let mut writes = HashSet::new();
        I::declare(Context {
            read: &mut reads,
            write: &mut writes,
        })?;
        Ok(Query {
            database: self,
            reads,
            writes,
            states: Vec::new(),
            index: 0,
            indices: HashMap::new(),
            done: VecDeque::new(),
            pending: VecDeque::new(),
            filter,
        })
    }
}
