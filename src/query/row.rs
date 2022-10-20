use super::*;
use crate::{
    key::Key,
    table::{self, Table},
    Datum, Error,
};
use std::{
    any::TypeId,
    collections::{HashMap, HashSet, VecDeque},
    marker::PhantomData,
    mem::swap,
    ptr::NonNull,
    slice::{from_raw_parts, from_raw_parts_mut},
};

pub struct Read<D>(usize, PhantomData<D>);
pub struct Write<D>(usize, PhantomData<D>);
pub struct DeclareContext<'a>(&'a mut HashSet<Access>);
pub struct InitializeContext<'a>(&'a HashMap<Access, usize>);
pub struct ItemContext<'a, 'b>(
    pub(crate) &'a [Key],
    pub(crate) &'b [NonNull<()>],
    pub(crate) usize,
);
pub struct ChunkContext<'a, 'b>(&'a [Key], &'b [NonNull<()>]);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Access {
    Read(TypeId),
    Write(TypeId),
}

pub unsafe trait Row {
    type State;
    type Read: Row;
    type Item<'a>;
    type Chunk<'a>;

    fn declare(context: DeclareContext) -> Result<(), Error>;
    fn initialize(context: InitializeContext) -> Result<Self::State, Error>;
    fn read(state: Self::State) -> <Self::Read as Row>::State;
    fn item<'a>(state: &Self::State, context: ItemContext<'a, '_>) -> Self::Item<'a>;
    fn chunk<'a>(state: &Self::State, context: ChunkContext<'a, '_>) -> Self::Chunk<'a>;
}

pub struct Rows<'d, R: Row> {
    pub(crate) indices: HashMap<u32, u32>, // From table index to state index.
    pub(crate) states: Vec<(R::State, &'d Table, Vec<(usize, Access)>)>,
    pub(crate) done: VecDeque<u32>,
    pub(crate) pending: VecDeque<u32>,
    accesses: HashSet<Access>,
    pointers: Vec<NonNull<()>>,
    _marker: PhantomData<fn(R)>,
}

impl<'d, R: Row> Rows<'d, R> {
    pub fn new() -> Result<Self, Error> {
        // Detects violations of rust's invariants.
        let mut accesses = HashSet::new();
        R::declare(DeclareContext(&mut accesses))?;
        Ok(Self {
            indices: HashMap::new(),
            states: Vec::new(),
            done: VecDeque::new(),
            pending: VecDeque::new(),
            accesses,
            pointers: Vec::new(),
            _marker: PhantomData,
        })
    }

    #[inline]
    pub(crate) fn try_guards<
        S,
        F: FnMut(S, u32, &R::State, &[NonNull<()>], &Table, &table::Inner) -> Result<S, S>,
    >(
        &mut self,
        state: S,
        mut fold: F,
    ) -> S {
        let mut fold = |mut state: S| -> Result<S, S> {
            for _ in 0..self.pending.len() {
                let index = unsafe { self.pending.pop_front().unwrap_unchecked() };
                state = match self.try_lock(state, index, |state, row, pointers, table, inner| {
                    fold(state, index, row, pointers, table, inner)
                }) {
                    Ok(result) => {
                        self.done.push_back(index);
                        result?
                    }
                    Err(state) => {
                        self.pending.push_back(index);
                        state
                    }
                };
            }

            while let Some(state_index) = self.pending.pop_front() {
                self.done.push_back(state_index);
                state = self.lock(state_index, |row, pointers, table, inner| {
                    fold(state, state_index, row, pointers, table, inner)
                })?;
            }
            Ok(state)
        };

        match fold(state) {
            Ok(state) => {
                swap(&mut self.done, &mut self.pending);
                state
            }
            Err(state) => {
                // Fold was interrupted, so move remaining indices in `pending` while preserving the order of the indices in `done`.
                if self.done.len() < self.pending.len() {
                    while let Some(index) = self.done.pop_back() {
                        self.pending.push_front(index);
                    }
                } else {
                    while let Some(index) = self.pending.pop_front() {
                        self.done.push_back(index);
                    }
                    swap(&mut self.done, &mut self.pending);
                }
                state
            }
        }
    }

    pub(crate) fn guards<
        S,
        F: FnMut(S, u32, &R::State, &[NonNull<()>], &Table, &table::Inner) -> S,
    >(
        &mut self,
        mut state: S,
        mut fold: F,
    ) -> S {
        for _ in 0..self.pending.len() {
            let index = unsafe { self.pending.pop_front().unwrap_unchecked() };
            state = match self.try_lock(state, index, |state, row, pointers, table, inner| {
                fold(state, index, row, pointers, table, inner)
            }) {
                Ok(state) => {
                    self.done.push_back(index);
                    state
                }
                Err(state) => {
                    self.pending.push_back(index);
                    state
                }
            };
        }

        while let Some(state_index) = self.pending.pop_front() {
            self.done.push_back(state_index);
            state = self.lock(state_index, |row, pointers, table, inner| {
                fold(state, state_index, row, pointers, table, inner)
            });
        }

        swap(&mut self.done, &mut self.pending);
        state
    }

    fn try_lock<S, T>(
        &mut self,
        state: S,
        index: u32,
        with: impl FnOnce(S, &R::State, &[NonNull<()>], &Table, &table::Inner) -> T,
    ) -> Result<T, S> {
        fn next<S, T>(
            state: S,
            indices: &[(usize, Access)],
            pointers: &mut Vec<NonNull<()>>,
            inner: &table::Inner,
            with: impl FnOnce(S, &[NonNull<()>], &table::Inner) -> T,
        ) -> Result<T, S> {
            match indices.split_first() {
                Some((&(index, access), rest)) => {
                    let store = unsafe { inner.stores.get_unchecked(index) };
                    debug_assert_eq!(access.identifier(), store.meta().identifier());
                    match access {
                        Access::Read(_) => {
                            let guard = match store.data().try_read() {
                                Some(guard) => guard,
                                None => return Err(state),
                            };
                            pointers.push(*guard);
                            let result = next(state, rest, pointers, inner, with);
                            drop(guard);
                            result
                        }
                        Access::Write(_) => {
                            let guard = match store.data().try_write() {
                                Some(guard) => guard,
                                None => return Err(state),
                            };
                            pointers.push(*guard);
                            let result = next(state, rest, pointers, inner, with);
                            drop(guard);
                            result
                        }
                    }
                }
                None => {
                    let value = with(state, pointers, inner);
                    pointers.clear();
                    Ok(value)
                }
            }
        }

        let (row, table, indices) = unsafe { self.states.get_unchecked(index as usize) };
        let inner = match table.inner.try_read() {
            Some(inner) => inner,
            None => return Err(state),
        };
        next(
            state,
            indices,
            &mut self.pointers,
            &inner,
            |state, pointers, inner| with(state, row, pointers, table, inner),
        )
    }

    fn lock<T>(
        &mut self,
        index: u32,
        with: impl FnOnce(&R::State, &[NonNull<()>], &Table, &table::Inner) -> T,
    ) -> T {
        #[inline]
        fn next<T>(
            indices: &[(usize, Access)],
            pointers: &mut Vec<NonNull<()>>,
            inner: &table::Inner,
            with: impl FnOnce(&[NonNull<()>], &table::Inner) -> T,
        ) -> T {
            match indices.split_first() {
                Some((&(index, access), rest)) => {
                    let store = unsafe { inner.stores.get_unchecked(index) };
                    debug_assert_eq!(access.identifier(), store.meta().identifier());
                    match access {
                        Access::Read(_) => {
                            let guard = store.data().read();
                            pointers.push(*guard);
                            let state = next(rest, pointers, inner, with);
                            drop(guard);
                            state
                        }
                        Access::Write(_) => {
                            let guard = store.data().write();
                            pointers.push(*guard);
                            let state = next(rest, pointers, inner, with);
                            drop(guard);
                            state
                        }
                    }
                }
                None => {
                    let state = with(pointers, inner);
                    pointers.clear();
                    state
                }
            }
        }

        let (row, table, indices) = unsafe { self.states.get_unchecked(index as usize) };
        let inner = table.inner.read();
        next(indices, &mut self.pointers, &inner, |pointers, inner| {
            with(row, pointers, table, inner)
        })
    }
}

impl<'d, R: Row> Query<'d> for Rows<'d, R> {
    type Item<'a> = R::Item<'a>;
    type Read = Rows<'d, R::Read>;

    fn initialize(&mut self, table: &'d Table) -> Result<(), Error> {
        let mut indices = Vec::with_capacity(self.accesses.len());
        for &access in self.accesses.iter() {
            indices.push((table.store_with(access.identifier())?, access));
        }

        // The sorting of indices ensures that there cannot be a deadlock between `Rows` when locking multiple stores as long as this
        // happens while holding at most 1 table lock.
        indices.sort_unstable_by_key(|&(index, _)| index);
        let map = indices
            .iter()
            .enumerate()
            .map(|(i, &(_, access))| (access, i))
            .collect();

        let state = R::initialize(InitializeContext(&map))?;
        let index = self.states.len() as _;
        self.pending.push_back(index);
        self.indices.insert(table.index(), index);
        self.states.push((state, table, indices));
        Ok(())
    }

    fn read(self) -> Self::Read {
        Rows {
            indices: self.indices,
            states: self
                .states
                .into_iter()
                .map(|(state, table, mut indices)| {
                    for (_, access) in indices.iter_mut() {
                        *access = access.read();
                    }
                    (R::read(state), table, indices)
                })
                .collect(),
            done: self.done,
            pending: self.pending,
            accesses: self
                .accesses
                .into_iter()
                .map(|access| access.read())
                .collect(),
            pointers: self.pointers,
            _marker: PhantomData,
        }
    }

    #[inline]
    fn has(&mut self, key: Key, context: Context<'d>) -> bool {
        if let Ok(slot) = context.database.keys().get(key) {
            self.indices.get(&slot.table()).is_some()
        } else {
            false
        }
    }

    #[inline]
    fn count(&mut self, _: Context<'d>) -> usize {
        self.states
            .iter()
            .map(|(_, table, _)| table.inner.read().count() as usize)
            .sum()
    }

    fn try_find<T, F: FnOnce(Result<Self::Item<'_>, Error>) -> T>(
        &mut self,
        key: Key,
        context: super::Context<'d>,
        mut find: F,
    ) -> T {
        let keys = context.database.keys();
        loop {
            let slot = match keys.get(key) {
                Ok(slot) => slot,
                Err(error) => break find(Err(error)),
            };
            let (table_index, store_index) = slot.indices();
            let state_index = match self.indices.get(&table_index) {
                Some(&state_index) => state_index,
                None => break find(Err(Error::KeyNotInQuery(key))),
            };

            find = match self.lock(state_index, |row, pointers, _, inner| {
                // If this check fails, it means that the `key` has just been moved or destroyed.
                if slot.indices() == (table_index, store_index) {
                    let context = ItemContext(inner.keys(), pointers, store_index as _);
                    Ok(find(Ok(R::item(row, context))))
                } else {
                    Err(find)
                }
            }) {
                Ok(value) => break value,
                Err(find) => find,
            };
        }
    }

    #[inline]
    fn try_fold<S, F: FnMut(S, Self::Item<'_>) -> Result<S, S>>(
        &mut self,
        _: super::Context<'d>,
        state: S,
        mut fold: F,
    ) -> S {
        self.try_guards(state, |mut state, _, row, pointers, _, inner| {
            let context = ItemContext(inner.keys(), pointers, 0);
            for i in 0..inner.count() {
                state = fold(state, R::item(row, context.with(i as _)))?;
            }
            Ok(state)
        })
    }

    #[inline]
    fn fold<S, F: FnMut(S, Self::Item<'_>) -> S>(
        &mut self,
        _: super::Context<'d>,
        state: S,
        mut fold: F,
    ) -> S {
        self.guards(state, |mut state, _, row, pointers, _, inner| {
            let context = ItemContext(inner.keys(), pointers, 0);
            for i in 0..inner.count() {
                state = fold(state, R::item(row, context.with(i as _)));
            }
            state
        })
    }
}

impl Access {
    #[inline]
    pub fn identifier(&self) -> TypeId {
        match *self {
            Access::Read(identifier) => identifier,
            Access::Write(identifier) => identifier,
        }
    }

    #[inline]
    pub fn read(&self) -> Self {
        Self::Read(self.identifier())
    }
}
impl DeclareContext<'_> {
    pub fn own(&mut self) -> DeclareContext<'_> {
        DeclareContext(self.0)
    }

    pub fn read<D: Datum>(&mut self) -> Result<(), Error> {
        let identifier = TypeId::of::<D>();
        if self.0.contains(&Access::Write(identifier)) {
            Err(Error::ReadWriteConflict)
        } else {
            self.0.insert(Access::Read(identifier));
            Ok(())
        }
    }

    pub fn write<D: Datum>(&mut self) -> Result<(), Error> {
        let identifier = TypeId::of::<D>();
        if self.0.contains(&Access::Read(identifier)) {
            Err(Error::ReadWriteConflict)
        } else if self.0.insert(Access::Write(identifier)) {
            Ok(())
        } else {
            Err(Error::WriteWriteConflict)
        }
    }
}

impl InitializeContext<'_> {
    pub fn own(&self) -> Self {
        Self(self.0)
    }

    pub fn read<D: Datum>(&self) -> Result<Read<D>, Error> {
        let index = self.index(Access::Read(TypeId::of::<D>()))?;
        Ok(Read(index, PhantomData))
    }

    pub fn write<D: Datum>(&self) -> Result<Write<D>, Error> {
        let index = self.index(Access::Write(TypeId::of::<D>()))?;
        Ok(Write(index, PhantomData))
    }

    fn index(&self, access: Access) -> Result<usize, Error> {
        self.0.get(&access).copied().ok_or(Error::MissingStore)
    }
}

impl<D> Write<D> {
    pub fn read(self) -> Read<D> {
        Read(self.0, PhantomData)
    }
}

impl<'a> ItemContext<'a, '_> {
    #[inline]
    pub const fn own(&self) -> Self {
        Self(self.0, self.1, self.2)
    }

    #[inline]
    pub const fn with(&self, index: usize) -> Self {
        Self(self.0, self.1, index)
    }

    #[inline]
    pub fn key(&self) -> Key {
        unsafe { *self.0.get_unchecked(self.2) }
    }

    #[inline]
    pub fn read<D: Datum>(&self, state: &Read<D>) -> &'a D {
        let data = unsafe { *self.1.get_unchecked(state.0) };
        unsafe { &*data.as_ptr().cast::<D>().add(self.2) }
    }

    #[inline]
    pub fn write<D: Datum>(&self, state: &Write<D>) -> &'a mut D {
        let data = unsafe { *self.1.get_unchecked(state.0) };
        unsafe { &mut *data.as_ptr().cast::<D>().add(self.2) }
    }
}

impl<'a> ChunkContext<'a, '_> {
    #[inline]
    pub const fn own(&self) -> Self {
        Self(self.0, self.1)
    }

    #[inline]
    pub const fn key(&self) -> &'a [Key] {
        self.0
    }

    #[inline]
    pub fn read<D: Datum>(&self, state: &Read<D>) -> &'a [D] {
        let data = unsafe { *self.1.get_unchecked(state.0) };
        unsafe { from_raw_parts(data.as_ptr().cast::<D>(), self.0.len()) }
    }

    #[inline]
    pub fn write<D: Datum>(&self, state: &Write<D>) -> &'a mut [D] {
        let data = unsafe { *self.1.get_unchecked(state.0) };
        unsafe { from_raw_parts_mut(data.as_ptr().cast::<D>(), self.0.len()) }
    }
}

unsafe impl Row for Key {
    type State = ();
    type Read = Self;
    type Item<'a> = Key;
    type Chunk<'a> = &'a [Key];

    fn declare(_: DeclareContext) -> Result<(), Error> {
        Ok(())
    }
    fn initialize(_: InitializeContext) -> Result<Self::State, Error> {
        Ok(())
    }
    fn read(state: Self::State) -> <Self::Read as Row>::State {
        state
    }
    fn item<'a>(_: &Self::State, context: ItemContext<'a, '_>) -> Self::Item<'a> {
        context.key()
    }
    fn chunk<'a>(_: &Self::State, context: ChunkContext<'a, '_>) -> Self::Chunk<'a> {
        context.key()
    }
}

unsafe impl<D: Datum> Row for &D {
    type State = Read<D>;
    type Read = Self;
    type Item<'a> = &'a D;
    type Chunk<'a> = &'a [D];

    fn declare(mut context: DeclareContext) -> Result<(), Error> {
        context.read::<D>()
    }
    fn initialize(context: InitializeContext) -> Result<Self::State, Error> {
        context.read::<D>()
    }
    fn read(state: Self::State) -> <Self::Read as Row>::State {
        state
    }
    #[inline]
    fn item<'a>(state: &Self::State, context: ItemContext<'a, '_>) -> Self::Item<'a> {
        context.read(state)
    }
    #[inline]
    fn chunk<'a>(state: &Self::State, context: ChunkContext<'a, '_>) -> Self::Chunk<'a> {
        context.read(state)
    }
}

unsafe impl<'b, D: Datum> Row for &'b mut D {
    type State = Write<D>;
    type Read = &'b D;
    type Item<'a> = &'a mut D;
    type Chunk<'a> = &'a mut [D];

    fn declare(mut context: DeclareContext) -> Result<(), Error> {
        context.write::<D>()
    }
    fn initialize(context: InitializeContext) -> Result<Self::State, Error> {
        context.write::<D>()
    }
    fn read(state: Self::State) -> <Self::Read as Row>::State {
        state.read()
    }
    #[inline]
    fn item<'a>(state: &Self::State, context: ItemContext<'a, '_>) -> Self::Item<'a> {
        context.write(state)
    }
    #[inline]
    fn chunk<'a>(state: &Self::State, context: ChunkContext<'a, '_>) -> Self::Chunk<'a> {
        context.write(state)
    }
}

unsafe impl Row for () {
    type State = ();
    type Read = ();
    type Item<'a> = ();
    type Chunk<'a> = ();

    fn declare(_: DeclareContext) -> Result<(), Error> {
        Ok(())
    }
    fn initialize(_: InitializeContext) -> Result<Self::State, Error> {
        Ok(())
    }
    fn read(_: Self::State) -> <Self::Read as Row>::State {
        ()
    }
    #[inline]
    fn item<'a>(_: &Self::State, _: ItemContext<'a, '_>) -> Self::Item<'a> {
        ()
    }
    #[inline]
    fn chunk<'a>(_: &Self::State, _: ChunkContext<'a, '_>) -> Self::Chunk<'a> {
        ()
    }
}

unsafe impl<R1: Row> Row for (R1,) {
    type State = (R1::State,);
    type Read = (R1::Read,);
    type Item<'a> = (R1::Item<'a>,);
    type Chunk<'a> = (R1::Chunk<'a>,);

    fn declare(mut context: DeclareContext) -> Result<(), Error> {
        R1::declare(context.own())?;
        Ok(())
    }
    fn initialize(context: InitializeContext) -> Result<Self::State, Error> {
        Ok((R1::initialize(context.own())?,))
    }
    fn read(state: Self::State) -> <Self::Read as Row>::State {
        (R1::read(state.0),)
    }
    #[inline]
    fn item<'a>(state: &Self::State, context: ItemContext<'a, '_>) -> Self::Item<'a> {
        (R1::item(&state.0, context.own()),)
    }
    #[inline]
    fn chunk<'a>(state: &Self::State, context: ChunkContext<'a, '_>) -> Self::Chunk<'a> {
        (R1::chunk(&state.0, context.own()),)
    }
}

unsafe impl<R1: Row, R2: Row> Row for (R1, R2) {
    type State = (R1::State, R2::State);
    type Read = (R1::Read, R2::Read);
    type Item<'a> = (R1::Item<'a>, R2::Item<'a>);
    type Chunk<'a> = (R1::Chunk<'a>, R2::Chunk<'a>);

    fn declare(mut context: DeclareContext) -> Result<(), Error> {
        R1::declare(context.own())?;
        R2::declare(context.own())?;
        Ok(())
    }
    fn initialize(context: InitializeContext) -> Result<Self::State, Error> {
        Ok((
            R1::initialize(context.own())?,
            R2::initialize(context.own())?,
        ))
    }
    fn read(state: Self::State) -> <Self::Read as Row>::State {
        (R1::read(state.0), R2::read(state.1))
    }
    #[inline]
    fn item<'a>(state: &Self::State, context: ItemContext<'a, '_>) -> Self::Item<'a> {
        (
            R1::item(&state.0, context.own()),
            R2::item(&state.1, context.own()),
        )
    }
    #[inline]
    fn chunk<'a>(state: &Self::State, context: ChunkContext<'a, '_>) -> Self::Chunk<'a> {
        (
            R1::chunk(&state.0, context.own()),
            R2::chunk(&state.1, context.own()),
        )
    }
}

unsafe impl<R1: Row, R2: Row, R3: Row> Row for (R1, R2, R3) {
    type State = (R1::State, R2::State, R3::State);
    type Read = (R1::Read, R2::Read, R3::Read);
    type Item<'a> = (R1::Item<'a>, R2::Item<'a>, R3::Item<'a>);
    type Chunk<'a> = (R1::Chunk<'a>, R2::Chunk<'a>, R3::Chunk<'a>);

    fn declare(mut context: DeclareContext) -> Result<(), Error> {
        R1::declare(context.own())?;
        R2::declare(context.own())?;
        R3::declare(context.own())?;
        Ok(())
    }
    fn initialize(context: InitializeContext) -> Result<Self::State, Error> {
        Ok((
            R1::initialize(context.own())?,
            R2::initialize(context.own())?,
            R3::initialize(context.own())?,
        ))
    }
    fn read(state: Self::State) -> <Self::Read as Row>::State {
        (R1::read(state.0), R2::read(state.1), R3::read(state.2))
    }
    #[inline]
    fn item<'a>(state: &Self::State, context: ItemContext<'a, '_>) -> Self::Item<'a> {
        (
            R1::item(&state.0, context.own()),
            R2::item(&state.1, context.own()),
            R3::item(&state.2, context.own()),
        )
    }
    #[inline]
    fn chunk<'a>(state: &Self::State, context: ChunkContext<'a, '_>) -> Self::Chunk<'a> {
        (
            R1::chunk(&state.0, context.own()),
            R2::chunk(&state.1, context.own()),
            R3::chunk(&state.2, context.own()),
        )
    }
}
