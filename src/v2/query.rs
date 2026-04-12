//! Query vocabulary and chunk-view projection for the rewrite lane.
//!
//! Task 04 adds the first real query surface:
//!
//! - typed read and write descriptors,
//! - generated row-handle requests,
//! - optional sub-queries that remain zip-friendly,
//! - table-level filters,
//! - conjunctive query composition,
//! - and conservative access-conflict analysis.

pub use crate::v2::schema::{Row, Rows};

use crate::v2::schema::{ChunkError, ChunkIndex, Table, TableIndex};
use core::{
    any::{type_name, TypeId},
    iter::FusedIterator,
    marker::PhantomData,
    slice::{Iter, IterMut},
};
use std::collections::BTreeSet;

/// The access level a future query item may request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Access {
    Read,
    Write,
}

impl Access {
    const fn conflicts_with(self, other: Self) -> bool {
        matches!(
            (self, other),
            (Self::Write, Self::Read | Self::Write) | (Self::Read, Self::Write)
        )
    }
}

/// One declared access requirement produced by a query descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeclaredAccess {
    identifier: TypeId,
    type_name: &'static str,
    access: Access,
}

impl DeclaredAccess {
    pub const fn identifier(self) -> TypeId {
        self.identifier
    }

    pub const fn type_name(self) -> &'static str {
        self.type_name
    }

    pub const fn access(self) -> Access {
        self.access
    }
}

/// Query-construction and projection failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    ConflictingAccess {
        left_type_name: &'static str,
        left_access: Access,
        right_type_name: &'static str,
        right_access: Access,
    },
    TableDoesNotMatch {
        table_index: TableIndex,
    },
    InvalidOneCardinality {
        table_index: TableIndex,
        chunk_index: ChunkIndex,
        count: usize,
    },
    Chunk(ChunkError),
}

impl From<ChunkError> for Error {
    fn from(error: ChunkError) -> Self {
        Self::Chunk(error)
    }
}

/// Slice-like chunk view behavior shared by real slices, generated rows, and optional wrappers.
pub trait View: Sized {
    type Element;
    type IntoIter: DoubleEndedIterator<Item = Self::Element> + ExactSizeIterator + FusedIterator;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn into_view_iter(self) -> Self::IntoIter;

    fn zip<I>(self, other: I) -> core::iter::Zip<Self::IntoIter, I::IntoIter>
    where
        I: IntoIterator,
    {
        self.into_view_iter().zip(other)
    }
}

impl<'table, T> View for &'table [T] {
    type Element = &'table T;
    type IntoIter = Iter<'table, T>;

    fn len(&self) -> usize {
        <[T]>::len(self)
    }

    fn into_view_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'table, T> View for &'table mut [T] {
    type Element = &'table mut T;
    type IntoIter = IterMut<'table, T>;

    fn len(&self) -> usize {
        <[T]>::len(self)
    }

    fn into_view_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'job> View for Rows<'job> {
    type Element = Row<'job>;
    type IntoIter = crate::v2::schema::RowsIter<'job>;

    fn len(&self) -> usize {
        self.len()
    }

    fn into_view_iter(self) -> Self::IntoIter {
        self.into_iter()
    }
}

/// A zip-friendly optional chunk view.
///
/// When the wrapped sub-query is absent from the matched table, this view still exposes the chunk
/// length and yields `None` for every iterated row position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Optional<V> {
    value: Option<V>,
    len: usize,
}

impl<V> Optional<V> {
    pub const fn is_present(&self) -> bool {
        self.value.is_some()
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<V: View> Optional<V> {
    fn some(value: V) -> Self {
        let len = value.len();
        Self {
            value: Some(value),
            len,
        }
    }

    fn none(len: usize) -> Self {
        Self { value: None, len }
    }

    pub fn zip<I>(
        self,
        other: I,
    ) -> core::iter::Zip<OptionalIter<V::IntoIter, V::Element>, I::IntoIter>
    where
        I: IntoIterator,
    {
        self.into_iter().zip(other)
    }
}

impl<V: View> IntoIterator for Optional<V> {
    type IntoIter = OptionalIter<V::IntoIter, V::Element>;
    type Item = Option<V::Element>;

    fn into_iter(self) -> Self::IntoIter {
        match self.value {
            Some(value) => OptionalIter::Present(PresentOptionalIter {
                iterator: value.into_view_iter(),
            }),
            None => OptionalIter::Missing(MissingOptionalIter {
                remaining: self.len,
                marker: PhantomData,
            }),
        }
    }
}

impl<V: View> View for Optional<V> {
    type Element = Option<V::Element>;
    type IntoIter = OptionalIter<V::IntoIter, V::Element>;

    fn len(&self) -> usize {
        self.len
    }

    fn into_view_iter(self) -> Self::IntoIter {
        self.into_iter()
    }
}

#[derive(Debug, Clone)]
pub enum OptionalIter<I, T> {
    Present(PresentOptionalIter<I>),
    Missing(MissingOptionalIter<T>),
}

impl<I, T> Iterator for OptionalIter<I, T>
where
    I: Iterator<Item = T>,
{
    type Item = Option<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Present(iterator) => iterator.next(),
            Self::Missing(iterator) => iterator.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Present(iterator) => iterator.size_hint(),
            Self::Missing(iterator) => iterator.size_hint(),
        }
    }
}

impl<I, T> DoubleEndedIterator for OptionalIter<I, T>
where
    I: DoubleEndedIterator<Item = T> + ExactSizeIterator,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match self {
            Self::Present(iterator) => iterator.next_back(),
            Self::Missing(iterator) => iterator.next_back(),
        }
    }
}

impl<I, T> ExactSizeIterator for OptionalIter<I, T>
where
    I: ExactSizeIterator<Item = T>,
{
    fn len(&self) -> usize {
        match self {
            Self::Present(iterator) => iterator.len(),
            Self::Missing(iterator) => iterator.len(),
        }
    }
}

impl<I, T> FusedIterator for OptionalIter<I, T> where I: FusedIterator<Item = T> {}

#[derive(Debug, Clone)]
pub struct PresentOptionalIter<I> {
    iterator: I,
}

impl<I> Iterator for PresentOptionalIter<I>
where
    I: Iterator,
{
    type Item = Option<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iterator.next().map(Some)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iterator.size_hint()
    }
}

impl<I> DoubleEndedIterator for PresentOptionalIter<I>
where
    I: DoubleEndedIterator + ExactSizeIterator,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iterator.next_back().map(Some)
    }
}

impl<I> ExactSizeIterator for PresentOptionalIter<I>
where
    I: ExactSizeIterator,
{
    fn len(&self) -> usize {
        self.iterator.len()
    }
}

impl<I> FusedIterator for PresentOptionalIter<I> where I: FusedIterator {}

#[derive(Debug, Clone)]
pub struct MissingOptionalIter<T> {
    remaining: usize,
    marker: PhantomData<fn() -> T>,
}

impl<T> Iterator for MissingOptionalIter<T> {
    type Item = Option<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        self.remaining -= 1;
        Some(None)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T> DoubleEndedIterator for MissingOptionalIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.next()
    }
}

impl<T> ExactSizeIterator for MissingOptionalIter<T> {
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<T> FusedIterator for MissingOptionalIter<T> {}

/// A request for transient row handles aligned with a chunk's inhabited prefix.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct RowsRequest;

/// A read-only chunk-slice query item.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Read<T> {
    marker: PhantomData<fn() -> T>,
}

/// A mutable chunk-slice query item.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Write<T> {
    marker: PhantomData<fn() -> T>,
}

/// A query item wrapper that yields a zip-friendly optional view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OptionQuery<Q> {
    query: Q,
}

/// A query item wrapper that expects exactly one row in the matched chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct One<Q> {
    query: Q,
}

/// A table-level admission filter requiring the presence of one type.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Has<T> {
    marker: PhantomData<fn() -> T>,
}

/// A negated table-level filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Not<F> {
    filter: F,
}

/// The default filter that admits every table.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct AllowAll;

/// The analyzed filter constraints for conservative disjointness proofs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FilterPlan {
    required_identifiers: BTreeSet<TypeId>,
    forbidden_identifiers: BTreeSet<TypeId>,
}

impl FilterPlan {
    pub fn required_identifiers(&self) -> &BTreeSet<TypeId> {
        &self.required_identifiers
    }

    pub fn forbidden_identifiers(&self) -> &BTreeSet<TypeId> {
        &self.forbidden_identifiers
    }

    pub fn proves_disjointness_with(&self, other: &Self) -> bool {
        !self
            .required_identifiers
            .is_disjoint(&other.forbidden_identifiers)
            || !self
                .forbidden_identifiers
                .is_disjoint(&other.required_identifiers)
    }

    fn merge(&mut self, other: &Self) {
        self.required_identifiers
            .extend(other.required_identifiers.iter().copied());
        self.forbidden_identifiers
            .extend(other.forbidden_identifiers.iter().copied());
    }
}

/// The analyzed form of one conjunctive query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Analysis {
    declared_accesses: Box<[DeclaredAccess]>,
    filter_plan: FilterPlan,
}

impl Analysis {
    pub fn declared_accesses(&self) -> &[DeclaredAccess] {
        &self.declared_accesses
    }

    pub fn filter_plan(&self) -> &FilterPlan {
        &self.filter_plan
    }

    pub fn conflicts_with(&self, other: &Self) -> bool {
        if self
            .filter_plan
            .proves_disjointness_with(&other.filter_plan)
        {
            return false;
        }

        self.declared_accesses.iter().any(|left| {
            other.declared_accesses.iter().any(|right| {
                left.identifier == right.identifier && left.access.conflicts_with(right.access)
            })
        })
    }
}

/// One conjunctive query stream with optional table-level filters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct All<Q, F = AllowAll> {
    query: Q,
    filter: F,
    analysis: Analysis,
}

pub const fn rows() -> RowsRequest {
    RowsRequest
}

pub const fn read<T>() -> Read<T> {
    Read {
        marker: PhantomData,
    }
}

pub const fn write<T>() -> Write<T> {
    Write {
        marker: PhantomData,
    }
}

pub const fn option<Q>(query: Q) -> OptionQuery<Q> {
    OptionQuery { query }
}

pub const fn one<Q>(query: Q) -> One<Q> {
    One { query }
}

pub const fn has<T>() -> Has<T> {
    Has {
        marker: PhantomData,
    }
}

pub const fn not<F>(filter: F) -> Not<F> {
    Not { filter }
}

pub fn all<Q>(query: Q) -> Result<All<Q>, Error>
where
    Q: QueryTuple,
{
    let filter = AllowAll;
    let analysis = build_analysis(&query, &filter)?;

    Ok(All {
        query,
        filter,
        analysis,
    })
}

pub trait Filter: Sized {
    fn matches(&self, table: &Table) -> bool;

    fn plan(&self) -> Option<FilterPlan> {
        Some(FilterPlan::default())
    }
}

impl Filter for AllowAll {
    fn matches(&self, _table: &Table) -> bool {
        true
    }
}

impl<T: 'static> Filter for Has<T> {
    fn matches(&self, table: &Table) -> bool {
        table.meta_for::<T>().is_some()
    }

    fn plan(&self) -> Option<FilterPlan> {
        let mut plan = FilterPlan::default();
        plan.required_identifiers.insert(TypeId::of::<T>());
        Some(plan)
    }
}

impl<F: Filter> Filter for Not<F> {
    fn matches(&self, table: &Table) -> bool {
        !self.filter.matches(table)
    }

    fn plan(&self) -> Option<FilterPlan> {
        let inner = self.filter.plan()?;
        Some(FilterPlan {
            required_identifiers: inner.forbidden_identifiers,
            forbidden_identifiers: inner.required_identifiers,
        })
    }
}

macro_rules! impl_filter_tuple {
    ($(($( $type_name:ident : $value_name:ident ),+)),+ $(,)?) => {
        $(
            impl<$($type_name),+> Filter for ($($type_name,)+)
            where
                $($type_name: Filter,)+
            {
                fn matches(&self, table: &Table) -> bool {
                    let ($($value_name,)+) = self;
                    true $(&& $value_name.matches(table))+
                }

                fn plan(&self) -> Option<FilterPlan> {
                    let ($($value_name,)+) = self;
                    let mut plan = FilterPlan::default();
                    $(
                        plan.merge(&$value_name.plan()?);
                    )+
                    Some(plan)
                }
            }
        )+
    };
}

impl_filter_tuple!(
    (F0: f0, F1: f1),
    (F0: f0, F1: f1, F2: f2),
    (F0: f0, F1: f1, F2: f2, F3: f3)
);

pub trait Query {
    type Item<'table, 'job>
    where
        Self: 'table;

    fn collect_declared_accesses(&self, declared_accesses: &mut Vec<DeclaredAccess>);

    fn matches_table(&self, table: &Table) -> bool;

    /// # Safety
    ///
    /// The caller must ensure that all combined projections from the same raw `table` pointer have
    /// been validated for alias safety up front.
    unsafe fn project_chunk<'table, 'job>(
        &self,
        table: *mut Table,
        chunk_index: ChunkIndex,
    ) -> Result<Self::Item<'table, 'job>, Error>;
}

impl Query for RowsRequest {
    type Item<'table, 'job> = Rows<'job>;

    fn collect_declared_accesses(&self, _declared_accesses: &mut Vec<DeclaredAccess>) {}

    fn matches_table(&self, _table: &Table) -> bool {
        true
    }

    unsafe fn project_chunk<'table, 'job>(
        &self,
        table: *mut Table,
        chunk_index: ChunkIndex,
    ) -> Result<Self::Item<'table, 'job>, Error> {
        // Safety: the caller ensures projection aliasing has already been validated.
        unsafe { (&*table).rows(chunk_index).map_err(Error::from) }
    }
}

impl<T: 'static> Query for Read<T> {
    type Item<'table, 'job> = &'table [T];

    fn collect_declared_accesses(&self, declared_accesses: &mut Vec<DeclaredAccess>) {
        declared_accesses.push(DeclaredAccess {
            identifier: TypeId::of::<T>(),
            type_name: type_name::<T>(),
            access: Access::Read,
        });
    }

    fn matches_table(&self, table: &Table) -> bool {
        table.inline_meta_for::<T>().is_some()
    }

    unsafe fn project_chunk<'table, 'job>(
        &self,
        table: *mut Table,
        chunk_index: ChunkIndex,
    ) -> Result<Self::Item<'table, 'job>, Error> {
        // Safety: the caller ensures projection aliasing has already been validated.
        unsafe { (&*table).slice::<T>(chunk_index).map_err(Error::from) }
    }
}

impl<T: 'static> Query for Write<T> {
    type Item<'table, 'job> = &'table mut [T];

    fn collect_declared_accesses(&self, declared_accesses: &mut Vec<DeclaredAccess>) {
        declared_accesses.push(DeclaredAccess {
            identifier: TypeId::of::<T>(),
            type_name: type_name::<T>(),
            access: Access::Write,
        });
    }

    fn matches_table(&self, table: &Table) -> bool {
        table.inline_meta_for::<T>().is_some()
    }

    unsafe fn project_chunk<'table, 'job>(
        &self,
        table: *mut Table,
        chunk_index: ChunkIndex,
    ) -> Result<Self::Item<'table, 'job>, Error> {
        // Safety: the caller ensures projection aliasing has already been validated.
        unsafe {
            (&mut *table)
                .slice_mut::<T>(chunk_index)
                .map_err(Error::from)
        }
    }
}

impl<Q> Query for OptionQuery<Q>
where
    Q: Query,
    for<'table, 'job> Q::Item<'table, 'job>: View,
{
    type Item<'table, 'job>
        = Optional<Q::Item<'table, 'job>>
    where
        Self: 'table;

    fn collect_declared_accesses(&self, declared_accesses: &mut Vec<DeclaredAccess>) {
        self.query.collect_declared_accesses(declared_accesses);
    }

    fn matches_table(&self, _table: &Table) -> bool {
        true
    }

    unsafe fn project_chunk<'table, 'job>(
        &self,
        table: *mut Table,
        chunk_index: ChunkIndex,
    ) -> Result<Self::Item<'table, 'job>, Error> {
        // Safety: the caller ensures projection aliasing has already been validated.
        if unsafe { self.query.matches_table(&*table) } {
            // Safety: same as above.
            let value = unsafe { self.query.project_chunk(table, chunk_index)? };
            Ok(Optional::some(value))
        } else {
            // Safety: the caller ensures the pointer stays valid for the projection.
            let chunk = unsafe { (&*table).chunk(chunk_index) }
                .ok_or(ChunkError::MissingChunk { chunk_index })?;
            Ok(Optional::none(chunk.count()))
        }
    }
}

impl<Q> Query for One<Q>
where
    Q: Query,
    for<'table, 'job> Q::Item<'table, 'job>: View,
{
    type Item<'table, 'job>
        = <Q::Item<'table, 'job> as View>::Element
    where
        Self: 'table;

    fn collect_declared_accesses(&self, declared_accesses: &mut Vec<DeclaredAccess>) {
        self.query.collect_declared_accesses(declared_accesses);
    }

    fn matches_table(&self, table: &Table) -> bool {
        self.query.matches_table(table)
    }

    unsafe fn project_chunk<'table, 'job>(
        &self,
        table: *mut Table,
        chunk_index: ChunkIndex,
    ) -> Result<Self::Item<'table, 'job>, Error> {
        // Safety: the caller ensures projection aliasing has already been validated.
        let view = unsafe { self.query.project_chunk(table, chunk_index)? };
        let count = view.len();
        if count != 1 {
            // Safety: the caller ensures `table` remains a valid pointer here.
            let table_index = unsafe { (&*table).index() };
            return Err(Error::InvalidOneCardinality {
                table_index,
                chunk_index,
                count,
            });
        }

        view.into_view_iter().next().ok_or_else(|| {
            let table_index = unsafe { (&*table).index() };
            Error::InvalidOneCardinality {
                table_index,
                chunk_index,
                count,
            }
        })
    }
}

pub trait QueryTuple {
    type Item<'table, 'job>
    where
        Self: 'table;

    fn collect_declared_accesses(&self, declared_accesses: &mut Vec<DeclaredAccess>);

    fn matches_table(&self, table: &Table) -> bool;

    /// # Safety
    ///
    /// The caller must ensure that all combined projections from the same raw `table` pointer have
    /// been validated for alias safety up front.
    unsafe fn project_chunk<'table, 'job>(
        &self,
        table: *mut Table,
        chunk_index: ChunkIndex,
    ) -> Result<Self::Item<'table, 'job>, Error>;
}

impl<Q> QueryTuple for Q
where
    Q: Query,
{
    type Item<'table, 'job>
        = Q::Item<'table, 'job>
    where
        Self: 'table;

    fn collect_declared_accesses(&self, declared_accesses: &mut Vec<DeclaredAccess>) {
        self.collect_declared_accesses(declared_accesses);
    }

    fn matches_table(&self, table: &Table) -> bool {
        self.matches_table(table)
    }

    unsafe fn project_chunk<'table, 'job>(
        &self,
        table: *mut Table,
        chunk_index: ChunkIndex,
    ) -> Result<Self::Item<'table, 'job>, Error> {
        // Safety: forwarded from caller.
        unsafe { Query::project_chunk(self, table, chunk_index) }
    }
}

macro_rules! impl_query_tuple {
    ($(($( $type_name:ident : $value_name:ident ),+)),+ $(,)?) => {
        $(
            impl<$($type_name),+> QueryTuple for ($($type_name,)+)
            where
                $($type_name: Query,)+
            {
                type Item<'table, 'job> = ($($type_name::Item<'table, 'job>,)+)
                where
                    Self: 'table;

                fn collect_declared_accesses(&self, declared_accesses: &mut Vec<DeclaredAccess>) {
                    let ($($value_name,)+) = self;
                    $(
                        $value_name.collect_declared_accesses(declared_accesses);
                    )+
                }

                fn matches_table(&self, table: &Table) -> bool {
                    let ($($value_name,)+) = self;
                    true $(&& $value_name.matches_table(table))+
                }

                unsafe fn project_chunk<'table, 'job>(
                    &self,
                    table: *mut Table,
                    chunk_index: ChunkIndex,
                ) -> Result<Self::Item<'table, 'job>, Error> {
                    let ($($value_name,)+) = self;
                    Ok((
                        $(
                            // Safety: forwarded from caller after access validation.
                            unsafe { $value_name.project_chunk(table, chunk_index)? },
                        )+
                    ))
                }
            }
        )+
    };
}

impl_query_tuple!(
    (Q0: q0, Q1: q1),
    (Q0: q0, Q1: q1, Q2: q2),
    (Q0: q0, Q1: q1, Q2: q2, Q3: q3)
);

impl<Q, F> All<Q, F>
where
    Q: QueryTuple,
    F: Filter,
{
    pub fn filter<G>(self, filter: G) -> All<Q, (F, G)>
    where
        G: Filter,
    {
        let filter = (self.filter, filter);
        let analysis = Analysis {
            declared_accesses: self.analysis.declared_accesses,
            filter_plan: filter.plan().unwrap_or_default(),
        };

        All {
            query: self.query,
            filter,
            analysis,
        }
    }

    pub fn analysis(&self) -> &Analysis {
        &self.analysis
    }

    pub fn matches_table(&self, table: &Table) -> bool {
        self.filter.matches(table) && self.query.matches_table(table)
    }

    pub fn conflicts_with<Q2, F2>(&self, other: &All<Q2, F2>) -> bool
    where
        Q2: QueryTuple,
        F2: Filter,
    {
        self.analysis.conflicts_with(&other.analysis)
    }

    pub fn project_chunk<'table, 'job>(
        &self,
        table: &'table mut Table,
        chunk_index: ChunkIndex,
    ) -> Result<Q::Item<'table, 'job>, Error> {
        if !self.matches_table(table) {
            return Err(Error::TableDoesNotMatch {
                table_index: table.index(),
            });
        }

        let table_pointer = table as *mut Table;

        // Safety: `query::all(...)` validated this conjunctive query at construction time, and the
        // filter plus required-column checks above ensure projection only runs on matching tables.
        unsafe { self.query.project_chunk(table_pointer, chunk_index) }
    }
}

fn build_analysis<Q, F>(query: &Q, filter: &F) -> Result<Analysis, Error>
where
    Q: QueryTuple,
    F: Filter,
{
    let mut declared_accesses = Vec::new();
    query.collect_declared_accesses(&mut declared_accesses);
    validate_declared_accesses(&declared_accesses)?;

    Ok(Analysis {
        declared_accesses: declared_accesses.into_boxed_slice(),
        filter_plan: filter.plan().unwrap_or_default(),
    })
}

fn validate_declared_accesses(declared_accesses: &[DeclaredAccess]) -> Result<(), Error> {
    for (left_index, left_access) in declared_accesses.iter().enumerate() {
        for right_access in &declared_accesses[(left_index + 1)..] {
            if left_access.identifier == right_access.identifier
                && left_access.access.conflicts_with(right_access.access)
            {
                return Err(Error::ConflictingAccess {
                    left_type_name: left_access.type_name,
                    left_access: left_access.access,
                    right_type_name: right_access.type_name,
                    right_access: right_access.access,
                });
            }
        }
    }

    Ok(())
}
