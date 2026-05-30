use crate::v4::{
    error::Error,
    guard::{Read, Write},
    meta::Meta,
    utility::{self, IteratorExtension, allocate, deallocate},
};
use arc_swap::{ArcSwapAny, AsRaw};
use core::{
    alloc::Layout,
    any::{Any, TypeId},
    iter::{FusedIterator, empty},
    ops::Range,
    ptr::{NonNull, copy_nonoverlapping, slice_from_raw_parts_mut},
    slice::{from_raw_parts, from_raw_parts_mut},
};
use parking_lot::{
    MappedRwLockReadGuard, MappedRwLockWriteGuard, RwLock, RwLockReadGuard, RwLockWriteGuard,
};
use triomphe::{Arc, ThinArc};

#[derive(Debug, Clone)]
pub struct Table(ThinArc<Header, Column>);

#[derive(Clone)]
pub struct Tables(Arc<ArcSwapAny<ThinArc<(), Table>>>);

#[derive(Debug)]
struct Header {
    pub(crate) index: u32,
    pub(crate) count: u32,
    pending: u32,
    capacity: u32,
    data: NonNull<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Row<'a> {
    row: u32,
    table: &'a Table,
}

#[derive(Clone)]
pub struct Rows<'a> {
    rows: Range<u32>,
    table: &'a Table,
}

#[derive(Debug)]
pub struct Column {
    pub(crate) meta: Meta,
    pub(crate) data: RwLock<NonNull<u8>>,
}

// TODO: Is this correct?
unsafe impl Send for Table {}
unsafe impl Sync for Table {}

// TODO: Is this correct?
unsafe impl Send for Column {}
unsafe impl Sync for Column {}

impl Tables {
    pub(crate) fn new() -> Self {
        Self(Arc::new(ArcSwapAny::new(ThinArc::from_header_and_iter(
            (),
            empty(),
        ))))
    }

    pub(crate) fn get(&self, index: u32) -> Option<Table> {
        self.0.load().slice.get(index as usize).cloned()
    }

    pub(crate) fn map<T>(&self, index: u32, map: impl FnOnce(&Table) -> T) -> Option<T> {
        self.0.load().slice.get(index as usize).map(map)
    }

    pub(crate) fn find_or_add(
        &self,
        metas: impl IntoIterator<Item = Meta>,
    ) -> Result<Table, Error> {
        let metas = sort(metas).ok_or(Error::DuplicateMeta)?;
        Ok(match self.find(&metas) {
            Some(table) => table,
            None => {
                let mut old = self.0.load();
                loop {
                    let index = old.slice.len().try_into().map_err(Error::TablesOverflow)?;
                    let table = Table::new(index, &metas);
                    let tables = ThinArc::from_header_and_iter(
                        (),
                        old.slice.iter().cloned().and(table.clone()),
                    );
                    let new = self.0.compare_and_swap(&*old, tables);
                    if old.as_raw() == new.as_raw() {
                        break table;
                    } else {
                        old = new;
                    }
                }
            }
        })
    }

    fn find(&self, metas: &[Meta]) -> Option<Table> {
        self.0
            .load()
            .slice
            .iter()
            .find(|table| {
                table
                    .columns()
                    .iter()
                    .map(|column| column.meta().identifier())
                    .eq(metas.iter().map(|meta| meta.identifier()))
            })
            .cloned()
    }
}

impl Column {
    pub(crate) const fn new(meta: Meta) -> Self {
        Self {
            meta,
            data: RwLock::new(NonNull::dangling()),
        }
    }

    pub const fn meta(&self) -> Meta {
        self.meta
    }

    pub(crate) unsafe fn read<T: 'static>(&self, count: u32) -> Read<'_, [T]> {
        debug_assert_eq!(self.meta.identifier(), TypeId::of::<T>());
        RwLockReadGuard::map(self.data.read(), |data| unsafe {
            from_raw_parts(data.cast::<T>().as_ptr(), count as usize)
        })
        .into()
    }

    pub(crate) unsafe fn try_read<T: 'static>(&self, count: u32) -> Option<Read<'_, [T]>> {
        debug_assert_eq!(self.meta.identifier(), TypeId::of::<T>());
        Some(
            RwLockReadGuard::map(self.data.try_read()?, |data| unsafe {
                from_raw_parts(data.cast::<T>().as_ptr(), count as usize)
            })
            .into(),
        )
    }

    pub(crate) unsafe fn write<T: 'static>(&self, count: u32) -> Write<'_, [T]> {
        debug_assert_eq!(self.meta.identifier(), TypeId::of::<T>());
        RwLockWriteGuard::map(self.data.write(), |data| unsafe {
            from_raw_parts_mut(data.cast::<T>().as_ptr(), count as usize)
        })
        .into()
    }

    pub(crate) unsafe fn try_write<T: 'static>(&self, count: u32) -> Option<Write<'_, [T]>> {
        debug_assert_eq!(self.meta.identifier(), TypeId::of::<T>());
        Some(
            RwLockWriteGuard::map(self.data.try_write()?, |data| unsafe {
                from_raw_parts_mut(data.cast::<T>().as_ptr(), count as usize)
            })
            .into(),
        )
    }

    pub(crate) unsafe fn set<T: 'static>(&self, item: T, row: u32) {
        debug_assert_eq!(self.meta.identifier(), TypeId::of::<T>());
        unsafe { self.data().cast::<T>().add(row as usize).write(item) };
    }

    pub(crate) unsafe fn copy<T: 'static>(&self, source: NonNull<T>, row: u32, count: u32) -> bool {
        debug_assert_eq!(self.meta.identifier(), TypeId::of::<T>());
        if size_of::<T>() > 0 && count > 0 {
            let target = unsafe { self.data().cast::<T>().add(row as usize) };
            unsafe { copy_nonoverlapping(source.as_ptr(), target.as_ptr(), count as usize) };
            true
        } else {
            false
        }
    }

    pub(crate) unsafe fn drop<T: 'static>(&self, row: u32, count: u32) {
        debug_assert_eq!(self.meta.identifier(), TypeId::of::<T>());
        let data = unsafe { self.data().cast::<T>().add(row as usize) };
        unsafe { slice_from_raw_parts_mut(data.as_ptr(), count as usize).drop_in_place() };
    }

    pub(crate) unsafe fn get_with(&self, meta: Meta, row: u32) -> &dyn Any {
        unsafe { meta.get(meta.offset(self.data(), row)) }
    }

    pub(crate) unsafe fn set_with(&self, item: Box<dyn Any>, row: u32, meta: Meta) -> bool {
        unsafe { meta.set(meta.offset(self.data(), row), item) }
    }

    pub(crate) unsafe fn drop_with(&self, row: u32, count: u32, meta: Meta) -> bool {
        unsafe { meta.drop(meta.offset(self.data(), row), count) }
    }

    pub(crate) fn data(&self) -> NonNull<u8> {
        unsafe { *self.data.data_ptr() }
    }
}

impl<'a> Row<'a> {
    pub(crate) const fn new(row: u32, table: &'a Table) -> Self {
        Self { row, table }
    }

    pub const fn row(&self) -> u32 {
        self.row
    }

    pub fn table(&self) -> u32 {
        self.table.index()
    }
}

impl PartialOrd for Row<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (self.table.address(), self.row).partial_cmp(&(other.table.address(), other.row))
    }
}

impl Ord for Row<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.table.address(), self.row).cmp(&(other.table.address(), other.row))
    }
}

impl<'a> Rows<'a> {
    pub(crate) const fn new(rows: Range<u32>, table: &'a Table) -> Self {
        Self { rows, table }
    }
}

impl<'a> Iterator for Rows<'a> {
    type Item = Row<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(Row::new(self.rows.next()?, self.table))
    }
}

impl ExactSizeIterator for Rows<'_> {
    fn len(&self) -> usize {
        self.rows.len()
    }
}

impl DoubleEndedIterator for Rows<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(Row::new(self.rows.next_back()?, self.table))
    }
}

impl FusedIterator for Rows<'_> {}

impl Table {
    pub(super) fn new(index: u32, metas: &[Meta]) -> Self {
        Self(ThinArc::from_header_and_iter(
            Header {
                index,
                count: 0,
                pending: 0,
                capacity: 0,
                data: NonNull::dangling(),
            },
            metas.iter().copied().map(Column::new),
        ))
    }

    pub(crate) fn address(&self) -> usize {
        self.0.as_ptr().addr()
    }

    pub(crate) fn column(&self, identifier: TypeId) -> Option<u32> {
        utility::find(&self.0.slice, identifier, |column| column.meta.identifier())?
            .try_into()
            .ok()
    }

    pub fn columns(&self) -> &[Column] {
        &self.0.slice
    }

    pub fn index(&self) -> u32 {
        self.header().index
    }

    pub fn count(&self) -> u32 {
        self.header().count
    }

    pub fn capacity(&self) -> u32 {
        self.header().capacity
    }

    pub(crate) fn reserve(&self, count: u32) -> Result<Range<u32>, Error> {
        let old = self.header().pending;
        let new = self
            .header()
            .pending
            .checked_add(count)
            .ok_or(Error::TableOverflow)?;
        self.header().pending = new;
        Ok(old..new)
    }

    pub(crate) fn ensure(&self) -> Result<bool, Error> {
        if self.header().pending > self.header().capacity {
            let capacity = self
                .header()
                .pending
                .checked_next_power_of_two()
                .ok_or(Error::TableOverflow)?;
            self.header().data = resize(
                &self.0.slice,
                self.header().data,
                self.header().count,
                (self.header().capacity, capacity),
            )?;
            self.header().capacity = capacity;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub(crate) fn commit(&self) -> Range<u32> {
        debug_assert!(self.header().count <= self.header().pending);
        let rows = self.header().count..self.header().pending;
        self.header().count = self.header().pending;
        rows
    }

    pub(super) fn release(&self, rows: Range<u32>) {
        if rows.is_empty() {
            return;
        }

        let count = rows.end.saturating_sub(rows.start);
        debug_assert!(rows.end <= self.header().pending);

        let copy = self.header().pending.saturating_sub(rows.end).min(count);
        let copy = (self.header().pending - copy, copy);
        for column in &self.0.slice {
            let data = column.data();
            unsafe { column.meta.drop_at(data, rows.start, count) };
            unsafe {
                column
                    .meta
                    .copy_at((data, copy.0), (data, rows.start), copy.1)
            };
        }
        self.header().pending = self.header().pending.saturating_sub(count);
        self.header().count = self
            .header()
            .count
            .saturating_sub(count)
            .min(self.header().pending);
    }

    fn header(&self) -> &Header {
        &self.0.header.header
    }
}

impl PartialEq for Table {
    fn eq(&self, other: &Self) -> bool {
        self.address() == other.address()
    }
}

impl Eq for Table {}

impl Drop for Table {
    fn drop(&mut self) {
        let header = self.header();
        let _ = resize(
            &self.columns(),
            header.data,
            header.count,
            (header.capacity, 0),
        );
    }
}
pub(crate) fn resize(
    columns: &[Column],
    data: NonNull<u8>,
    count: u32,
    capacities: (u32, u32),
) -> Result<NonNull<u8>, Error> {
    fn next(
        columns: &[Column],
        layouts: (Layout, Layout),
        count: u32,
        capacities: (u32, u32),
    ) -> Result<(Layout, NonNull<u8>), Error> {
        Ok(match columns.split_first_mut() {
            Some((head, tail)) => {
                let old = head
                    .meta
                    .extend(layouts.0, capacities.0)
                    .map_err(Error::Layout)?;
                let new = head
                    .meta
                    .extend(layouts.1, capacities.1)
                    .map_err(Error::Layout)?;
                let pair = next(tail, (old.0, new.0), count, capacities)?;
                let source = head.data();
                let target = unsafe { pair.1.add(new.1) };
                head.meta.initialize(source, target, count, capacities.1);
                head.data = target;
                pair
            }
            None if layouts.1.size() == 0 => (layouts.0.pad_to_align(), NonNull::dangling()),
            None => (layouts.0.pad_to_align(), unsafe {
                allocate(layouts.1.pad_to_align())
            }?),
        })
    }

    let (old, new) = next(
        columns,
        (Layout::new::<()>(), Layout::new::<()>()),
        count,
        capacities,
    )?;
    unsafe { deallocate(data, old) };
    Ok(new)
}

pub(crate) fn sort<T: Ord>(items: impl IntoIterator<Item = T>) -> Option<Vec<T>> {
    let mut items = items.into_iter().collect::<Vec<_>>();
    items.sort_unstable();
    for [left, right] in items.array_windows::<2>() {
        if left == right {
            return None;
        }
    }
    Some(items)
}
