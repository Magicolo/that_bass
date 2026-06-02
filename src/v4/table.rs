use crate::v4::{
    depend::Access,
    error::Error,
    meta::Meta,
    slice::Slice,
    utility::{self, IteratorExtension, allocate, deallocate, defer},
};
use arc_swap::{ArcSwapAny, AsRaw};
use core::{
    alloc::Layout,
    any::{Any, TypeId},
    iter::{FusedIterator, empty},
    ops::Range,
    ptr::{NonNull, replace},
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::atomic::{AtomicU32, Ordering},
};
use parking_lot::{RwLock, lock_api::RawRwLock};
use triomphe::{Arc, ThinArc};

#[derive(Debug, Clone)]
pub struct Table(ThinArc<Header, Column>);

#[derive(Debug, Clone)]
pub struct Tables(Arc<ArcSwapAny<ThinArc<(), Table>>>);

#[derive(Debug)]
struct Header {
    index: u32,
    count: AtomicU32,
    capacity: AtomicU32,
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
    meta: Meta,
    data: RwLock<NonNull<u8>>,
}

unsafe impl Send for Column {}
unsafe impl Sync for Column {}

impl Tables {
    pub(crate) fn new() -> Self {
        Self(Arc::new(ArcSwapAny::new(ThinArc::from_header_and_iter(
            (),
            empty(),
        ))))
    }

    pub fn len(&self) -> usize {
        self.0.load().slice.len()
    }

    pub fn get(&self, index: u32) -> Option<Table> {
        self.map(index, Table::clone)
    }

    pub(crate) fn map<T>(&self, index: u32, map: impl FnOnce(&Table) -> T) -> Option<T> {
        self.0.load().slice.get(index as usize).map(map)
    }

    pub(crate) fn find_or_add(
        &self,
        metas: impl IntoIterator<Item = Meta>,
    ) -> Result<Table, Error> {
        let metas = sort(metas).ok_or(Error::DuplicateMeta)?;
        let mut old = self.0.load();
        loop {
            match old
                .slice
                .iter()
                .find(|table| table.is(metas.iter().copied()))
            {
                Some(table) => break Ok(table.clone()),
                None => {
                    let index = old.slice.len().try_into().map_err(Error::TablesOverflow)?;
                    let table = Table::new(index, &metas);
                    let tables = ThinArc::from_header_and_iter(
                        (),
                        old.slice.iter().cloned().and(table.clone()),
                    );
                    let new = self.0.compare_and_swap(&*old, tables);
                    if old.as_raw() == new.as_raw() {
                        break Ok(table);
                    } else {
                        old = new;
                    }
                }
            }
        }
    }
}

impl Default for Tables {
    fn default() -> Self {
        Self::new()
    }
}

impl Column {
    #[inline]
    pub(crate) const fn new(meta: Meta) -> Self {
        Self {
            meta,
            data: RwLock::new(NonNull::dangling()),
        }
    }

    #[inline]
    pub const fn meta(&self) -> Meta {
        self.meta
    }

    #[inline]
    pub(crate) unsafe fn lock(&self, access: Access) -> bool {
        if self.meta.size() == 0 {
            false
        } else {
            match access {
                Access::Read => unsafe { self.data.raw().lock_shared() },
                Access::Write => unsafe { self.data.raw().lock_exclusive() },
            }
            true
        }
    }

    #[inline]
    pub(crate) unsafe fn unlock(&self, access: Access) -> bool {
        if self.meta.size() == 0 {
            false
        } else {
            match access {
                Access::Read => unsafe { self.data.raw().unlock_shared() },
                Access::Write => unsafe { self.data.raw().unlock_exclusive() },
            }
            true
        }
    }

    #[inline]
    pub(crate) unsafe fn get<T: 'static>(&self, count: u32) -> &[T] {
        debug_assert_eq!(self.meta.identifier(), TypeId::of::<T>());
        unsafe { from_raw_parts(self.data().cast::<T>().as_ptr(), count as usize) }
    }

    #[inline]
    pub(crate) unsafe fn get_mut<T: 'static>(&self, count: u32) -> &mut [T] {
        debug_assert_eq!(self.meta.identifier(), TypeId::of::<T>());
        unsafe { from_raw_parts_mut(self.data().cast::<T>().as_ptr(), count as usize) }
    }

    #[inline]
    pub(crate) unsafe fn get_in(&self, slice: &mut Slice, count: u32) {
        debug_assert_eq!(self.meta, slice.meta());
        unsafe { slice.set_parts(self.data(), count as _) };
    }

    #[inline]
    pub(crate) unsafe fn set_at<T: 'static>(&self, item: T, row: u32) {
        debug_assert_eq!(self.meta.identifier(), TypeId::of::<T>());
        unsafe { self.data().cast::<T>().add(row as usize).write(item) };
    }

    #[inline]
    pub(crate) unsafe fn set_at_with(&self, item: Box<dyn Any>, row: u32) {
        debug_assert_eq!(self.meta.identifier(), item.type_id());
        unsafe { self.meta.set_at(self.data(), item, row) };
    }

    #[inline]
    pub(crate) unsafe fn copy_at(&self, source: u32, target: u32, count: u32) -> bool {
        let data = unsafe { self.data() };
        unsafe { self.meta.copy_at((data, source), (data, target), count) }
    }

    #[inline]
    pub(crate) unsafe fn drop_at(&self, row: u32, count: u32) -> bool {
        unsafe { self.meta.drop_at(self.data(), row, count) }
    }

    unsafe fn data(&self) -> NonNull<u8> {
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

    pub fn table(&self) -> u32 {
        self.table.index()
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
                count: AtomicU32::new(0),
                capacity: AtomicU32::new(0),
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
        self.header().count.load(Ordering::Acquire)
    }

    pub fn capacity(&self) -> u32 {
        self.header().capacity.load(Ordering::Acquire)
    }

    pub(crate) fn is(&self, metas: impl IntoIterator<Item = Meta>) -> bool {
        self.columns().iter().map(|column| column.meta()).eq(metas)
    }

    pub(crate) unsafe fn lock(&self, locks: impl Iterator<Item = (u32, Access)>) {
        let columns = self.columns();
        for (index, access) in locks {
            if let Some(column) = columns.get(index as usize) {
                unsafe { column.lock(access) };
            }
        }
    }

    pub(crate) unsafe fn lock_all(&self, access: Access) {
        for column in self.columns() {
            unsafe { column.lock(access) };
        }
    }

    pub(crate) unsafe fn unlock(&self, locks: impl Iterator<Item = (u32, Access)>) {
        let columns = self.columns();
        for (index, access) in locks {
            if let Some(column) = columns.get(index as usize) {
                unsafe { column.unlock(access) };
            }
        }
    }

    pub(crate) unsafe fn unlock_all(&self, access: Access) {
        for column in self.columns() {
            unsafe { column.unlock(access) };
        }
    }

    pub(crate) fn insert<F: FnOnce(u32)>(&self, count: u32, apply: F) -> Result<Range<u32>, Error> {
        if count == 0 {
            return Ok(0..0);
        }

        loop {
            unsafe { self.lock_all(Access::Read) };
            let guard = defer(|| unsafe { self.unlock_all(Access::Read) });
            let header = self.header();
            let start = header.count.load(Ordering::Acquire);
            let end = start.checked_add(count).ok_or(Error::TableOverflow)?;
            let capacity = header.capacity.load(Ordering::Acquire);
            if end <= capacity {
                let rows = start..end;
                apply(start);
                header.count.store(end, Ordering::Release);
                drop(guard);
                break Ok(rows);
            } else {
                drop(guard);
                let next = end
                    .checked_next_power_of_two()
                    .ok_or(Error::TableOverflow)?;
                self.grow((capacity, next))?;
            }
        }
    }

    pub(crate) fn remove(&self, rows: impl Iterator<Item = u32>) -> Result<(), Error> {
        unsafe { self.lock_all(Access::Write) };
        let guard = defer(|| unsafe { self.unlock_all(Access::Write) });
        let header = self.header();
        let columns = self.columns();
        let mut count = header.count.load(Ordering::Acquire);
        for row in rows {
            debug_assert!(row < count);
            count = count.checked_sub(1).ok_or(Error::TableUnderflow)?;
            for column in columns {
                unsafe { column.drop_at(row, 1) };
                if row < count {
                    unsafe { column.copy_at(count, row, 1) };
                }
            }
        }
        header.count.store(count, Ordering::Release);
        drop(guard);
        Ok(())
    }

    fn grow(&self, mut capacities: (u32, u32)) -> Result<u32, Error> {
        let header = self.header();
        let columns = self.columns();
        let mut new_layout = Layout::new::<()>();
        for column in columns {
            (new_layout, _) = column.meta.extend(new_layout, capacities.1)?;
        }

        while capacities.0 < capacities.1 {
            unsafe { self.lock_all(Access::Write) };
            let guard = defer(|| unsafe { self.unlock_all(Access::Write) });
            let new_data = match header.capacity.compare_exchange(
                capacities.0,
                capacities.1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => unsafe { allocate(new_layout.pad_to_align())? },
                Err(capacity) => {
                    capacities.0 = capacity;
                    continue;
                }
            };

            let mut old_layout = Layout::new::<()>();
            let mut new_layout = Layout::new::<()>();
            let mut old_data = NonNull::dangling();
            let count = header.count.load(Ordering::Acquire);
            for column in columns {
                let old_pair = column.meta.extend(old_layout, capacities.0)?;
                let new_pair = column.meta.extend(new_layout, capacities.1)?;
                let target = unsafe { new_data.add(new_pair.1) };
                let source = unsafe { replace(column.data.data_ptr(), target) };
                unsafe { column.meta.copy(source, target, count) };
                old_data = unsafe { source.sub(old_pair.1) };
                old_layout = old_pair.0;
                new_layout = new_pair.0;
            }
            drop(guard);
            unsafe { deallocate(old_data, old_layout.pad_to_align()) };
        }
        Ok(capacities.0)
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
        let _ = self.0.with_arc_mut(|table| {
            if let Some(table) = Arc::get_mut(table) {
                let header = table.header_mut();
                let count = *header.count.get_mut();
                let capacity = *header.capacity.get_mut();
                let mut root = NonNull::dangling();
                let mut layout = Layout::new::<()>();
                for column in table.slice_mut() {
                    let pair = column.meta.extend(layout, capacity)?;
                    let data = *column.data.get_mut();
                    unsafe { column.meta.drop(data, count) };
                    root = unsafe { data.sub(pair.1) };
                    layout = pair.0;
                }
                unsafe { deallocate(root, layout) };
                Ok::<_, Error>(true)
            } else {
                Ok::<_, Error>(false)
            }
        });
    }
}

fn sort<T: Ord>(items: impl IntoIterator<Item = T>) -> Option<Vec<T>> {
    let mut items = items.into_iter().collect::<Vec<_>>();
    items.sort_unstable();
    for [left, right] in items.array_windows::<2>() {
        if left == right {
            return None;
        }
    }
    Some(items)
}
