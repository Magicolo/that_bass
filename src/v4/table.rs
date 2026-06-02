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
    ptr::NonNull,
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

impl Default for Tables {
    fn default() -> Self {
        Self::new()
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

    pub(crate) unsafe fn get<T: 'static>(&self, count: u32) -> &[T] {
        debug_assert_eq!(self.meta.identifier(), TypeId::of::<T>());
        unsafe { from_raw_parts(self.data().cast::<T>().as_ptr(), count as usize) }
    }

    pub(crate) unsafe fn get_mut<T: 'static>(&self, count: u32) -> &mut [T] {
        debug_assert_eq!(self.meta.identifier(), TypeId::of::<T>());
        unsafe { from_raw_parts_mut(self.data().cast::<T>().as_ptr(), count as usize) }
    }

    pub(crate) unsafe fn get_in(&self, slice: &mut Slice, count: u32) {
        debug_assert_eq!(self.meta, slice.meta());
        unsafe { slice.set_parts(self.data(), count as _) };
    }

    pub(crate) unsafe fn set<T: 'static>(&self, item: T, row: u32) {
        debug_assert_eq!(self.meta.identifier(), TypeId::of::<T>());
        unsafe { self.data().cast::<T>().add(row as usize).write(item) };
    }

    pub(crate) unsafe fn set_at(&self, item: Box<dyn Any>, row: u32) {
        debug_assert_eq!(self.meta.identifier(), item.type_id());
        unsafe { self.meta.set_at(self.data(), item, row) };
    }

    pub(crate) unsafe fn copy_at(&self, source: u32, target: u32, count: u32) -> bool {
        let data = unsafe { self.data() };
        unsafe { self.meta.copy_at((data, source), (data, target), count) }
    }

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
        enum Next {
            Done {
                old: (NonNull<u8>, Layout),
                new: NonNull<u8>,
                count: u32,
                capacity: u32,
            },
            Retry(u32),
        }

        fn next(
            header: &Header,
            columns: &[Column],
            layouts: (Layout, Layout),
            capacities: (u32, u32),
        ) -> Result<Next, Error> {
            match columns.split_first() {
                Some((head, tail)) => {
                    let old = head
                        .meta
                        .extend(layouts.0, capacities.0)
                        .map_err(Error::Layout)?;
                    let new = head
                        .meta
                        .extend(layouts.1, capacities.1)
                        .map_err(Error::Layout)?;
                    let mut data = head.data.write();
                    match next(header, tail, (old.0, new.0), capacities)? {
                        Next::Done {
                            old: done_old,
                            new: done_new,
                            count: done_count,
                            capacity,
                        } => {
                            let source = *data;
                            let target = unsafe { done_new.add(new.1) };
                            unsafe { head.meta.copy(source, target, done_count) };
                            *data = target;
                            Ok(Next::Done {
                                old: (unsafe { source.sub(old.1) }, done_old.1),
                                new: done_new,
                                count: done_count,
                                capacity,
                            })
                        }
                        slow => Ok(slow),
                    }
                }
                None => {
                    match header.capacity.compare_exchange(
                        capacities.0,
                        capacities.1,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(capacity) => Ok(Next::Done {
                            old: (NonNull::dangling(), layouts.0.pad_to_align()),
                            new: unsafe { allocate(layouts.1.pad_to_align())? },
                            count: header.count.load(Ordering::Acquire),
                            capacity,
                        }),
                        Err(capacity) => Ok(Next::Retry(capacity)),
                    }
                }
            }
        }

        while capacities.0 > capacities.1 {
            capacities.0 = match next(
                self.header(),
                self.columns(),
                (Layout::new::<()>(), Layout::new::<()>()),
                capacities,
            )? {
                Next::Done { old, capacity, .. } => {
                    unsafe { deallocate(old.0, old.1) };
                    capacity
                }
                Next::Retry(old) => old,
            };
        }
        Ok(capacities.0)
    }

    fn resize_mut(
        columns: &mut [Column],
        count: u32,
        capacities: (u32, u32),
    ) -> Result<bool, Error> {
        struct Next {
            old: (NonNull<u8>, Layout),
            new: NonNull<u8>,
        }

        fn next(
            columns: &mut [Column],
            layouts: (Layout, Layout),
            count: u32,
            capacities: (u32, u32),
        ) -> Result<Next, Error> {
            match columns.split_first_mut() {
                Some((head, tail)) => {
                    let old = head
                        .meta
                        .extend(layouts.0, capacities.0)
                        .map_err(Error::Layout)?;
                    let new = head
                        .meta
                        .extend(layouts.1, capacities.1)
                        .map_err(Error::Layout)?;
                    let Next {
                        old: done_old,
                        new: done_new,
                    } = next(tail, (old.0, new.0), count, capacities)?;
                    let data = head.data.get_mut();
                    let source = *data;
                    let target = unsafe { done_new.add(new.1) };
                    unsafe { head.meta.initialize(source, target, count, capacities.1) };
                    *data = target;
                    Ok(Next {
                        old: (unsafe { source.sub(old.1) }, done_old.1),
                        new: done_new,
                    })
                }
                None => Ok(Next {
                    old: (NonNull::dangling(), layouts.0.pad_to_align()),
                    new: unsafe { allocate(layouts.1.pad_to_align())? },
                }),
            }
        }

        if capacities.0 == capacities.1 {
            Ok(false)
        } else {
            let Next { old, .. } = next(
                columns,
                (Layout::new::<()>(), Layout::new::<()>()),
                count,
                capacities,
            )?;
            unsafe { deallocate(old.0, old.1) };
            Ok(true)
        }
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
        self.0.with_arc_mut(|table| {
            if let Some(table) = Arc::get_mut(table) {
                let header = table.header_mut();
                let count = *header.count.get_mut();
                let capacity = *header.capacity.get_mut();
                if let Ok(true) = Self::resize_mut(table.slice_mut(), count, (capacity, 0)) {
                    *table.header_mut().capacity.get_mut() = 0;
                }
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
