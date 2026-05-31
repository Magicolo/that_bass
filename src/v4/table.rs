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
    sync::atomic::{AtomicU32, AtomicU64, Ordering},
};
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use triomphe::{Arc, ThinArc};

#[derive(Debug, Clone)]
pub struct Table(ThinArc<Header, Column>);

#[derive(Clone)]
pub struct Tables(Arc<ArcSwapAny<ThinArc<(), Table>>>);

#[derive(Debug)]
struct Header {
    index: u32,
    count: AtomicU32,
    pending: AtomicU32,
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

    pub(crate) unsafe fn data(&self) -> NonNull<u8> {
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
                count: AtomicU32::new(0),
                pending: AtomicU32::new(0),
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

    fn pending(&self) -> u32 {
        self.header().pending.load(Ordering::Acquire)
    }

    pub(crate) fn reserve(&self, count: u32) -> Result<Range<u32>, Error> {
        let header = self.header();
        let mut new = 0;
        let old = header
            .pending
            .try_update(Ordering::AcqRel, Ordering::Acquire, |pending| {
                new = pending.checked_add(count)?;
                Some(new)
            })
            .map_err(|_| Error::TableOverflow)?;

        let mut capacity = header.capacity.load(Ordering::Acquire);
        // This loop is important because `self.resize` may lose the resizing race and
        // another thread could've resized to a smaller capacity than `new`; in that
        // case, we must resize again.
        while new > capacity {
            capacity = self.resize((capacity, new.checked_next_power_of_two().unwrap_or(new)))?;
        }
        Ok(old..new)
    }

    pub(crate) fn resize(&self, capacities: (u32, u32)) -> Result<u32, Error> {
        struct Hit {
            old: (NonNull<u8>, Layout),
            new: NonNull<u8>,
            count: u32,
            capacity: u32,
        }

        struct Miss {
            capacity: u32,
        }

        enum Next {
            Hit(Hit),
            Miss(Miss),
        }

        fn next(
            header: &Header,
            columns: &[Column],
            layouts: (Layout, Layout),
            capacities: (u32, u32),
        ) -> Result<Next, Error> {
            match columns.split_first() {
                Some((head, tail)) if head.meta.size() == 0 => {
                    next(header, tail, layouts, capacities)
                }
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
                        Next::Hit(hit) => {
                            let source = *data;
                            let target = unsafe { hit.new.add(new.1) };
                            head.meta
                                .initialize(source, target, hit.count, capacities.1);
                            *data = target;
                            Ok(Next::Hit(Hit {
                                old: (unsafe { source.sub(old.1) }, hit.old.1),
                                ..hit
                            }))
                        }
                        Next::Miss(miss) => Ok(Next::Miss(miss)),
                    }
                }
                None if layouts.1.size() == 0 => Ok(Next::Hit(Hit {
                    old: (NonNull::dangling(), layouts.0.pad_to_align()),
                    new: NonNull::dangling(),
                    count: header.count.load(Ordering::Acquire),
                    capacity: header.capacity.load(Ordering::Acquire),
                })),
                None => {
                    match header.capacity.compare_exchange(
                        capacities.0,
                        capacities.1,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(capacity) if layouts.1.size() == 0 => Ok(Next::Hit(Hit {
                            old: (NonNull::dangling(), layouts.0.pad_to_align()),
                            new: NonNull::dangling(),
                            count: header.count.load(Ordering::Acquire),
                            capacity,
                        })),
                        Ok(capacity) => Ok(Next::Hit(Hit {
                            old: (NonNull::dangling(), layouts.0.pad_to_align()),
                            new: unsafe { allocate(layouts.1.pad_to_align())? },
                            count: header.count.load(Ordering::Acquire),
                            capacity,
                        })),
                        Err(capacity) => Ok(Next::Miss(Miss { capacity })),
                    }
                }
            }
        }

        match next(
            self.header(),
            self.columns(),
            (Layout::new::<()>(), Layout::new::<()>()),
            capacities,
        )? {
            Next::Hit(hit) => {
                unsafe { deallocate(hit.old.0, hit.old.1) };
                Ok(hit.capacity)
            }
            Next::Miss(miss) => Ok(miss.capacity),
        }
    }

    pub(crate) fn commit(&self) -> Range<u32> {
        debug_assert!(self.count() <= self.pending());
        let header = self.header();
        let pending = header.pending.load(Ordering::Acquire);
        let count = header.count.swap(pending, Ordering::AcqRel);
        count..pending
    }

    pub(super) fn release(&self, rows: Range<u32>) {
        if rows.is_empty() {
            return;
        }

        let count = rows.end.saturating_sub(rows.start);
        debug_assert!(rows.end <= self.pending());

        let header = self.header();
        let copy = header.pending.saturating_sub(rows.end).min(count);
        let copy = (header.pending - copy, copy);
        for column in &self.0.slice {
            let data = column.data();
            unsafe { column.meta.drop_at(data, rows.start, count) };
            unsafe {
                column
                    .meta
                    .copy_at((data, copy.0), (data, rows.start), copy.1)
            };
        }
        header.pending = header.pending.saturating_sub(count);
        header.count = self
            .header()
            .count
            .saturating_sub(count)
            .min(header.pending);
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
        if self.0.with_arc(Arc::is_unique) {
            let _ = self.resize((self.capacity(), 0));
        }
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
