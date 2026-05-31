use crate::v4::{
    error::Error,
    guard::{Bind, Raw, Read, Write},
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
    sync::atomic::{AtomicU32, Ordering},
};
use parking_lot::RwLock;
use triomphe::{Arc, ThinArc};

#[derive(Debug, Clone)]
pub struct Table(ThinArc<Header, Column>);

#[derive(Clone)]
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

    pub(crate) unsafe fn read<T: 'static>(&self) -> Read<'_, T, Raw> {
        debug_assert_eq!(self.meta.identifier(), TypeId::of::<T>());
        Read::new(self.data.read())
    }

    pub(crate) unsafe fn try_read<T: 'static>(&self) -> Option<Read<'_, T, Raw>> {
        debug_assert_eq!(self.meta.identifier(), TypeId::of::<T>());
        Some(Read::new(self.data.try_read()?))
    }

    pub(crate) unsafe fn write<T: 'static>(&self) -> Write<'_, T, Raw> {
        debug_assert_eq!(self.meta.identifier(), TypeId::of::<T>());
        Write::new(self.data.write())
    }

    pub(crate) unsafe fn try_write<T: 'static>(&self) -> Option<Write<'_, T, Raw>> {
        debug_assert_eq!(self.meta.identifier(), TypeId::of::<T>());
        Some(Write::new(self.data.try_write()?))
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

impl<'a> Bind for Rows<'a> {
    type Guard = Self;

    fn bind(self, count: u32) -> Self::Guard {
        Self::new(0..count, self.table)
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

    pub(crate) fn insert<F: FnOnce(Range<u32>)>(
        &self,
        count: u32,
        mut resolve: F,
    ) -> Result<Rows<'_>, Error> {
        enum Next<F> {
            Done(Range<u32>),
            Grow(u32, u32, F),
        }

        fn next<F: FnOnce(Range<u32>)>(
            header: &Header,
            columns: &[Column],
            count: u32,
            resolve: F,
        ) -> Result<Next<F>, Error> {
            match columns.split_first() {
                Some((head, tail)) if head.meta.size() == 0 => next(header, tail, count, resolve),
                Some((head, tail)) => {
                    let data = head.data.read();
                    let result = next(header, tail, count, resolve);
                    drop(data);
                    result
                }
                None => {
                    let start = header.count.load(Ordering::Acquire);
                    let end = start.checked_add(count).ok_or(Error::TableOverflow)?;
                    let capacity = header.capacity.load(Ordering::Acquire);
                    if end <= capacity {
                        let rows = start..end;
                        resolve(rows.clone());
                        header.count.store(end, Ordering::Release);
                        Ok(Next::Done(rows))
                    } else {
                        let next = end
                            .checked_next_power_of_two()
                            .ok_or(Error::TableOverflow)?;
                        Ok(Next::Grow(capacity, next, resolve))
                    }
                }
            }
        }

        let header = self.header();
        let columns = self.columns();
        let rows = loop {
            resolve = match next(header, columns, count, resolve)? {
                Next::Grow(old, new, resolve) => {
                    self.resize((old, new))?;
                    resolve
                }
                Next::Done(rows) => break rows,
            };
        };
        Ok(Rows::new(rows, self))
    }

    // pub(crate) fn resize(&self, capacities: (u32, u32)) -> Result<u32, Error> {
    //     struct Hit {
    //         old: (NonNull<u8>, Layout),
    //         new: NonNull<u8>,
    //         count: u32,
    //         capacity: u32,
    //     }

    //     struct Miss {
    //         capacity: u32,
    //     }

    //     enum Next {
    //         Hit(Hit),
    //         Miss(Miss),
    //     }

    //     fn next(
    //         header: &Header,
    //         columns: &[Column],
    //         layouts: (Layout, Layout),
    //         capacities: (u32, u32),
    //     ) -> Result<Next, Error> {
    //         match columns.split_first() {
    //             Some((head, tail)) => {
    //                 let old = head
    //                     .meta
    //                     .extend(layouts.0, capacities.0)
    //                     .map_err(Error::Layout)?;
    //                 let new = head
    //                     .meta
    //                     .extend(layouts.1, capacities.1)
    //                     .map_err(Error::Layout)?;
    //                 let mut data = head.data.write();
    //                 match next(header, tail, (old.0, new.0), capacities)? {
    //                     Next::Hit(hit) => {
    //                         let source = *data;
    //                         let target = unsafe { hit.new.add(new.1) };
    //                         unsafe {
    //                             head.meta
    //                                 .initialize(source, target, hit.count,
    // capacities.1)                         };
    //                         *data = target;
    //                         Ok(Next::Hit(Hit {
    //                             old: (unsafe { source.sub(old.1) }, hit.old.1),
    //                             ..hit
    //                         }))
    //                     }
    //                     Next::Miss(miss) => Ok(Next::Miss(miss)),
    //                 }
    //             }
    //             None if layouts.1.size() == 0 => Ok(Next::Hit(Hit {
    //                 old: (NonNull::dangling(), layouts.0.pad_to_align()),
    //                 new: NonNull::dangling(),
    //                 count: header.count.load(Ordering::Acquire),
    //                 capacity: header.capacity.load(Ordering::Acquire),
    //             })),
    //             None => {
    //                 match header.capacity.compare_exchange(
    //                     capacities.0,
    //                     capacities.1,
    //                     Ordering::AcqRel,
    //                     Ordering::Acquire,
    //                 ) {
    //                     Ok(capacity) if layouts.1.size() == 0 => Ok(Next::Hit(Hit
    // {                         old: (NonNull::dangling(),
    // layouts.0.pad_to_align()),                         new:
    // NonNull::dangling(),                         count:
    // header.count.load(Ordering::Acquire),                         capacity,
    //                     })),
    //                     Ok(capacity) => Ok(Next::Hit(Hit {
    //                         old: (NonNull::dangling(), layouts.0.pad_to_align()),
    //                         new: unsafe { allocate(layouts.1.pad_to_align())? },
    //                         count: header.count.load(Ordering::Acquire),
    //                         capacity,
    //                     })),
    //                     Err(capacity) => Ok(Next::Miss(Miss { capacity })),
    //                 }
    //             }
    //         }
    //     }

    //     match next(
    //         self.header(),
    //         self.columns(),
    //         (Layout::new::<()>(), Layout::new::<()>()),
    //         capacities,
    //     )? {
    //         Next::Hit(hit) => {
    //             unsafe { deallocate(hit.old.0, hit.old.1) };
    //             Ok(hit.capacity)
    //         }
    //         Next::Miss(miss) => Ok(miss.capacity),
    //     }
    // }

    pub(crate) fn remove(&self, rows: impl Iterator<Item = u32>) -> Result<(), Error> {
        fn next(
            header: &Header,
            columns: (&[Column], &[Column]),
            rows: impl Iterator<Item = u32>,
        ) -> Result<(), Error> {
            match columns.0.split_first() {
                Some((head, tail)) if head.meta.size() == 0 => {
                    next(header, (tail, columns.1), rows)
                }
                Some((head, tail)) => {
                    let data = head.data.write();
                    let result = next(header, (tail, columns.1), rows);
                    drop(data);
                    result
                }
                None => {
                    let mut count = header.count.load(Ordering::Acquire);
                    for row in rows {
                        debug_assert!(row < count);
                        count = count.checked_sub(1).ok_or(Error::TableUnderflow)?;
                        for column in columns.1 {
                            unsafe { column.drop_at(row, 1) };
                            if row < count {
                                unsafe { column.copy_at(count, row, 1) };
                            }
                        }
                    }
                    header.count.store(count, Ordering::Release);
                    Ok(())
                }
            }
        }

        let header = self.header();
        let columns = self.columns();
        next(header, (columns, columns), rows)
    }

    fn resize(&self, mut capacities: (u32, u32)) -> Result<u32, Error> {
        enum Next {
            Done {
                old: (NonNull<u8>, Layout),
                new: NonNull<u8>,
                count: u32,
                capacity: u32,
            },
            Skip(u32),
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
                            unsafe { head.meta.initialize(source, target, done_count, capacity) };
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
                        Err(capacity) if capacity < capacities.1 => Ok(Next::Retry(capacity)),
                        Err(capacity) => Ok(Next::Skip(capacity)),
                    }
                }
            }
        }

        loop {
            capacities.0 = match next(
                self.header(),
                self.columns(),
                (Layout::new::<()>(), Layout::new::<()>()),
                capacities,
            )? {
                Next::Done { old, capacity, .. } => {
                    unsafe { deallocate(old.0, old.1) };
                    break Ok(capacity);
                }
                Next::Skip(capacity) => break Ok(capacity),
                Next::Retry(old) => old,
            };
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
        if self.0.with_arc(Arc::is_unique) {
            let _ = self.resize((self.capacity(), 0));
        }
    }
}

impl Bind for &Table {
    type Guard = Self;

    fn bind(self, _: u32) -> Self::Guard {
        self
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
