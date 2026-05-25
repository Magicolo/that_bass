use crate::v4::{At, error::Error, meta::Meta, utility, utility::resize};
use core::{
    any::{Any, TypeId},
    iter::FusedIterator,
    marker::PhantomData,
    ops::Range,
    ptr::{NonNull, copy_nonoverlapping, slice_from_raw_parts_mut},
    slice::{from_raw_parts, from_raw_parts_mut},
};
pub struct Table {
    index: u32,
    pub(crate) count: u32,
    pending: u32,
    capacity: u32,
    data: NonNull<u8>,
    pub(crate) columns: Box<[Column]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Row<'a> {
    row: u32,
    table: u32,
    _marker: PhantomData<&'a ()>,
}

#[derive(Clone)]
pub struct Rows<'a> {
    rows: Range<u32>,
    table: u32,
    _marker: PhantomData<&'a ()>,
}

pub struct Column {
    pub(crate) meta: Meta,
    pub(crate) data: NonNull<u8>,
}

impl Column {
    pub(crate) const fn new(meta: Meta) -> Self {
        Self {
            meta,
            data: NonNull::dangling(),
        }
    }

    pub const fn meta(&self) -> &Meta {
        &self.meta
    }

    pub(crate) unsafe fn as_ref<T: 'static>(&self, count: u32) -> &[T] {
        debug_assert_eq!(self.meta.identifier, TypeId::of::<T>());
        unsafe { from_raw_parts(self.data.cast::<T>().as_ptr(), count as usize) }
    }

    pub(crate) unsafe fn as_mut<T: 'static>(&self, count: u32) -> &mut [T] {
        debug_assert_eq!(self.meta.identifier, TypeId::of::<T>());
        unsafe { from_raw_parts_mut(self.data.cast::<T>().as_ptr(), count as usize) }
    }

    pub(crate) unsafe fn set<T: 'static>(&self, item: T, row: u32) {
        debug_assert_eq!(self.meta.identifier, TypeId::of::<T>());
        unsafe { self.data.cast::<T>().add(row as usize).write(item) };
    }

    pub(crate) unsafe fn copy<T: 'static>(&self, source: NonNull<T>, row: u32, count: u32) -> bool {
        debug_assert_eq!(self.meta.identifier, TypeId::of::<T>());
        if size_of::<T>() > 0 && count > 0 {
            let target = unsafe { self.data.cast::<T>().add(row as usize) };
            unsafe { copy_nonoverlapping(source.as_ptr(), target.as_ptr(), count as usize) };
            true
        } else {
            false
        }
    }

    pub(crate) unsafe fn drop<T: 'static>(&self, row: u32, count: u32) {
        debug_assert_eq!(self.meta.identifier, TypeId::of::<T>());
        let data = unsafe { self.data.cast::<T>().add(row as usize) };
        unsafe { slice_from_raw_parts_mut(data.as_ptr(), count as usize).drop_in_place() };
    }

    pub(crate) unsafe fn get_with(&self, meta: Meta, row: u32) -> &dyn Any {
        unsafe { meta.get(meta.offset(self.data, row)) }
    }

    pub(crate) unsafe fn set_with(&self, item: Box<dyn Any>, row: u32, meta: Meta) -> bool {
        unsafe { meta.set(meta.offset(self.data, row), item) }
    }

    pub(crate) unsafe fn drop_with(&self, row: u32, count: u32, meta: Meta) -> bool {
        unsafe { meta.drop(meta.offset(self.data, row), count) }
    }
}

impl Row<'_> {
    pub(crate) const fn new(row: u32, table: u32) -> Self {
        Self {
            row,
            table,
            _marker: PhantomData,
        }
    }

    pub const fn row(&self) -> u32 {
        self.row
    }

    pub const fn table(&self) -> u32 {
        self.table
    }
}

impl Rows<'_> {
    pub(crate) const fn new(rows: Range<u32>, table: u32) -> Self {
        Self {
            rows,
            table,
            _marker: PhantomData,
        }
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
    pub(super) fn new(index: u32, metas: impl IntoIterator<Item = Meta>) -> Result<Self, Error> {
        Ok(Self {
            index,
            count: 0,
            pending: 0,
            capacity: 0,
            data: NonNull::dangling(),
            columns: metas.into_iter().map(Column::new).collect::<Box<[_]>>(),
        })
    }

    pub(super) fn column(&self, identifier: TypeId) -> Option<At<'_, Column>> {
        utility::find(&self.columns, identifier, |column| column.meta.identifier)
    }

    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    pub fn columns_mut(&mut self) -> &mut [Column] {
        &mut self.columns
    }

    pub const fn index(&self) -> u32 {
        self.index
    }

    pub const fn count(&self) -> u32 {
        self.count
    }

    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    pub(crate) fn reserve(&mut self, count: u32) -> Result<Range<u32>, Error> {
        let old = self.pending;
        let new = self
            .pending
            .checked_add(count)
            .ok_or(Error::TableOverflow)?;
        self.pending = new;
        Ok(old..new)
    }

    pub(crate) fn ensure(&mut self) -> Result<bool, Error> {
        if self.pending > self.capacity {
            let capacity = self
                .pending
                .checked_next_power_of_two()
                .ok_or(Error::TableOverflow)?;
            self.data = resize(
                &mut self.columns,
                self.data,
                self.count,
                (self.capacity, capacity),
            )?;
            self.capacity = capacity;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub(crate) fn commit(&mut self) -> Range<u32> {
        debug_assert!(self.count <= self.pending);
        let rows = self.count..self.pending;
        self.count = self.pending;
        rows
    }

    pub(super) fn release(&mut self, rows: Range<u32>) {
        if rows.is_empty() {
            return;
        }

        let count = rows.end.saturating_sub(rows.start);
        debug_assert!(rows.end <= self.pending);

        let copy = self.pending.saturating_sub(rows.end).min(count);
        let copy = (self.pending - copy, copy);
        for column in &mut self.columns {
            unsafe { column.meta.drop_at(column.data, rows.start, count) };
            unsafe {
                column
                    .meta
                    .copy_at((column.data, copy.0), (column.data, rows.start), copy.1)
            };
        }
        self.pending = self.pending.saturating_sub(count);
        self.count = self.count.saturating_sub(count).min(self.pending);
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        let _ = resize(&mut self.columns, self.data, self.count, (self.capacity, 0));
    }
}
