use super::{column::Column, error::Error, meta::Meta, utility::resize};
use crate::v4::{At, utility};
use core::{any::TypeId, ops::Range, ptr::NonNull};

pub struct Table {
    index: u32,
    pub(crate) count: u32,
    pending: u32,
    capacity: u32,
    data: NonNull<u8>,
    pub(crate) columns: Box<[Column]>,
}

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

    pub(super) fn column(&self, identifier: TypeId) -> Option<At<Column>> {
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

    pub(super) fn reserve(&mut self, count: u32) -> Result<Range<u32>, Error> {
        let old = self.pending;
        let new = self
            .pending
            .checked_add(count)
            .ok_or(Error::TableOverflow)?;
        self.pending = new;
        Ok(old..new)
    }

    pub(super) fn ensure(&mut self) -> Result<bool, Error> {
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

    pub(super) fn commit(&mut self) -> Range<u32> {
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
