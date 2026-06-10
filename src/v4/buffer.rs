use crate::v4::{Error, Meta, table::allocate, utility::deallocate};
use core::{alloc::Layout, any::Any, ptr::NonNull};

pub struct Buffer {
    pub(crate) count: u32,
    pub(crate) capacity: u32,
    pub(crate) columns: Box<[Column]>,
}

pub(crate) struct Column {
    meta: Meta,
    data: NonNull<u8>,
}

impl Column {
    pub(crate) const fn new(meta: Meta) -> Self {
        Self {
            meta,
            data: NonNull::dangling(),
        }
    }

    pub(crate) const fn meta(&self) -> Meta {
        self.meta
    }

    pub(crate) const fn data(&self) -> NonNull<u8> {
        self.data
    }
}

impl Buffer {
    pub(crate) fn new(metas: impl IntoIterator<Item = Meta>) -> Self {
        Self {
            count: 0,
            capacity: 0,
            columns: metas.into_iter().map(Column::new).collect(),
        }
    }

    #[inline]
    pub(crate) unsafe fn set<T: 'static>(&mut self, column: u32, item: T) {
        let column = unsafe { self.columns.get_unchecked_mut(column as usize) };
        debug_assert_eq!(column.meta.identifier(), item.type_id());
        debug_assert!(self.count < self.capacity);
        unsafe { column.data.cast::<T>().add(self.count as usize).write(item) };
    }

    #[inline]
    pub(crate) unsafe fn set_with(&mut self, column: u32, item: Box<dyn Any>) {
        let column = unsafe { self.columns.get_unchecked_mut(column as usize) };
        debug_assert_eq!(column.meta.identifier(), item.type_id());
        debug_assert!(self.count < self.capacity);
        unsafe { column.meta.set_at(column.data, item, self.count) };
    }

    #[inline]
    pub(crate) const unsafe fn commit(&mut self) {
        debug_assert!(self.count < self.capacity);
        self.count += 1;
    }

    pub(crate) fn reserve(&mut self, count: u32) -> Result<(), Error> {
        let count = self.count.checked_add(count).ok_or(Error::BufferOverflow)?;
        if count <= self.capacity {
            return Ok(());
        }

        let capacity = count
            .checked_next_power_of_two()
            .ok_or(Error::TableOverflow)?;

        let new_data = unsafe { allocate(self.columns.iter().map(Column::meta), capacity)? };
        let mut old_layout = Layout::new::<()>();
        let mut new_layout = Layout::new::<()>();
        let mut old_data = NonNull::dangling();
        for Column { meta, data: source } in self.columns.iter_mut() {
            let old_pair = meta.extend(old_layout, self.capacity)?;
            let new_pair = meta.extend(new_layout, capacity)?;
            let target = unsafe { new_data.add(new_pair.1) };
            unsafe { meta.copy(*source, target, self.count) };
            *source = target;
            old_data = unsafe { source.sub(old_pair.1) };
            old_layout = old_pair.0;
            new_layout = new_pair.0;
        }
        unsafe { deallocate(old_data, old_layout.pad_to_align()) };
        self.capacity = capacity;
        Ok(())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let count = self.count;
        let capacity = self.capacity;
        let mut old_data = NonNull::dangling();
        let mut old_layout = Layout::new::<()>();
        for Column { meta, data } in self.columns.iter_mut() {
            if let Ok(pair) = meta.extend(old_layout, capacity) {
                unsafe { meta.drop(*data, count) };
                old_data = unsafe { data.sub(pair.1) };
                old_layout = pair.0;
            }
        }
        unsafe { deallocate(old_data, old_layout.pad_to_align()) };
    }
}
