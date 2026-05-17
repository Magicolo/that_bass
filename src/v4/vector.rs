use core::ptr::NonNull;

use super::error::Error;
use super::meta::Meta;

pub struct Vector {
    meta: Meta,
    data: NonNull<u8>,
    count: u32,
    capacity: u32,
}

impl Vector {
    pub(crate) fn new(meta: Meta) -> Self {
        Vector {
            meta,
            data: NonNull::dangling(),
            count: 0,
            capacity: 0,
        }
    }

    pub(crate) fn push(&mut self, item: Box<dyn core::any::Any>) -> Result<(), Error> {
        if self.meta.identifier == item.type_id() {
            let index = self.count;
            self.reserve(1)?;
            unsafe { self.meta.set_at(self.data, item, index) };
            Ok(())
        } else {
            Err(Error::InvalidItem)
        }
    }

    pub(crate) unsafe fn move_at(&mut self, data: NonNull<u8>, index: u32) -> bool {
        let source = self.data;
        let target = unsafe { self.meta.offset(data, index) };
        let success = unsafe { self.meta.copy(source, target, self.count) };
        self.count = 0;
        success
    }

    fn reserve(&mut self, count: u32) -> Result<(), Error> {
        let old = self.count;
        let new = self.count.checked_add(count).ok_or(Error::VectorOverflow)?;
        self.count = new;

        if self.count > self.capacity {
            let capacity = self
                .count
                .checked_next_power_of_two()
                .ok_or(Error::VectorOverflow)?;
            self.data = self
                .meta
                .resize(self.data, old, (self.capacity, capacity))?;
            self.capacity = capacity;
            debug_assert!(self.count <= self.capacity);
        }
        Ok(())
    }
}

impl Drop for Vector {
    fn drop(&mut self) {
        let _ = self.meta.resize(self.data, self.count, (self.capacity, 0));
    }
}
