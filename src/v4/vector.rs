use super::{error::Error, meta::Meta};
use core::ptr::NonNull;

pub struct Vector {
    meta: Meta,
    data: NonNull<u8>,
    len: u32,
    cap: u32,
}

impl Vector {
    pub const fn new(meta: Meta) -> Self {
        Vector {
            meta,
            data: NonNull::dangling(),
            len: 0,
            cap: 0,
        }
    }

    pub const fn len(&self) -> u32 {
        self.len
    }

    pub const fn capacity(&self) -> u32 {
        self.cap
    }

    pub fn push(&mut self, item: Box<dyn core::any::Any>) -> Result<(), Error> {
        if self.meta.identifier() == item.type_id() {
            let index = self.len;
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
        let success = unsafe { self.meta.copy(source, target, self.len) };
        self.len = 0;
        success
    }

    fn reserve(&mut self, count: u32) -> Result<(), Error> {
        let old = self.len;
        let new = self.len.checked_add(count).ok_or(Error::VectorOverflow)?;
        self.len = new;

        if self.len > self.cap {
            let capacity = self
                .len
                .checked_next_power_of_two()
                .ok_or(Error::VectorOverflow)?;
            self.data = self.meta.resize(self.data, old, (self.cap, capacity))?;
            self.cap = capacity;
            debug_assert!(self.len <= self.cap);
        }
        Ok(())
    }
}

impl Drop for Vector {
    fn drop(&mut self) {
        let _ = self.meta.resize(self.data, self.len, (self.cap, 0));
    }
}
