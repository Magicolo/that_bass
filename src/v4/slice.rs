use crate::v4::Meta;
use core::{
    any::{Any, TypeId},
    ptr::NonNull,
    slice,
};

pub struct Slice {
    data: NonNull<u8>,
    len: usize,
    meta: Meta,
}

impl Slice {
    pub const fn empty(meta: Meta) -> Self {
        Self {
            data: NonNull::dangling(),
            len: 0,
            meta,
        }
    }

    pub const fn meta(&self) -> Meta {
        self.meta
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub fn get(&self, index: usize) -> Option<&dyn Any> {
        if index < self.len {
            Some(unsafe { self.meta.get_at(self.data, index.try_into().ok()?) })
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut dyn Any> {
        if index < self.len {
            Some(unsafe { self.meta.get_mut_at(self.data, index.try_into().ok()?) })
        } else {
            None
        }
    }

    pub(crate) const unsafe fn set_parts(&mut self, data: NonNull<u8>, len: usize) {
        self.data = data;
        self.len = len;
    }

    pub fn downcast_ref<T: 'static>(&self) -> Option<&[T]> {
        Some(unsafe { slice::from_raw_parts(self.cast::<T>()?.as_ptr(), self.len) })
    }

    pub fn downcast_mut<T: 'static>(&mut self) -> Option<&mut [T]> {
        Some(unsafe { slice::from_raw_parts_mut(self.cast::<T>()?.as_ptr(), self.len) })
    }

    fn cast<T: 'static>(&self) -> Option<NonNull<T>> {
        if self.meta.identifier() == TypeId::of::<T>() {
            Some(self.data.cast())
        } else {
            None
        }
    }
}
