use super::meta::Meta;
use core::{
    any::{Any, TypeId},
    ptr::{NonNull, copy_nonoverlapping, slice_from_raw_parts_mut},
    slice::{from_raw_parts, from_raw_parts_mut},
};

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
