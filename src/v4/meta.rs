use super::{
    error::Error,
    utility::{allocate, deallocate},
};
use core::{
    alloc::{Layout, LayoutError},
    any::{Any, TypeId},
    mem::needs_drop,
    ptr::{NonNull, slice_from_raw_parts_mut},
};

struct Functions {
    layout: fn(u32) -> Result<Layout, LayoutError>,
    drop: unsafe fn(NonNull<u8>, u32),
    get: unsafe fn(NonNull<u8>) -> &'static dyn Any,
    set: unsafe fn(Box<dyn Any>, NonNull<u8>),
}

#[derive(Clone)]
pub struct Meta {
    pub(crate) identifier: TypeId,
    pub(crate) size: usize,
    pub(crate) drop: bool,
    functions: &'static Functions,
}

impl Meta {
    pub fn of<T: 'static>() -> Self {
        Self {
            identifier: TypeId::of::<T>(),
            size: size_of::<T>(),
            drop: needs_drop::<T>(),
            functions: &Functions {
                layout: |count| Layout::array::<T>(count as usize),
                drop: |data, count| unsafe {
                    slice_from_raw_parts_mut(data.cast::<T>().as_ptr(), count as usize)
                        .drop_in_place();
                },
                get: |data| unsafe { data.cast::<T>().as_ref() },
                set: |item, data| {
                    let item = unsafe { item.downcast::<T>().unwrap_unchecked() };
                    unsafe { data.cast::<T>().write(*item) };
                },
            },
        }
    }

    pub(crate) fn layout(&self, count: u32) -> Result<Layout, LayoutError> {
        (self.functions.layout)(count)
    }

    pub(crate) fn extend(
        &self,
        layout: Layout,
        count: u32,
    ) -> Result<(Layout, usize), LayoutError> {
        layout.extend(self.layout(count)?)
    }

    pub(crate) fn initialize(
        &self,
        source: NonNull<u8>,
        target: NonNull<u8>,
        count: u32,
        capacity: u32,
    ) {
        unsafe { self.copy(source, target, core::cmp::min(count, capacity)) };
        unsafe { self.drop_at(source, count, count.saturating_sub(capacity)) };
    }

    pub(crate) fn resize(
        &self,
        data: NonNull<u8>,
        count: u32,
        capacities: (u32, u32),
    ) -> Result<NonNull<u8>, Error> {
        let layouts = (
            self.layout(capacities.0).map_err(Error::Layout)?,
            self.layout(capacities.1).map_err(Error::Layout)?,
        );
        let source = data;
        let target = unsafe { allocate(layouts.1)? };
        self.initialize(source, target, count, capacities.1);
        unsafe { deallocate(source, layouts.0) };
        Ok(target)
    }

    pub(crate) unsafe fn offset(&self, data: NonNull<u8>, count: u32) -> NonNull<u8> {
        unsafe { data.add(self.size * count as usize) }
    }

    pub(crate) unsafe fn copy(&self, source: NonNull<u8>, target: NonNull<u8>, count: u32) -> bool {
        let count = self.size * count as usize;
        if count > 0 {
            unsafe { core::ptr::copy_nonoverlapping(source.as_ptr(), target.as_ptr(), count) };
            true
        } else {
            false
        }
    }

    pub(crate) unsafe fn copy_at(
        &self,
        source: (NonNull<u8>, u32),
        target: (NonNull<u8>, u32),
        count: u32,
    ) -> bool {
        unsafe {
            self.copy(
                self.offset(source.0, source.1),
                self.offset(target.0, target.1),
                count,
            )
        }
    }

    pub(crate) unsafe fn drop(&self, data: NonNull<u8>, count: u32) -> bool {
        if self.drop {
            unsafe { (self.functions.drop)(data, count) };
            true
        } else {
            false
        }
    }

    pub(crate) unsafe fn drop_at(&self, data: NonNull<u8>, index: u32, count: u32) -> bool {
        unsafe { self.drop(self.offset(data, index), count) }
    }

    pub(crate) unsafe fn get<'a>(&self, data: NonNull<u8>) -> &'a dyn Any {
        unsafe { (self.functions.get)(data) }
    }

    pub(crate) unsafe fn get_at<'a>(&self, data: NonNull<u8>, index: u32) -> &'a dyn Any {
        unsafe { (self.functions.get)(self.offset(data, index)) }
    }

    pub(crate) unsafe fn set(&self, data: NonNull<u8>, value: Box<dyn Any>) -> bool {
        if self.identifier == (*value).type_id() {
            unsafe { (self.functions.set)(value, data) };
            true
        } else {
            false
        }
    }

    pub(crate) unsafe fn set_at(&self, data: NonNull<u8>, value: Box<dyn Any>, index: u32) -> bool {
        if self.identifier == (*value).type_id() {
            unsafe { (self.functions.set)(value, self.offset(data, index)) };
            true
        } else {
            false
        }
    }
}
