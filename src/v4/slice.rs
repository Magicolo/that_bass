use crate::v4::Meta;
use core::{
    any::{Any, TypeId},
    marker::PhantomData,
    ptr::NonNull,
    slice,
};
use derive_more::{Deref, DerefMut, From};

#[derive(Deref, From)]
pub struct Read<'a>(Slice<'a>);
#[derive(Deref, DerefMut, From)]
pub struct Write<'a>(Slice<'a>);

pub(crate) struct Slice<'a> {
    data: NonNull<u8>,
    len: usize,
    meta: Meta,
    _marker: PhantomData<&'a ()>,
}

impl Slice<'_> {
    pub(crate) const unsafe fn from_raw_parts(meta: Meta, data: NonNull<u8>, len: usize) -> Self {
        Self {
            data,
            len,
            meta,
            _marker: PhantomData,
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
