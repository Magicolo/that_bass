use core::{any::TypeId, ptr::NonNull, slice};

pub struct Slice {
    data: NonNull<u8>,
    len: usize,
    type_id: TypeId,
}

impl Slice {
    pub const fn empty(type_id: TypeId) -> Self {
        unsafe { Self::from_raw_parts(NonNull::dangling(), 0, type_id) }
    }

    pub const unsafe fn from_raw_parts(data: NonNull<()>, len: usize, type_id: TypeId) -> Self {
        Self {
            data: data.cast(),
            len,
            type_id,
        }
    }

    pub const fn type_id(&self) -> TypeId {
        self.type_id
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub fn downcast_ref<T: 'static>(&self) -> Option<&[T]> {
        Some(unsafe { slice::from_raw_parts(self.cast::<T>()?.as_ptr(), self.len) })
    }

    pub fn downcast_mut<T: 'static>(&mut self) -> Option<&mut [T]> {
        Some(unsafe { slice::from_raw_parts_mut(self.cast::<T>()?.as_ptr(), self.len) })
    }

    fn cast<T: 'static>(&self) -> Option<NonNull<T>> {
        if self.type_id == TypeId::of::<T>() {
            Some(self.data.cast())
        } else {
            None
        }
    }
}
