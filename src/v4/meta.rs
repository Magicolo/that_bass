use super::{
    error::Error,
    utility::{allocate, deallocate},
};
use core::{
    alloc::{Layout, LayoutError},
    any::{Any, TypeId, type_name},
    hash::Hash,
    mem::needs_drop,
    ptr::{NonNull, copy_nonoverlapping, slice_from_raw_parts_mut},
};
use parking_lot::Mutex;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy)]
pub struct Meta(&'static Inner);

#[derive(Debug)]
struct Inner {
    identifier: TypeId,
    size: usize,
    name: &'static str,
    layout: fn(u32) -> Result<Layout, LayoutError>,
    drop: Option<unsafe fn(NonNull<u8>, u32)>,
    get: unsafe fn(NonNull<u8>) -> &'static dyn Any,
    get_mut: unsafe fn(NonNull<u8>) -> &'static mut dyn Any,
    set: unsafe fn(Box<dyn Any>, NonNull<u8>),
}

static METAS: Mutex<BTreeMap<TypeId, &'static Inner>> = Mutex::new(BTreeMap::new());

impl Meta {
    pub fn of<T: 'static>() -> Self {
        let key = TypeId::of::<T>();
        let mut guard = METAS.lock();
        Self(*guard.entry(key).or_insert_with(|| {
            Box::leak(Box::new(Inner {
                identifier: TypeId::of::<T>(),
                size: size_of::<T>(),
                name: type_name::<T>(),
                layout: |count| Layout::array::<T>(count as usize),
                drop: if needs_drop::<T>() {
                    Some(|data, count| unsafe {
                        slice_from_raw_parts_mut(data.cast::<T>().as_ptr(), count as usize)
                            .drop_in_place();
                    })
                } else {
                    None
                },
                get: |data| unsafe { data.cast::<T>().as_ref() },
                get_mut: |data| unsafe { data.cast::<T>().as_mut() },
                set: |item, data| {
                    let item = unsafe { item.downcast::<T>().unwrap_unchecked() };
                    unsafe { data.cast::<T>().write(*item) };
                },
            }))
        }))
    }

    #[inline]
    pub const fn identifier(self) -> TypeId {
        self.0.identifier
    }

    #[inline]
    pub const fn size(self) -> usize {
        self.0.size
    }

    #[inline]
    pub const fn name(self) -> &'static str {
        self.0.name
    }

    #[inline]
    pub const fn drops(self) -> bool {
        self.0.drop.is_some()
    }

    #[inline]
    pub(crate) fn layout(self, count: u32) -> Result<Layout, LayoutError> {
        (self.0.layout)(count)
    }

    #[inline]
    pub(crate) fn extend(self, layout: Layout, count: u32) -> Result<(Layout, usize), LayoutError> {
        layout.extend(self.layout(count)?)
    }

    #[inline]
    pub(crate) unsafe fn initialize(
        self,
        source: NonNull<u8>,
        target: NonNull<u8>,
        count: u32,
        capacity: u32,
    ) {
        unsafe { self.copy(source, target, count.min(capacity)) };
        unsafe { self.drop_at(source, count, count.saturating_sub(capacity)) };
    }

    #[inline]
    pub(crate) fn resize(
        self,
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
        unsafe { self.initialize(source, target, count, capacities.1) };
        unsafe { deallocate(source, layouts.0) };
        Ok(target)
    }

    #[inline]
    pub(crate) const unsafe fn offset(self, data: NonNull<u8>, count: u32) -> NonNull<u8> {
        unsafe { data.add(self.0.size * count as usize) }
    }

    #[inline]
    pub(crate) const unsafe fn copy(
        self,
        source: NonNull<u8>,
        target: NonNull<u8>,
        count: u32,
    ) -> bool {
        let count = self.0.size * count as usize;
        if count > 0 {
            unsafe { copy_nonoverlapping(source.as_ptr(), target.as_ptr(), count) };
            true
        } else {
            false
        }
    }

    #[inline]
    pub(crate) const unsafe fn copy_at(
        self,
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

    #[inline]
    pub(crate) unsafe fn drop(self, data: NonNull<u8>, count: u32) -> bool {
        if count > 0
            && let Some(drop) = self.0.drop
        {
            unsafe { drop(data, count) };
            true
        } else {
            false
        }
    }

    #[inline]
    pub(crate) unsafe fn drop_at(self, data: NonNull<u8>, index: u32, count: u32) -> bool {
        unsafe { self.drop(self.offset(data, index), count) }
    }

    #[inline]
    pub(crate) unsafe fn get<'a>(self, data: NonNull<u8>) -> &'a dyn Any {
        let item = unsafe { (self.0.get)(data) };
        debug_assert_eq!(item.type_id(), self.identifier());
        item
    }

    #[inline]
    pub(crate) unsafe fn get_at<'a>(self, data: NonNull<u8>, index: u32) -> &'a dyn Any {
        unsafe { self.get(self.offset(data, index)) }
    }

    #[inline]
    pub(crate) unsafe fn get_mut<'a>(self, data: NonNull<u8>) -> &'a mut dyn Any {
        let item = unsafe { (self.0.get_mut)(data) };
        debug_assert_eq!(item.type_id(), self.identifier());
        item
    }

    #[inline]
    pub(crate) unsafe fn get_mut_at<'a>(self, data: NonNull<u8>, index: u32) -> &'a mut dyn Any {
        unsafe { self.get_mut(self.offset(data, index)) }
    }

    #[inline]
    pub(crate) unsafe fn set(self, data: NonNull<u8>, value: Box<dyn Any>) {
        debug_assert_eq!(self.identifier(), value.type_id());
        unsafe { (self.0.set)(value, data) };
    }

    #[inline]
    pub(crate) unsafe fn set_at(self, data: NonNull<u8>, value: Box<dyn Any>, index: u32) {
        unsafe { self.set(self.offset(data, index), value) };
    }
}

impl PartialEq for Meta {
    fn eq(&self, other: &Self) -> bool {
        self.identifier() == other.identifier()
    }
}

impl Eq for Meta {}

impl PartialOrd for Meta {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.identifier().partial_cmp(&other.identifier())
    }
}

impl Ord for Meta {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.identifier().cmp(&other.identifier())
    }
}

impl Hash for Meta {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.identifier().hash(state);
    }
}
