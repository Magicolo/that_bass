use crate::v4::{Error, key, table};
use core::{
    alloc::{Layout, LayoutError},
    any::{Any, TypeId, type_name},
    cmp::Ordering,
    hash::Hash,
    mem::needs_drop,
    ptr::{NonNull, copy_nonoverlapping, slice_from_raw_parts_mut},
};
use parking_lot::Mutex;
use std::{collections::BTreeMap, sync::LazyLock};

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

    pub(crate) fn key() -> Self {
        static KEY: LazyLock<Meta> = LazyLock::new(Meta::of::<key::Key>);
        *KEY
    }

    pub(crate) fn table() -> Self {
        static TABLE: LazyLock<Meta> = LazyLock::new(Meta::of::<table::Table>);
        *TABLE
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

    pub(crate) fn is_key(self) -> bool {
        self.0.identifier == TypeId::of::<key::Key>()
    }

    pub(crate) fn is_table(self) -> bool {
        self.0.identifier == TypeId::of::<table::Table>()
    }

    #[inline]
    pub(crate) fn layout(self, count: u32) -> Result<Layout, Error> {
        (self.0.layout)(count).map_err(Error::Layout)
    }

    #[inline]
    pub(crate) fn extend(self, layout: Layout, count: u32) -> Result<(Layout, usize), Error> {
        layout.extend(self.layout(count)?).map_err(Error::Layout)
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
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Meta {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.is_key(), other.is_key()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => self.identifier().cmp(&other.identifier()),
        }
    }
}

impl Hash for Meta {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.identifier().hash(state);
    }
}
