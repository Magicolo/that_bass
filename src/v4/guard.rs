use core::{
    ptr::NonNull,
    slice::{from_raw_parts, from_raw_parts_mut},
};
use parking_lot::{
    MappedRwLockReadGuard, MappedRwLockWriteGuard, RwLockReadGuard, RwLockWriteGuard,
};

pub trait Guard {
    type Read<'a, T: 'a>;
    type Write<'a, T: 'a>;
}

pub trait Bind {
    type Guard;
    fn bind(self, count: u32) -> Self::Guard;
}

pub struct Read<'a, T: 'a, G: Guard>(G::Read<'a, T>);
pub struct Write<'a, T: 'a, G: Guard>(G::Write<'a, T>);
pub struct Raw;
pub struct Map;

impl Guard for Raw {
    type Read<'a, T: 'a> = RwLockReadGuard<'a, NonNull<u8>>;
    type Write<'a, T: 'a> = RwLockWriteGuard<'a, NonNull<u8>>;
}

impl Guard for Map {
    type Read<'a, T: 'a> = MappedRwLockReadGuard<'a, [T]>;
    type Write<'a, T: 'a> = MappedRwLockWriteGuard<'a, [T]>;
}

impl<'a, T: 'a, G: Guard> Read<'a, T, G> {
    pub(crate) const fn new(guard: G::Read<'a, T>) -> Self {
        Self(guard)
    }
}

impl<'a, T: 'a> Bind for Read<'a, T, Raw> {
    type Guard = Read<'a, T, Map>;

    fn bind(self, count: u32) -> Self::Guard {
        Read(RwLockReadGuard::map(self.0, |data| unsafe {
            from_raw_parts(data.cast::<T>().as_ptr(), count as usize)
        }))
    }
}

impl<'a, T: 'a, G: Guard> Write<'a, T, G> {
    pub(crate) const fn new(guard: G::Write<'a, T>) -> Self {
        Self(guard)
    }
}

impl<'a, T: 'a> Bind for Write<'a, T, Raw> {
    type Guard = Write<'a, T, Map>;

    fn bind(self, count: u32) -> Self::Guard {
        Write(RwLockWriteGuard::map(self.0, |data| unsafe {
            from_raw_parts_mut(data.cast::<T>().as_ptr(), count as usize)
        }))
    }
}

impl Bind for () {
    type Guard = ();

    fn bind(self, _: u32) -> Self::Guard {}
}

impl<B0: Bind, B1: Bind> Bind for (B0, B1) {
    type Guard = (B0::Guard, B1::Guard);

    fn bind(self, count: u32) -> Self::Guard {
        (self.0.bind(count), self.1.bind(count))
    }
}

impl<B: Bind> Bind for Option<B> {
    type Guard = Option<B::Guard>;

    fn bind(self, count: u32) -> Self::Guard {
        self.map(|guard| guard.bind(count))
    }
}
