use crate::v4::{Meta, error::Error, table::Column};
use core::{
    alloc::Layout,
    iter::{FusedIterator, from_fn},
    mem::{replace, take},
    ops::Range,
    ptr::NonNull,
};
use std::alloc::{alloc, dealloc};

pub trait IntoNest {
    type Nest;
    fn into_nest(self) -> Self::Nest;
}

pub trait IntoFlat {
    type Flat;
    fn into_flat(self) -> Self::Flat;
}

pub struct NestIter<I>(I);
pub struct FlatIter<I>(I);
pub struct AndIter<I: Iterator>(I, Option<I::Item>);

pub trait IteratorExtension: Iterator {
    fn into_nest(self) -> NestIter<Self>
    where
        Self: Sized,
        Self::Item: IntoNest,
    {
        NestIter(self)
    }

    fn into_flat(self) -> FlatIter<Self>
    where
        Self: Sized,
        Self::Item: IntoFlat,
    {
        FlatIter(self)
    }

    fn and(self, item: Self::Item) -> AndIter<Self>
    where
        Self: Sized,
    {
        AndIter(self, Some(item))
    }
}

pub trait Push<T> {
    type Out;
    fn push(self, item: T) -> Self::Out;
}

pub struct Defer<T, F: FnOnce() -> T>(Option<F>);

impl<T, F: FnOnce() -> T> Defer<T, F> {
    pub const fn new(defer: F) -> Self {
        Self(Some(defer))
    }

    pub fn run(mut self) -> T {
        self.0.take().expect("must be `Some`")()
    }
}

impl<T, F: FnOnce() -> T> Drop for Defer<T, F> {
    fn drop(&mut self) {
        if let Some(defer) = self.0.take() {
            defer();
        }
    }
}

pub fn defer<T, F: FnOnce() -> T>(defer: F) -> Defer<T, F> {
    Defer::new(defer)
}

impl<I> Push<I> for () {
    type Out = (I, Self);

    fn push(self, item: I) -> Self::Out {
        (item, ())
    }
}

impl<I, H, T: Push<I>> Push<I> for (H, T) {
    type Out = (H, T::Out);

    fn push(self, item: I) -> Self::Out {
        (self.0, self.1.push(item))
    }
}

pub(crate) unsafe fn allocate(layout: Layout) -> Result<NonNull<u8>, Error> {
    if layout.size() == 0 {
        Ok(NonNull::dangling())
    } else {
        NonNull::new(unsafe { alloc(layout) }).ok_or(Error::FailedToAllocate)
    }
}

pub(crate) unsafe fn deallocate(data: NonNull<u8>, layout: Layout) -> bool {
    if data == NonNull::dangling() || layout.size() == 0 {
        false
    } else {
        unsafe { dealloc(data.as_ptr(), layout) };
        true
    }
}

/// The `pairs` iterator must be sorted by `pair.0` (ascending or
/// descending), then by `pair.1` descending.
pub(crate) fn ranges(
    pairs: impl IntoIterator<Item = (u32, u32)>,
) -> impl Iterator<Item = (u32, Range<u32>)> {
    let mut table = u32::MAX;
    let mut start = u32::MAX;
    let mut count = 0u32;
    let mut iterator = pairs.into_iter();
    from_fn(move || {
        loop {
            match iterator.next() {
                Some(pair) if pair.0 == table => match start - pair.1 {
                    0 => continue,
                    1 => (start, count) = (pair.1, count + 1),
                    _ if count > 0 => {
                        let range = start..start + replace(&mut count, 1);
                        start = pair.1;
                        break Some((table, range));
                    }
                    _ => (start, count) = (pair.1, 1),
                },
                Some(pair) if count > 0 => {
                    let rows = start..start + replace(&mut count, 1);
                    let table = replace(&mut table, pair.0);
                    start = pair.1;
                    break Some((table, rows));
                }
                Some(pair) => (table, start, count) = (pair.0, pair.1, 1),
                None if count > 0 => break Some((table, start..start + take(&mut count))),
                None => break None,
            }
        }
    })
}

pub(crate) fn find<T, K: Ord, F: FnMut(&T) -> K>(slice: &[T], key: K, mut map: F) -> Option<usize> {
    if slice.len() < 32 {
        slice.iter().position(|item| map(item) == key)
    } else {
        slice.binary_search_by_key(&key, map).ok()
    }
}

impl<I: Iterator<Item: IntoNest>> Iterator for NestIter<I> {
    type Item = <I::Item as IntoNest>::Nest;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.0.next()?.into_nest())
    }
}

impl<I: Iterator<Item: IntoFlat>> Iterator for FlatIter<I> {
    type Item = <I::Item as IntoFlat>::Flat;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.0.next()?.into_flat())
    }
}

impl<I: Iterator> Iterator for AndIter<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().or_else(|| self.1.take())
    }
}

impl<I: DoubleEndedIterator> DoubleEndedIterator for AndIter<I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.1.take().or_else(|| self.0.next_back())
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for AndIter<I> {
    fn len(&self) -> usize {
        self.0.len() + self.1.iter().len()
    }
}

impl<I: FusedIterator> FusedIterator for AndIter<I> {}

impl<I: Iterator> IteratorExtension for I {}

macro_rules! tuple {
    ($($name: ident),*) => {
        tuple!(@recurse $($name),* [] [()]);
    };
    (@recurse [$($flat: ident),*] [$nest: tt]) => {
        tuple!(@implement [$($flat),*] [$nest]);
    };
    (@recurse $name: ident $(, $names: ident)* [$($flat: ident),*] [$nest: tt]) => {
        tuple!(@recurse $($names),* [$name $(, $flat)*] [($name, $nest)]);
        tuple!(@implement [$($flat),*] [$nest]);
    };
    (@implement [$($flat: ident),*] [$nest: tt]) => {
        #[allow(non_snake_case)]
        #[automatically_derived]
        impl<$($flat,)*> IntoNest for ($($flat,)*) {
            type Nest = $nest;

            fn into_nest(self) -> Self::Nest {
                let ($($flat,)*) = self;
                $nest
            }
        }

        #[allow(non_snake_case)]
        #[automatically_derived]
        impl<$($flat,)*> IntoFlat for $nest {
            type Flat = ($($flat,)*);

            fn into_flat(self) -> Self::Flat {
                let $nest = self;
                ($($flat,)*)
            }
        }
    }
}

tuple!(
    T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15
);
