use crate::v4::{error::Error, table::Column};
use core::{
    alloc::Layout,
    iter::from_fn,
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
}

pub trait Push<T> {
    type Out;
    fn push(self, item: T) -> Self::Out;
}

pub trait Next {
    type Item<'a>
    where
        Self: 'a;
    type Rest<'a>: Next
    where
        Self: 'a;

    fn next(&mut self) -> (Self::Item<'_>, Self::Rest<'_>);
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

impl Next for () {
    type Item<'a> = ();
    type Rest<'a> = ();

    fn next(&mut self) -> (Self::Item<'_>, Self::Rest<'_>) {
        ((), ())
    }
}

impl<N: Next> Next for &mut N {
    type Item<'a>
        = N::Item<'a>
    where
        Self: 'a;
    type Rest<'a>
        = N::Rest<'a>
    where
        Self: 'a;

    fn next(&mut self) -> (Self::Item<'_>, Self::Rest<'_>) {
        N::next(self)
    }
}

impl<H, T: Next> Next for (H, T) {
    type Item<'a>
        = &'a mut H
    where
        Self: 'a;
    type Rest<'a>
        = &'a mut T
    where
        Self: 'a;

    fn next(&mut self) -> (Self::Item<'_>, Self::Rest<'_>) {
        (&mut self.0, &mut self.1)
    }
}

pub(crate) unsafe fn vec_as_slice<T>(vector: *const Vec<T>) -> *const [T] {
    todo!()
}

pub(crate) unsafe fn vec_as_slice_mut<T>(vector: *mut Vec<T>) -> *mut [T] {
    todo!()
}

pub(crate) unsafe fn box_as_slice<T>(slice: *const Box<[T]>) -> *const [T] {
    todo!()
}

pub(crate) unsafe fn box_as_slice_mut<T>(slice: *mut Box<[T]>) -> *mut [T] {
    todo!()
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

pub(crate) fn resize(
    columns: &mut [Column],
    data: NonNull<u8>,
    count: u32,
    capacities: (u32, u32),
) -> Result<NonNull<u8>, Error> {
    fn next(
        columns: &mut [Column],
        layouts: (Layout, Layout),
        count: u32,
        capacities: (u32, u32),
    ) -> Result<(Layout, NonNull<u8>), Error> {
        Ok(match columns.split_first_mut() {
            Some((head, tail)) => {
                let old = head
                    .meta
                    .extend(layouts.0, capacities.0)
                    .map_err(Error::Layout)?;
                let new = head
                    .meta
                    .extend(layouts.1, capacities.1)
                    .map_err(Error::Layout)?;
                let pair = next(tail, (old.0, new.0), count, capacities)?;
                let source = head.data;
                let target = unsafe { pair.1.add(new.1) };
                head.meta.initialize(source, target, count, capacities.1);
                head.data = target;
                pair
            }
            None if layouts.1.size() == 0 => (layouts.0.pad_to_align(), NonNull::dangling()),
            None => (layouts.0.pad_to_align(), unsafe {
                allocate(layouts.1.pad_to_align())
            }?),
        })
    }

    let (old, new) = next(
        columns,
        (Layout::new::<()>(), Layout::new::<()>()),
        count,
        capacities,
    )?;
    unsafe { deallocate(data, old) };
    Ok(new)
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
