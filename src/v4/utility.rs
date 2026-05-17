use core::alloc::Layout;
use core::iter::from_fn;
use core::ops::Range;
use core::ptr::NonNull;
use core::{mem::replace, mem::take};
use std::alloc::{alloc, dealloc};

use super::column::Column;
use super::error::Error;
use super::meta::Meta;
use super::row::At;

pub(super) fn sort(metas: impl IntoIterator<Item = Meta>) -> Result<Vec<Meta>, Error> {
    let mut metas = metas.into_iter().collect::<Vec<_>>();
    metas.sort_by_key(|meta| meta.identifier);
    for [left, right] in metas.array_windows::<2>() {
        if left.identifier == right.identifier {
            return Err(Error::DuplicateMeta);
        }
    }
    Ok(metas)
}

pub(super) unsafe fn allocate(layout: Layout) -> Result<NonNull<u8>, Error> {
    if layout.size() == 0 {
        Ok(NonNull::dangling())
    } else {
        NonNull::new(unsafe { alloc(layout) }).ok_or(Error::FailedToAllocate)
    }
}

pub(super) unsafe fn deallocate(data: NonNull<u8>, layout: Layout) -> bool {
    if data == NonNull::dangling() || layout.size() == 0 {
        false
    } else {
        unsafe { dealloc(data.as_ptr(), layout) };
        true
    }
}

/// The `pairs` iterator must be sorted by `pair.0` (ascending or
/// descending), then by `pair.1` descending.
pub(super) fn ranges(
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

pub(super) fn resize(
    columns: &mut [Column],
    data: NonNull<u8>,
    count: u32,
    capacities: (u32, u32),
) -> Result<NonNull<u8>, Error> {
    fn next(
        columns: &mut [Column],
        data: NonNull<u8>,
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
                let pair = next(tail, data, (old.0, new.0), count, capacities)?;
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

    let (layout, data) = next(
        columns,
        data,
        (Layout::new::<()>(), Layout::new::<()>()),
        count,
        capacities,
    )?;
    unsafe { deallocate(data, layout) };
    Ok(data)
}

pub(super) fn find<T, K: Ord, F: FnMut(&T) -> K>(
    slice: &[T],
    key: K,
    mut map: F,
) -> Option<At<'_, T>> {
    let index = if slice.len() < 32 {
        slice.iter().position(|item| map(item) == key)?
    } else {
        slice.binary_search_by_key(&key, map).ok()?
    };
    Some(At(index.try_into().ok()?, slice.get(index)?))
}
