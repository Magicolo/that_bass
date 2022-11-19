use std::{
    cmp::Ordering,
    iter::from_fn,
    num::NonZeroUsize,
    ops::ControlFlow::{self, *},
    ptr::swap,
    slice::SliceIndex,
};

pub const ONE: NonZeroUsize = unsafe { NonZeroUsize::new_unchecked(1) };

#[inline(always)]
pub unsafe fn get_unchecked<T, I: SliceIndex<[T]>>(items: &[T], index: I) -> &I::Output {
    if cfg!(debug_assertions) {
        items.get(index).unwrap()
    } else {
        unsafe { items.get_unchecked(index) }
    }
}

#[inline(always)]
pub unsafe fn get_unchecked_mut<T, I: SliceIndex<[T]>>(
    items: &mut [T],
    index: I,
) -> &mut I::Output {
    if cfg!(debug_assertions) {
        items.get_mut(index).unwrap()
    } else {
        unsafe { items.get_unchecked_mut(index) }
    }
}

#[inline(always)]
pub unsafe fn swap_unchecked<T>(items: &mut [T], a: usize, b: usize) {
    if cfg!(debug_assertions) {
        items.swap(a, b);
    } else {
        let pointer = items.as_mut_ptr();
        swap(pointer.add(a), pointer.add(b));
    }
}

#[inline(always)]
pub unsafe fn unreachable() -> ! {
    if cfg!(debug_assertions) {
        unreachable!();
    } else {
        core::hint::unreachable_unchecked()
    }
}

pub fn try_fold_swap<T, S, C>(
    items: &mut [T],
    mut state: S,
    mut context: C,
    mut try_fold: impl FnMut(S, &mut C, &mut T) -> Result<ControlFlow<S, S>, S>,
    mut fold: impl FnMut(S, &mut C, &mut T) -> ControlFlow<S, S>,
) -> ControlFlow<S, S> {
    let mut head = 0;
    let mut tail = items.len();
    while head < tail {
        // SAFETY:
        // - `head` is only ever incremented and is always `< tail`.
        // - `tail` is only ever decremented and is always `< items.len() && > head`.
        let item = unsafe { get_unchecked_mut(items, head) };
        state = match try_fold(state, &mut context, item) {
            Ok(state) => {
                // Success: Move forward.
                head += 1;
                state?
            }
            Err(state) => {
                // Failure: Requeue the item at the end of `items`.
                tail -= 1;
                // SAFETY:
                // - `tail` must be greater than 0 before the decrement because of the `while` condition.
                // - `head` and `tail` are always valid indices because of the safety explanation above.
                debug_assert!(head < items.len());
                debug_assert!(tail < items.len());
                unsafe { swap_unchecked(items, head, tail) };
                state
            }
        };
    }

    // Iterate in reverse to visit the oldest requeued item first.
    let mut tail = items.len();
    while head < tail {
        // Decrement before accessing the item.
        tail -= 1;
        let item = unsafe { get_unchecked_mut(items, tail) };
        state = fold(state, &mut context, item)?
    }

    Continue(state)
}

pub fn fold_swap<T, S, C>(
    items: &mut [T],
    mut state: S,
    mut context: C,
    mut try_fold: impl FnMut(S, &mut C, &mut T) -> Result<S, S>,
    mut fold: impl FnMut(S, &mut C, &mut T) -> S,
) -> S {
    let mut head = 0;
    let mut tail = items.len();
    while head < tail {
        // SAFETY:
        // - `head` is only ever incremented and is always `< tail`.
        // - `tail` is only ever decremented and is always `< items.len() && > head`.
        let item = unsafe { get_unchecked_mut(items, head) };
        state = match try_fold(state, &mut context, item) {
            Ok(state) => {
                // Success: Move forward.
                head += 1;
                state
            }
            Err(state) => {
                // Failure: Requeue the item at the end of `items`.
                tail -= 1;
                // SAFETY:
                // - `tail` must be greater than 0 before the decrement because of the `while` condition.
                // - `head` and `tail` are always valid indices because of the safety explanation above.
                debug_assert!(head < items.len());
                debug_assert!(tail < items.len());
                unsafe { swap_unchecked(items, head, tail) };
                state
            }
        };
    }

    // Iterate in reverse to visit the oldest requeued item first.
    let mut tail = items.len();
    while head < tail {
        // Decrement before accessing the item.
        tail -= 1;
        let item = unsafe { get_unchecked_mut(items, tail) };
        state = fold(state, &mut context, item)
    }

    state
}

#[inline]
pub fn sorted_contains<T: Ord + 'static>(
    left: impl IntoIterator<Item = T>,
    right: impl IntoIterator<Item = T>,
) -> bool {
    let mut left = left.into_iter();
    for right in right {
        while let Some(left) = left.next() {
            match left.cmp(&right) {
                Ordering::Equal => break,
                Ordering::Less => continue,
                Ordering::Greater => return false,
            }
        }
    }
    true
}

#[inline]
pub fn sorted_difference<T: Ord>(
    left: impl IntoIterator<Item = T>,
    right: impl IntoIterator<Item = T>,
) -> impl Iterator<Item = T> {
    sorted_difference_by(T::cmp, left, right)
}

#[inline]
pub fn sorted_difference_by<T>(
    mut compare: impl FnMut(&T, &T) -> Ordering,
    left: impl IntoIterator<Item = T>,
    right: impl IntoIterator<Item = T>,
) -> impl Iterator<Item = T> {
    let mut right_pair = (None, right.into_iter());
    left.into_iter().filter_map(move |left| {
        while let Some(right) = Option::take(&mut right_pair.0).or_else(|| right_pair.1.next()) {
            match compare(&left, &right) {
                Ordering::Equal => return None,
                Ordering::Less => {
                    right_pair.0 = Some(right);
                    return Some(left);
                }
                Ordering::Greater => continue,
            }
        }
        Some(left)
    })
}

#[inline]
pub fn sorted_symmetric_difference<T: Ord>(
    left: impl IntoIterator<Item = T>,
    right: impl IntoIterator<Item = T>,
) -> impl Iterator<Item = T> {
    sorted_symmetric_difference_by(T::cmp, left, right)
}

#[inline]
pub fn sorted_symmetric_difference_by<T>(
    mut compare: impl FnMut(&T, &T) -> Ordering,
    left: impl IntoIterator<Item = T>,
    right: impl IntoIterator<Item = T>,
) -> impl Iterator<Item = T> {
    let mut left_pair = (None, left.into_iter());
    let mut right_pair = (None, right.into_iter());
    from_fn(move || loop {
        match Option::take(&mut left_pair.0).or_else(|| left_pair.1.next()) {
            Some(left) => match Option::take(&mut right_pair.0).or_else(|| right_pair.1.next()) {
                Some(right) => match compare(&left, &right) {
                    Ordering::Equal => continue,
                    Ordering::Less => {
                        right_pair.0 = Some(right);
                        break Some(left);
                    }
                    Ordering::Greater => {
                        left_pair.0 = Some(left);
                        break Some(right);
                    }
                },
                None => break Some(left),
            },
            None => break Option::take(&mut right_pair.0).or_else(|| right_pair.1.next()),
        }
    })
}
