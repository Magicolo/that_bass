use std::{
    ops::ControlFlow::{self, *},
    slice::SliceIndex,
};

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
                unsafe { items.swap_unchecked(head, tail) };
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
                unsafe { items.swap_unchecked(head, tail) };
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
