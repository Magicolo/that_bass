use std::ops::ControlFlow::{self, *};

pub fn try_fold_swap<S, C, T>(
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
        let item = unsafe { items.get_unchecked_mut(head) };
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
        let item = unsafe { items.get_unchecked_mut(tail) };
        state = fold(state, &mut context, item)?
    }

    Continue(state)
}

pub fn fold_swap<S, C, T>(
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
        let item = unsafe { items.get_unchecked_mut(head) };
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
        let item = unsafe { items.get_unchecked_mut(tail) };
        state = fold(state, &mut context, item)
    }

    state
}
