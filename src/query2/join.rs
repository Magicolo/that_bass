use super::{
    row::{Row, Rows},
    *,
};
use crate::{
    key::{Key, Slot},
    Error,
};
use std::collections::VecDeque;

/// This query, as the name suggests, joins two queries together by using a function that retrieves join keys from the
/// left query to find an item in the right query.
///
/// Since this join only returns the items from the right query, it requires less synchronization than a `FullJoin` at the cost
/// of some coherence. It is possible that a key that was used to join the left query to the right query may have changed
/// while an item is in use. To ensure strict coherence, use a `FullJoin`.
pub struct Join<'d, L, R, K, B> {
    left: L,
    right: R,
    by: B,
    keys: VecDeque<K>,
    slots: Vec<Vec<(K, &'d Slot)>>,
}

pub trait JoinKey {
    fn key(&self) -> Key;
}

impl<L, R, K, B> Join<'_, L, R, K, B> {
    #[inline]
    pub fn new(left: L, right: R, by: B) -> Self {
        Self {
            left,
            right,
            by,
            keys: VecDeque::new(),
            slots: Vec::new(),
        }
    }
}

impl<'d, L: Query<'d>, R: Row, K: JoinKey, B: FnMut(L::Item<'_>) -> K> Query<'d>
    for Join<'d, L, Rows<'d, R>, K, B>
{
    type Item<'a> = (K, Result<R::Item<'a>, Error>);
    type Read = Join<'d, L, <Rows<'d, R> as Query<'d>>::Read, K, B>;

    fn initialize(&mut self, table: &'d Table) {
        self.left.initialize(table);
        self.right.initialize(table);
    }

    #[inline]
    fn try_find<T, F: FnOnce(Result<Self::Item<'_>, Error>) -> T>(
        &mut self,
        key: Key,
        mut context: super::Context<'d>,
        find: F,
    ) -> T {
        match self
            .left
            .try_find(key, context.own(), |item| item.map(&mut self.by))
        {
            Ok(by) => self
                .right
                .try_find(key.key(), context, |item| find(Ok((by, item)))),
            Err(error) => find(Err(error)),
        }
    }

    #[inline]
    fn find<T, F: FnOnce(Self::Item<'_>) -> T>(
        &mut self,
        key: Key,
        mut context: super::Context<'d>,
        find: F,
    ) -> Result<T, Error> {
        let by = self.left.find(key, context.own(), &mut self.by)?;
        Ok(self
            .right
            .try_find(by.key(), context, |item| find((by, item))))
    }

    #[inline]
    fn try_fold<S, F: FnMut(S, Self::Item<'_>) -> Result<S, S>>(
        &mut self,
        mut context: Context<'d>,
        state: S,
        mut fold: F,
    ) -> S {
        let Self {
            left,
            right,
            by,
            keys,
            slots,
            ..
        } = self;
        let mut fold = |mut state: S| -> Result<S, S> {
            // Collect all join keys and release all locks as fast as possible. This is important to reduce contention but also
            // to ensure that no deadlock occurs.
            left.each(context.own(), |item| keys.push_back(by(item)));
            // States with no joined keys will be skipped.
            right.done.clear();
            // Ensure that `states` has the same length as `right.states`.
            slots.resize_with(right.states.len(), Vec::new);

            // Sort keys by state such that table locks can be used for (hopefully) more than one key at a time.
            for join in keys.drain(..) {
                let key = join.key();
                match context.database.keys().get(key) {
                    Ok(slot) => match right.indices.get(&slot.table()) {
                        Some(&state_index) => {
                            let slots = unsafe { slots.get_unchecked_mut(state_index as usize) };
                            if slots.len() == 0 {
                                right.done.push_back(state_index);
                            }
                            slots.push((join, slot));
                            continue;
                        }
                        None => state = fold(state, (join, Err(Error::KeyNotInQuery(key))))?,
                    },
                    Err(error) => state = fold(state, (join, Err(error)))?,
                }
            }

            state = right.guards(
                context.database,
                state,
                |mut state, index, mut guard, table| {
                    let slots = unsafe { slots.get_unchecked_mut(index as usize) };
                    for (key, slot) in slots.drain(..) {
                        let (table_index, store_index) = slot.indices();
                        // The key is allowed to move within its table (such as with a swap as part of a remove).
                        if table.table().index() == table_index {
                            state =
                                fold(state, (key, Ok(R::item(&mut guard, store_index as usize))))?;
                        } else {
                            // The key has moved to another table between the last moment the slot indices were read and now.
                            keys.push_back(key);
                        }
                    }
                    Ok(state)
                },
            );

            for key in keys.drain(..) {
                let old = state;
                state = right.try_find(key.key(), context.own(), |item| fold(old, (key, item)))?;
            }

            Ok(state)
        };

        match fold(state) {
            Ok(value) => value,
            Err(value) => {
                // Fold was interrupted.
                keys.clear();
                for slots in slots.iter_mut() {
                    slots.clear();
                }
                value
            }
        }
    }

    fn read(self) -> Self::Read {
        Join {
            left: self.left,
            right: self.right.read(),
            by: self.by,
            slots: self.slots,
            keys: self.keys,
        }
    }
}

impl JoinKey for Key {
    #[inline]
    fn key(&self) -> Key {
        *self
    }
}

impl JoinKey for (Key,) {
    #[inline]
    fn key(&self) -> Key {
        self.0
    }
}

impl<T> JoinKey for (Key, T) {
    #[inline]
    fn key(&self) -> Key {
        self.0
    }
}

impl<T1, T2> JoinKey for (Key, T1, T2) {
    #[inline]
    fn key(&self) -> Key {
        self.0
    }
}
