use super::{
    row::{ItemContext, Row, Rows},
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
pub struct Join<'d, L, R, K: JoinKey, B> {
    left: L,
    right: R,
    by: B,
    pairs: VecDeque<(Key, K::Value)>,
    slots: Vec<Vec<(Key, K::Value, &'d Slot)>>,
}

pub trait JoinKey {
    type Value;
    fn split(self) -> Option<(Key, Self::Value)>;
}

impl<L, R, K: JoinKey, B> Join<'_, L, R, K, B> {
    #[inline]
    pub fn new(left: L, right: R, by: B) -> Self {
        Self {
            left,
            right,
            by,
            pairs: VecDeque::new(),
            slots: Vec::new(),
        }
    }
}

impl<'d, L: Query<'d>, R: Row, K: JoinKey, B: FnMut(L::Item<'_>) -> K> Query<'d>
    for Join<'d, L, Rows<'d, R>, K, B>
{
    type Item<'a> = (K::Value, Result<R::Item<'a>, Error>);
    type Read = Join<'d, L, <Rows<'d, R> as Query<'d>>::Read, K, B>;

    fn initialize(&mut self, table: &'d Table) -> Result<(), Error> {
        self.left.initialize(table)?;
        self.right.initialize(table)?;
        Ok(())
    }

    fn read(self) -> Self::Read {
        Join {
            left: self.left,
            right: self.right.read(),
            by: self.by,
            slots: self.slots,
            pairs: self.pairs,
        }
    }

    #[inline]
    fn has(&mut self, key: Key, context: Context<'d>) -> bool {
        self.left.has(key, context)
    }

    #[inline]
    fn count(&mut self, context: Context<'d>) -> usize {
        self.left.count(context)
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
            Ok(by) => match by.split() {
                Some((key, value)) => self
                    .right
                    .try_find(key, context, |item| find(Ok((value, item)))),
                None => find(Err(Error::MissingJoinKey)),
            },
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
        let (key, value) = by.split().ok_or(Error::MissingJoinKey)?;
        Ok(self
            .right
            .try_find(key, context, |item| find((value, item))))
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
            pairs,
            slots,
            ..
        } = self;
        let mut fold = |mut state: S| -> Result<S, S> {
            // Collect all join keys and release all locks as fast as possible. This is important to reduce contention but also
            // to ensure that no deadlock occurs.
            left.each(context.own(), |item| {
                if let Some(pair) = by(item).split() {
                    pairs.push_back(pair);
                }
            });
            // States with no joined keys will be skipped.
            right.pending.clear();
            // Ensure that `slots` has the same length as `right.states`.
            slots.resize_with(right.states.len(), Vec::new);

            // Sort keys by state such that table locks can be used for (hopefully) more than one key at a time.
            for pair in pairs.drain(..) {
                match context.database.keys().get(pair.0) {
                    Ok(slot) => match right.indices.get(&slot.table()) {
                        Some(&index) => {
                            let slots = unsafe { slots.get_unchecked_mut(index as usize) };
                            if slots.len() == 0 {
                                right.pending.push_back(index);
                            }
                            slots.push((pair.0, pair.1, slot));
                            continue;
                        }
                        None => state = fold(state, (pair.1, Err(Error::KeyNotInQuery(pair.0))))?,
                    },
                    Err(error) => state = fold(state, (pair.1, Err(error)))?,
                }
            }

            state = right.try_guards(state, |mut state, index, row, pointers, table, inner| {
                let context = ItemContext(inner.keys(), pointers, 0);
                let slots = unsafe { slots.get_unchecked_mut(index as usize) };
                for (key, value, slot) in slots.drain(..) {
                    let (table_index, row_index) = slot.indices();
                    // The key is allowed to move within its table (such as with a swap as part of a remove).
                    if table.index() == table_index {
                        state = fold(
                            state,
                            (value, Ok(R::item(row, context.with(row_index as _)))),
                        )?;
                    } else {
                        // The key has moved to another table between the last moment the slot indices were read and now.
                        pairs.push_back((key, value));
                    }
                }
                Ok(state)
            });

            for (key, value) in pairs.drain(..) {
                let old = state;
                state = right.try_find(key, context.own(), |item| fold(old, (value, item)))?;
            }

            Ok(state)
        };

        match fold(state) {
            Ok(value) => value,
            Err(value) => {
                // Fold was interrupted.
                pairs.clear();
                for slots in slots.iter_mut() {
                    slots.clear();
                }
                value
            }
        }
    }

    fn fold<S, F: FnMut(S, Self::Item<'_>) -> S>(
        &mut self,
        mut context: Context<'d>,
        mut state: S,
        mut fold: F,
    ) -> S {
        let Self {
            left,
            right,
            by,
            pairs,
            slots,
            ..
        } = self;
        // Collect all join keys and release all locks as fast as possible. This is important to reduce contention but also
        // to ensure that no deadlock occurs.
        left.each(context.own(), |item| {
            if let Some(pair) = by(item).split() {
                pairs.push_back(pair);
            }
        });
        // States with no joined keys will be skipped.
        right.pending.clear();
        // Ensure that `slots` has the same length as `right.states`.
        slots.resize_with(right.states.len(), Vec::new);

        // Sort keys by state such that table locks can be used for (hopefully) more than one key at a time.
        for pair in pairs.drain(..) {
            match context.database.keys().get(pair.0) {
                Ok(slot) => match right.indices.get(&slot.table()) {
                    Some(&index) => {
                        let slots = unsafe { slots.get_unchecked_mut(index as usize) };
                        if slots.len() == 0 {
                            right.pending.push_back(index);
                        }
                        slots.push((pair.0, pair.1, slot));
                        continue;
                    }
                    None => state = fold(state, (pair.1, Err(Error::KeyNotInQuery(pair.0)))),
                },
                Err(error) => state = fold(state, (pair.1, Err(error))),
            }
        }

        state = right.guards(state, |mut state, index, row, pointers, table, inner| {
            let context = ItemContext(inner.keys(), pointers, 0);
            let slots = unsafe { slots.get_unchecked_mut(index as usize) };
            for (key, value, slot) in slots.drain(..) {
                let (table_index, row_index) = slot.indices();
                // The key is allowed to move within its table (such as with a swap as part of a remove).
                if table.index() == table_index {
                    let item = R::item(row, context.with(row_index as usize));
                    state = fold(state, (value, Ok(item)));
                } else {
                    // The key has moved to another table between the last moment the slot indices were read and now.
                    pairs.push_back((key, value));
                }
            }
            state
        });

        for (key, value) in pairs.drain(..) {
            let old = state;
            state = right.try_find(key, context.own(), |item| fold(old, (value, item)));
        }

        state
    }
}

impl<K: JoinKey> JoinKey for Option<K> {
    type Value = K::Value;

    #[inline]
    fn split(self) -> Option<(Key, Self::Value)> {
        self?.split()
    }
}

impl<K: JoinKey, E> JoinKey for Result<K, E> {
    type Value = K::Value;

    #[inline]
    fn split(self) -> Option<(Key, Self::Value)> {
        self.ok()?.split()
    }
}

impl JoinKey for Key {
    type Value = ();

    #[inline]
    fn split(self) -> Option<(Key, Self::Value)> {
        Some((self, ()))
    }
}

impl JoinKey for (Key,) {
    type Value = ();

    #[inline]
    fn split(self) -> Option<(Key, Self::Value)> {
        Some((self.0, ()))
    }
}

impl<T> JoinKey for (Key, T) {
    type Value = T;

    #[inline]
    fn split(self) -> Option<(Key, Self::Value)> {
        Some((self.0, self.1))
    }
}

impl<T1, T2> JoinKey for (Key, T1, T2) {
    type Value = (T1, T2);

    #[inline]
    fn split(self) -> Option<(Key, Self::Value)> {
        Some((self.0, (self.1, self.2)))
    }
}

impl<T1, T2, T3> JoinKey for (Key, T1, T2, T3) {
    type Value = (T1, T2, T3);

    #[inline]
    fn split(self) -> Option<(Key, Self::Value)> {
        Some((self.0, (self.1, self.2, self.3)))
    }
}
