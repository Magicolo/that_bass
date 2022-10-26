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
pub struct Join<'d, L, R, V, B> {
    left: L,
    right: R,
    by: B,
    pairs: VecDeque<(Key, V)>,
    slots: Vec<Vec<(Key, V, &'d Slot)>>,
}

pub struct By<'a, V>(&'a mut VecDeque<(Key, V)>);

impl<L, R, V, B> Join<'_, L, R, V, B> {
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

impl<'d, L: TableQuery<'d>, R: Row, F: Filter, V, B: FnMut(By<V>, L::Item<'_>)> TableQuery<'d>
    for Join<'d, L, Rows<'d, R, F>, V, B>
{
    type Item<'a> = (V, Result<R::Item<'a>, Error>);

    #[inline]
    fn count(&mut self, context: Context<'d>) -> usize {
        self.left.count(context)
    }

    #[inline]
    fn try_fold<S, G: FnMut(S, Self::Item<'_>) -> Result<S, S>>(
        &mut self,
        mut context: Context<'d>,
        state: S,
        mut fold: G,
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
            left.each(context.own(), |item| by(By(pairs), item));

            // TODO: Measure to determine the best value here.
            // if pairs.len() > 4
            {
                // States with no joined keys will be skipped.
                right.update(context.database);
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
                            None => {
                                state = fold(state, (pair.1, Err(Error::KeyNotInQuery(pair.0))))?
                            }
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
            }

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

    fn fold<S, G: FnMut(S, Self::Item<'_>) -> S>(
        &mut self,
        mut context: Context<'d>,
        mut state: S,
        mut fold: G,
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
        left.each(context.own(), |item| by(By(pairs), item));

        // TODO: Measure to determine the best value here.
        // if pairs.len() > 4
        {
            // States with no joined keys will be skipped.
            right.update(context.database);
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
        }

        for (key, value) in pairs.drain(..) {
            let old = state;
            state = right.try_find(key, context.own(), |item| fold(old, (value, item)));
        }

        state
    }
}

impl<'d, L: KeyQuery<'d>, R: Row, F: Filter, V, B: FnMut(By<V>, L::Item<'_>)> KeyQuery<'d>
    for Join<'d, L, Rows<'d, R, F>, V, B>
{
    #[inline]
    fn has(&mut self, key: Key, context: Context<'d>) -> bool {
        self.left.has(key, context)
    }

    #[inline]
    fn try_find<T, G: FnOnce(Result<Self::Item<'_>, Error>) -> T>(
        &mut self,
        key: Key,
        mut context: super::Context<'d>,
        mut find: G,
    ) -> T {
        match self.left.try_find(key, context.own(), |item| {
            item.map(|item| (self.by)(By(&mut self.pairs), item))
        }) {
            Ok(_) => {
                let mut pairs = self.pairs.drain(..);
                match pairs.next() {
                    Some((mut key, mut value)) => {
                        while let Some(next) = pairs.next() {
                            let result =
                                self.right.try_find(key, context.own(), |item| match item {
                                    Ok(item) => Ok(find(Ok((value, Ok(item))))),
                                    Err(_) => Err(find),
                                });
                            find = match result {
                                Ok(value) => return value,
                                Err(find) => find,
                            };
                            (key, value) = next;
                        }

                        self.right
                            .try_find(key, context.own(), |item| find(Ok((value, item))))
                    }
                    None => find(Err(Error::MissingJoinKey)),
                }
            }
            Err(error) => find(Err(error)),
        }
    }

    #[inline]
    fn find<T, G: FnOnce(Self::Item<'_>) -> T>(
        &mut self,
        key: Key,
        mut context: Context<'d>,
        mut find: G,
    ) -> Result<T, Error> {
        self.left.find(key, context.own(), |item| {
            (self.by)(By(&mut self.pairs), item)
        })?;
        let mut pairs = self.pairs.drain(..);
        let (mut key, mut value) = pairs.next().ok_or(Error::MissingJoinKey)?;
        while let Some(next) = pairs.next() {
            let result = self.right.try_find(key, context.own(), |item| match item {
                Ok(item) => Ok(find((value, Ok(item)))),
                Err(_) => Err(find),
            });
            find = match result {
                Ok(value) => return Ok(value),
                Err(find) => find,
            };
            (key, value) = next;
        }

        Ok(self
            .right
            .try_find(key, context.own(), |item| find((value, item))))
    }
}

impl<'d, L: TableQuery<'d>, R: Row, F: Filter, V, B: FnMut(By<V>, L::Item<'_>)> ReadQuery<'d>
    for Join<'d, L, Rows<'d, R, F>, V, B>
{
    type Read = Join<'d, L, <Rows<'d, R, F> as ReadQuery<'d>>::Read, V, B>;

    fn read(self) -> Self::Read {
        Join {
            left: self.left,
            right: self.right.read(),
            by: self.by,
            pairs: self.pairs,
            slots: self.slots,
        }
    }
}

impl<'d, L: TableQuery<'d>, R: Row, F: Filter, V, B: FnMut(By<V>, L::Item<'_>)> FilterQuery<'d>
    for Join<'d, L, Rows<'d, R, F>, V, B>
{
    type Filter<G: Filter> = Join<'d, L, <Rows<'d, R, F> as FilterQuery<'d>>::Filter<G>, V, B>;

    fn filter<G: Filter>(self, filter: G) -> Self::Filter<G> {
        Join {
            left: self.left,
            right: self.right.filter(filter),
            by: self.by,
            pairs: self.pairs,
            slots: self.slots,
        }
    }
}

impl By<'_, ()> {
    #[inline]
    pub fn key(self, key: Key) {
        self.pair(key, ())
    }

    #[inline]
    pub fn keys<I: IntoIterator<Item = Key>>(self, keys: I) {
        self.pairs(keys.into_iter().map(|key| (key, ())))
    }
}

impl<V> By<'_, V> {
    #[inline]
    pub fn pair(self, key: Key, value: V) {
        self.0.push_back((key, value));
    }

    #[inline]
    pub fn pairs<I: IntoIterator<Item = (Key, V)>>(self, pairs: I) {
        self.0.extend(pairs);
    }
}
