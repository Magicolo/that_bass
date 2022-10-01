use super::{
    row::{Lock, Row, Rows},
    *,
};
use crate::{
    key::{Key, Slot},
    Error,
};
use std::iter::from_generator;

/// This query, as the name suggests, joins two queries together by using a function that retrieves join keys from the
/// left query to find an item in the right query.
///
/// Since this join only returns the items from the right query, it requires less synchronization than a `FullJoin` at the cost
/// of some coherence. It is possible that a key that was used to join the left query to the right query may have changed
/// while an item is in use. To ensure strict coherence, use a `FullJoin`.
pub struct Join<'a, L, R, K, B> {
    left: L,
    right: R,
    by: B,
    keys: Vec<K>,
    slots: Vec<Vec<(K, &'a Slot)>>,
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
            keys: Vec::new(),
            slots: Vec::new(),
        }
    }
}

impl<'a, L: Query<'a>, R: Row, K: JoinKey, B: FnMut(L::Item) -> K> Query<'a>
    for Join<'a, L, Rows<'a, R>, K, B>
{
    type Item = (K, Option<<Rows<'a, R> as Query<'a>>::Item>);
    type Items<'b> = impl Iterator<Item = Self::Item> where Self: 'b;
    type Guard = Option<<Rows<'a, R> as Query<'a>>::Guard>;
    type Read = Join<'a, L, <Rows<'a, R> as Query<'a>>::Read, K, B>;

    #[inline]
    fn item<'b>(
        &'b mut self,
        key: Key,
        mut context: ItemContext<'a, 'b>,
    ) -> Result<Guard<Self::Item, Self::Guard>, Error> {
        let left = self.left.item(key, context.own())?;
        let by = (self.by)(left.0);
        drop(left.1);
        match self.right.item(by.key(), context) {
            Ok(right) => Ok(Guard((by, Some(right.0)), Some(right.1))),
            Err(_) => Ok(Guard((by, None), None)),
        }
    }

    fn items<'b>(&'b mut self, mut context: ItemContext<'a, 'b>) -> Self::Items<'b> {
        let Self {
            left,
            right,
            by,
            keys,
            slots,
            ..
        } = self;
        // Collect all join keys and release all locks as fast as possible. This is important to reduce contention but also
        // to ensure that no deadlock occurs.
        keys.extend(left.items(context.own()).map(by));
        // For consistency, `update` must only be called before iterating, not while iterating.
        right.update(context.database);
        // States with no joined keys will be skipped.
        right.done.clear();
        // Ensure that `states` has the same length as `right.states`.
        slots.resize_with(right.states.len(), Vec::new);

        // Sort keys by state such that table locks can be used for (hopefully) more than one key at a time.
        for key in keys.drain(..) {
            if let Ok(slot) = context.database.keys().get(key.key()) {
                if let Some(&state_index) = right.indices.get(&slot.table()) {
                    let slots = unsafe { slots.get_unchecked_mut(state_index as usize) };
                    if slots.len() == 0 {
                        right.done.push_back(state_index);
                    }
                    slots.push((key, slot));
                }
            }
        }

        from_generator(move || {
            // TODO: What if the iterator is not consumed completely?
            // - `keys` is cleared in `JoinKeys`
            // - `slots` may not be cleared properly.
            // - Add a drop implementation that clears keys and slots.
            // - Will most likely require the manual implementation of an iterator.
            for Guard((state_index, mut guard), table_read) in right.guards(context.database) {
                let slots = unsafe { slots.get_unchecked_mut(state_index as usize) };
                for (key, slot) in slots.drain(..) {
                    let (table_index, store_index) = slot.indices();
                    if table_read.table().index() == table_index {
                        yield (
                            key,
                            Some(<R::State as Lock<'a>>::item(
                                &mut guard,
                                store_index as usize,
                            )),
                        )
                    } else {
                        // Key has moved to another table between the last moment the slot indices were read and now.
                        keys.push(key);
                    }
                }
                drop(table_read);
            }

            for key in keys.drain(..) {
                match right.with(key.key(), |guard| guard, context.database) {
                    Ok(right) => {
                        yield (key, Some(right.0));
                        drop(right.1);
                    }
                    Err(_) => yield (key, None),
                }
            }
        })
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
