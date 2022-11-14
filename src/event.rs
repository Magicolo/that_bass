use crate::{core::tuples, key::Key, table::Table, Database, Datum};
use dashmap::DashMap;
use std::{any::TypeId, collections::VecDeque, marker::PhantomData, num::NonZeroUsize, sync::Arc};

/// Allows to listen to database events. These events are guaranteed to be coherent (ex. `create` always happens before
/// `destroy` for a given key).
///
/// **All listen methods should be considered as time critical** since they are called while holding table locks and may add
/// contention on many other database operations. If these events need to be processed in some way, it is recommended to queue
/// the events and defer the processing.
pub trait Listen {
    fn on_create(&self, keys: &[Key], table: &Table);
    fn on_destroy(&self, keys: &[Key], table: &Table);
    fn on_add(&self, keys: &[Key], source: &Table, target: &Table);
    fn on_remove(&self, keys: &[Key], source: &Table, target: &Table);
}

pub trait Event {
    fn kind() -> Kind;
}

pub enum KeyEvent {
    Created(Key, u32),
    Destroyed(Key, u32),
    Added(Key, u32, u32),
    Removed(Key, u32, u32),
}

pub enum TableEvent {
    Created(Key),
    Destroyed(Key),
    Added(Key, u32),
    Removed(Key, u32),
}

#[derive(PartialEq, Eq, Hash)]
pub enum Kind {
    Create(TypeId),
    Destroy(TypeId),
    Add(TypeId),
    Remove(TypeId),
}

pub struct Listener<'a, E: Event> {
    broadcast: &'a Broadcast,
    keys: Vec<Key>,
    head: usize,
    index: usize,
    low: bool,
    _marker: PhantomData<fn(E)>,
}

pub struct Broadcast(Arc<DashMap<Kind, Entry>>);

pub struct OnAdd<D: Datum>(PhantomData<fn(D)>);

#[derive(Default)]
struct Entry {
    keys: VecDeque<Key>,
    head: usize,
    low: usize,
    high: usize,
    listeners: usize,
}

/*
    let (database, broadcast) = Database::new().broadcast();
    let query = database.query::<&mut Mass>()?;
    let mut on_add = broadcast.listen::<OnAdd<Mass>>();
    let mut by = By::new();
    by.keys(&mut on_add);
    query.each_by_ok(&mut by, |mass| { ... });
*/

impl<L> Database<L> {
    pub fn new_with<M: Listen>(self, listen: M) -> Database<(L, M)> {
        Database {
            inner: self.inner,
            listen: (self.listen, listen),
        }
    }

    pub fn broadcast(self) -> (Database<(L, Broadcast)>, Broadcast) {
        todo!()
        // let queue = ByKey(Arc::new(DashMap::new()));
        // (self.listen(queue.clone()), Broadcast { queue })
    }
}

impl Broadcast {
    pub fn listen<E: Event>(&self) -> Listener<E> {
        let mut entry = self.0.entry(E::kind()).or_default();
        entry.listeners += 1;
        Listener {
            broadcast: self,
            keys: Vec::new(),
            head: entry.head,
            index: 0,
            low: false,
            _marker: PhantomData,
        }
    }
}

impl<E: Event> Listener<'_, E> {
    pub fn next(&mut self) -> Option<Key> {
        if let Some(&key) = self.keys.get(self.index) {
            self.index += 1;
            return Some(key);
        }
        self.index = 0;
        self.keys.clear();

        let mut entry = unsafe { self.broadcast.0.get_mut(&E::kind()).unwrap_unchecked() };
        let count = NonZeroUsize::new(entry.keys.len().saturating_sub(self.head))?;
        self.keys.resize(count.get(), Key::NULL);
        let slices = entry.keys.as_slices();
        if let Some(keys) = slices.0.get(self.head..) {
            self.keys[..keys.len()].copy_from_slice(keys);
        }
        self.keys[slices.0.len()..slices.0.len() + slices.1.len()].copy_from_slice(slices.1);
        self.head += count.get();
        entry.high += 1;
        debug_assert!(entry.high <= entry.listeners);

        if self.low {
        } else if entry.low == entry.listeners {
        } else if entry.high == entry.listeners {
        }

        let key = *self.keys.get(self.index)?;
        self.index += 1;
        Some(key)
    }
}

impl<E: Event> Drop for Listener<'_, E> {
    fn drop(&mut self) {
        let mut entry = unsafe { self.broadcast.0.get_mut(&E::kind()).unwrap_unchecked() };
        debug_assert!(entry.listeners > 0);
        entry.listeners -= 1;
        self.broadcast
            .0
            .remove_if(&E::kind(), |_, entry| entry.listeners == 0);
    }
}

impl<D: Datum> Event for OnAdd<D> {
    #[inline]
    fn kind() -> Kind {
        Kind::Add(TypeId::of::<D>())
    }
}

// impl ByKey {
//     #[inline]
//     fn add(&self, keys: &[Key], mut with: impl FnMut(Key) -> KeyEvent) {
//         for &key in keys {
//             self.0.entry(key).or_insert_with(Vec::new).push(with(key))
//         }
//     }
// }

// impl ByTable {
//     #[inline]
//     fn add(&self, keys: &[Key], table: u32, mut with: impl FnMut(Key) -> TableEvent) {
//         self.0
//             .entry(table)
//             .or_insert_with(Vec::new)
//             .extend(keys.iter().copied().map(&mut with))
//     }
// }

// impl Listen for ByKey {
//     #[inline]
//     fn on_create(&self, keys: &[Key], table: &Table) {
//         self.add(keys, |key| KeyEvent::Created(key, table.index()));
//     }

//     #[inline]
//     fn on_destroy(&self, keys: &[Key], table: &Table) {
//         self.add(keys, |key| KeyEvent::Destroyed(key, table.index()));
//     }

//     #[inline]
//     fn on_add(&self, keys: &[Key], source: &Table, target: &Table) {
//         self.add(keys, |key| {
//             KeyEvent::Added(key, source.index(), target.index())
//         });
//     }

//     #[inline]
//     fn on_remove(&self, keys: &[Key], source: &Table, target: &Table) {
//         self.add(keys, |key| {
//             KeyEvent::Removed(key, source.index(), target.index())
//         });
//     }
// }

// impl Listen for ByTable {
//     #[inline]
//     fn on_create(&self, keys: &[Key], table: &Table) {
//         self.add(keys, table.index(), TableEvent::Created);
//     }

//     #[inline]
//     fn on_destroy(&self, keys: &[Key], table: &Table) {
//         self.add(keys, table.index(), TableEvent::Destroyed);
//     }

//     #[inline]
//     fn on_add(&self, keys: &[Key], source: &Table, target: &Table) {
//         self.add(keys, source.index(), |key| {
//             TableEvent::Removed(key, target.index())
//         });
//         self.add(keys, target.index(), |key| {
//             TableEvent::Added(key, source.index())
//         });
//     }

//     #[inline]
//     fn on_remove(&self, keys: &[Key], source: &Table, target: &Table) {
//         self.add(keys, source.index(), |key| {
//             TableEvent::Removed(key, target.index())
//         });
//         self.add(keys, target.index(), |key| {
//             TableEvent::Added(key, source.index())
//         });
//     }
// }

macro_rules! tuple {
    ($n:ident, $c:expr $(, $p:ident, $t:ident, $i:tt)*) => {
        impl<$($t: Listen,)*> Listen for ($($t,)*) {
            #[inline]
            fn on_create(&self, _keys: &[Key], _table: &Table) {
                $(self.$i.on_create(_keys, _table);)*
            }
            #[inline]
            fn on_destroy(&self, _keys: &[Key], _table: &Table) {
                $(self.$i.on_destroy(_keys, _table);)*
            }
            #[inline]
            fn on_add(&self, _keys: &[Key], _source: &Table, _target: &Table) {
                $(self.$i.on_add(_keys, _source, _target);)*
            }
            #[inline]
            fn on_remove(&self, _keys: &[Key], _source: &Table, _target: &Table) {
                $(self.$i.on_remove(_keys, _source, _target);)*
            }
        }
    };
}
tuples!(tuple);
