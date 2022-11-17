use crate::{core::tuples, key::Key, Database, Datum};
use parking_lot::{Mutex, RwLock, RwLockWriteGuard};
use std::{
    any::TypeId,
    collections::VecDeque,
    marker::PhantomData,
    mem::{replace, ManuallyDrop},
    num::NonZeroUsize,
    ops::ControlFlow::{self, *},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

/// Allows to listen to database events. These events are guaranteed to be coherent (ex. `create` always happens before
/// `destroy` for a given key).
///
/// **All listen methods should be considered as time critical** since they are called while holding locks and may add significant
/// contention on other database operations. If these events need to be processed in any way, it is recommended to queue
/// the events and defer the processing.
pub trait Listen {
    fn on_create(&self, keys: &[Key], types: &[TypeId]);
    fn on_destroy(&self, keys: &[Key], types: &[TypeId]);
    fn on_modify(&self, keys: &[Key], add: &[TypeId], remove: &[TypeId]);
}

pub trait Event: Sized {
    fn process(raw: &Raw, context: Context<Self>) -> ControlFlow<()>;
}

#[derive(Clone)]
pub struct Broadcast(Arc<Inner>);

pub struct Receive<'a, E> {
    pub keep: Keep,
    inner: &'a Inner,
    events: VecDeque<E>,
    head: usize,
    version: usize,
}

#[derive(Clone, Copy, Default)]
pub enum Keep {
    #[default]
    All,
    First(NonZeroUsize),
    Last(NonZeroUsize),
}

pub struct Context<'a, T> {
    items: &'a mut VecDeque<T>,
    keep: &'a Keep,
    keys: &'a [Key],
    types: &'a [TypeId],
}

#[derive(Clone, Copy)]
pub enum Raw {
    Create {
        keys: (u32, u32),
        types: (u32, u32),
    },
    Destroy {
        keys: (u32, u32),
        types: (u32, u32),
    },
    Modify {
        keys: (u32, u32),
        types: (u32, u32, u32),
    },
}

struct Inner {
    ready: RwLock<Ready>,
    pending: Mutex<Chunk>,
}

#[derive(Default)]
struct Ready {
    chunks: VecDeque<Chunk>,
    last: Option<Chunk>,
    version: usize,
    head: usize,
    next: usize,
    seen: AtomicUsize,
    buffers: AtomicUsize,
}

#[derive(Default)]
struct Chunk {
    events: Vec<Raw>,
    keys: Vec<Key>,
    types: Vec<TypeId>,
}

impl<L> Database<L> {
    pub fn listen<M: Listen>(self, listen: M) -> Database<(L, M)> {
        Database {
            inner: self.inner,
            listen: (self.listen, listen),
        }
    }

    pub fn broadcast(self) -> (Database<(L, Broadcast)>, Broadcast) {
        let inner = Arc::new(Inner {
            pending: Default::default(),
            ready: Default::default(),
        });
        let broadcast = Broadcast(inner);
        (self.listen(broadcast.clone()), broadcast)
    }
}

impl Broadcast {
    pub fn on<E: Event>(&self) -> Receive<E> {
        self.on_with(Keep::default())
    }

    pub fn on_with<E: Event>(&self, keep: Keep) -> Receive<E> {
        Receive::new(&self.0, keep)
    }
}

impl Listen for Broadcast {
    #[inline]
    fn on_create(&self, keys: &[Key], types: &[TypeId]) {
        let mut pending = self.0.pending.lock();
        let indices = (pending.keys.len(), pending.types.len());
        pending.events.push(Raw::Create {
            keys: (indices.0 as _, keys.len() as _),
            types: (indices.1 as _, types.len() as _),
        });
        pending.keys.extend_from_slice(keys);
        pending.types.extend_from_slice(types);
    }
    #[inline]
    fn on_destroy(&self, keys: &[Key], types: &[TypeId]) {
        let mut pending = self.0.pending.lock();
        let indices = (pending.keys.len(), pending.types.len());
        pending.events.push(Raw::Destroy {
            keys: (indices.0 as _, keys.len() as _),
            types: (indices.1 as _, types.len() as _),
        });
        pending.keys.extend_from_slice(keys);
        pending.types.extend_from_slice(types);
    }
    #[inline]
    fn on_modify(&self, keys: &[Key], add: &[TypeId], remove: &[TypeId]) {
        let mut pending = self.0.pending.lock();
        let indices = (pending.keys.len(), pending.types.len());
        pending.events.push(Raw::Modify {
            keys: (indices.0 as _, keys.len() as _),
            types: (indices.1 as _, add.len() as _, remove.len() as _),
        });
        pending.keys.extend_from_slice(keys);
        pending.types.extend_from_slice(add);
        pending.types.extend_from_slice(remove);
    }
}

impl<'a, T> Context<'a, T> {
    #[inline]
    pub fn one(&mut self, item: T) -> ControlFlow<()> {
        self.items.push_back(item);
        self.keep()
    }

    #[inline]
    pub fn all<I: IntoIterator<Item = T>>(&mut self, items: I) -> ControlFlow<()> {
        self.items.extend(items);
        self.keep()
    }

    #[inline]
    pub fn keys(&self, index: u32, count: u32) -> Option<&'a [Key]> {
        self.keys.get(index as usize..(index + count) as usize)
    }

    #[inline]
    pub fn types(&self, index: u32, count: u32) -> Option<&'a [TypeId]> {
        self.types.get(index as usize..(index + count) as usize)
    }

    pub fn own(&mut self) -> Context<T> {
        Context {
            items: self.items,
            keep: self.keep,
            keys: self.keys,
            types: self.types,
        }
    }

    fn keep(&mut self) -> ControlFlow<()> {
        match self.keep {
            Keep::All => Continue(()),
            Keep::First(count) if self.items.len() < count.get() => Continue(()),
            Keep::First(count) => Break(self.items.truncate(count.get())),
            Keep::Last(count) => {
                self.items.drain(..self.items.len() - count.get());
                Continue(())
            }
        }
    }
}

impl<'a, E: Event> Receive<'a, E> {
    fn new(inner: &'a Inner, keep: Keep) -> Self {
        let ready = inner.ready.read();
        ready.buffers.fetch_add(1, Ordering::Relaxed);
        ready.seen.fetch_add(1, Ordering::Relaxed);
        Receive {
            keep,
            inner,
            head: ready.chunks.len(),
            events: VecDeque::new(),
            version: ready.version,
        }
    }

    pub fn clear(&mut self) {
        self.events.clear();
        while let Some(_) = self.next() {
            self.events.clear();
        }
    }

    pub fn with_event<F: Event>(self) -> Receive<'a, F> {
        let receive = ManuallyDrop::new(self);
        Receive {
            inner: receive.inner,
            keep: receive.keep,
            events: VecDeque::new(),
            head: receive.head,
            version: receive.version,
        }
    }

    fn process(&mut self, ready: &Ready) {
        for chunk in ready.chunks.range(self.head..) {
            let mut context = Context {
                items: &mut self.events,
                keep: &self.keep,
                keys: &chunk.keys,
                types: &chunk.types,
            };
            for event in chunk.events.iter() {
                if let Break(_) = E::process(event, context.own()) {
                    break;
                };
            }
        }
        self.head = ready.chunks.len();
    }
}

impl<E: Event> Iterator for Receive<'_, E> {
    type Item = E;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(event) = self.events.pop_front() {
            return Some(event);
        }

        let ready = self.inner.ready.read();
        debug_assert!(self.version + 1 >= ready.version);
        if self.version < ready.version {
            self.head -= ready.head;
            self.version = ready.version;
            ready.seen.fetch_add(1, Ordering::Relaxed);
        }
        self.process(&ready);
        if let Some(event) = self.events.pop_front() {
            return Some(event);
        }

        drop(ready);

        let mut ready = self.inner.ready.write();
        let chunk = ready.last.take().unwrap_or_default();
        let chunk = {
            // The time spent with the `pending` lock is minimized as much as possible since it may block other threads
            // that hold database locks. The tradeoff here is that the `ready` write lock may be taken in vain when `pending`
            // has no new events.
            let mut pending = self.inner.pending.lock();
            if pending.events.len() == 0 {
                ready.last = Some(chunk);
                return None;
            }
            replace(&mut *pending, chunk)
        };

        ready.chunks.push_back(chunk);
        let drain = ready.next;
        let buffers = *ready.buffers.get_mut();
        let seen = ready.seen.get_mut();
        if *seen >= buffers {
            *seen = 1;
            if let Some(mut chunk) = ready.chunks.drain(..drain).max_by_key(|chunk| {
                chunk.events.capacity() + chunk.keys.capacity() + chunk.types.capacity()
            }) {
                chunk.events.clear();
                chunk.keys.clear();
                chunk.types.clear();
                ready.last = Some(chunk);
            }
            ready.version += 1;
            ready.head = ready.next;
            ready.next = ready.chunks.len();
            self.head -= ready.head;
            self.version = ready.version;
        }

        let ready = RwLockWriteGuard::downgrade(ready);
        self.process(&ready);
        self.events.pop_front()
    }
}

impl<E> Drop for Receive<'_, E> {
    fn drop(&mut self) {
        let ready = self.inner.ready.read();
        ready.buffers.fetch_sub(1, Ordering::Relaxed);
        if self.version == ready.version {
            ready.seen.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

macro_rules! tuple {
    ($n:ident, $c:expr $(, $p:ident, $t:ident, $i:tt)*) => {
        impl<$($t: Listen,)*> Listen for ($($t,)*) {
            #[inline]
            fn on_create(&self, _keys: &[Key], _types: &[TypeId]) {
                $(self.$i.on_create(_keys, _types);)*
            }
            #[inline]
            fn on_destroy(&self, _keys: &[Key], _types: &[TypeId]) {
                $(self.$i.on_destroy(_keys, _types);)*
            }
            #[inline]
            fn on_modify(&self, _keys: &[Key], _add: &[TypeId], _remove: &[TypeId]) {
                $(self.$i.on_modify(_keys, _add, _remove);)*
            }
        }
    };
}
tuples!(tuple);

pub mod events {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    pub enum Type {
        Add(TypeId),
        Remove(TypeId),
    }

    #[derive(Clone, Copy, Debug)]
    pub struct Types {
        pub add: usize,
        pub remove: usize,
    }

    macro_rules! with {
        ($n:ident, $nw:ident, $on:ident, $on_key:ident, $on_type:ident, $on_types:ident, $on_key_type:ident, $on_key_types:ident) => {
            impl Broadcast {
                pub fn $n(&self) -> Receive<$on> {
                    self.on()
                }
                pub fn $nw(&self, keep: Keep) -> Receive<$on> {
                    self.on_with(keep)
                }
            }
            impl<'a> Receive<'a, $on> {
                pub fn with_key(self) -> Receive<'a, $on_key> {
                    self.with_event()
                }
                pub fn with_type<D: Datum>(self) -> Receive<'a, $on_type<D>> {
                    self.with_event()
                }
                pub fn with_types(self) -> Receive<'a, $on_types> {
                    self.with_event()
                }
            }
            impl<'a> Receive<'a, $on_key> {
                pub fn with_type<D: Datum>(self) -> Receive<'a, $on_key_type<D>> {
                    self.with_event()
                }
                pub fn with_types(self) -> Receive<'a, $on_key_types> {
                    self.with_event()
                }
            }
            impl<'a, D: Datum> Receive<'a, $on_type<D>> {
                pub fn with_key(self) -> Receive<'a, $on_key_type<D>> {
                    self.with_event()
                }
            }
            impl<'a> Receive<'a, $on_types> {
                pub fn with_key(self) -> Receive<'a, $on_key_types> {
                    self.with_event()
                }
            }

            impl Into<Key> for $on_key {
                #[inline]
                fn into(self) -> Key {
                    self.key
                }
            }

            impl<T> Into<Key> for $on_key_type<T> {
                #[inline]
                fn into(self) -> Key {
                    self.key
                }
            }

            impl Into<Key> for $on_key_types {
                #[inline]
                fn into(self) -> Key {
                    self.key
                }
            }
        };
    }

    macro_rules! event {
        (
            $n:ident, $nw:ident, $t:ty, $ts:ty,
            $on:ident, $on_key:ident, $on_type:ident, $on_types:ident, $on_key_type:ident, $on_key_types:ident,
            $raw:ident,
            $count:ident,
            $types:ident,
            $has:ident
        ) => {
            #[derive(Clone, Debug)]
            pub struct $on {
                pub keys: usize,
                pub types: $ts,
            }
            #[derive(Clone, Debug)]
            pub struct $on_key {
                pub key: Key,
                pub types: $ts,
            }
            #[derive(Clone, Debug)]
            pub struct $on_type<T> {
                pub keys: usize,
                _marker: PhantomData<T>,
            }
            #[derive(Clone, Debug)]
            pub struct $on_types {
                pub keys: usize,
                pub r#type: $t,
            }
            #[derive(Clone, Debug)]
            pub struct $on_key_type<T> {
                pub key: Key,
                _marker: PhantomData<T>,
            }
            #[derive(Clone, Debug)]
            pub struct $on_key_types {
                pub key: Key,
                pub r#type: $t,
            }

            with!(
                $n,
                $nw,
                $on,
                $on_key,
                $on_type,
                $on_types,
                $on_key_type,
                $on_key_types
            );

            impl Event for $on {
                #[inline]
                fn process(raw: &Raw, mut context: Context<Self>) -> ControlFlow<()> {
                    let &Raw::$raw { keys, types } = raw else { return Continue(()); };
                    let Some(types) = $count(types) else { return Continue(()); };
                    context.one(Self {
                        keys: keys.1 as _,
                        types: types,
                    })
                }
            }

            impl Event for $on_key {
                #[inline]
                fn process(raw: &Raw, mut context: Context<Self>) -> ControlFlow<()> {
                    let &Raw::$raw { keys, types } = raw else { return Continue(()); };
                    let Some(keys) = context.keys(keys.0, keys.1) else { return Continue(()); };
                    let Some(types) = $count(types) else { return Continue(()); };
                    context.all(keys.iter().map(|&key| Self { key, types }))
                }
            }

            impl Event for $on_types {
                #[inline]
                fn process(raw: &Raw, mut context: Context<Self>) -> ControlFlow<()> {
                    let &Raw::$raw { keys, types } = raw else { return Continue(()); };
                    let Some(types) = $types(&context, types) else { return Continue(()); };
                    context.all(types.map(|r#type| Self {
                        keys: keys.1 as _,
                        r#type,
                    }))
                }
            }

            impl Event for $on_key_types {
                #[inline]
                fn process(raw: &Raw, mut context: Context<Self>) -> ControlFlow<()> {
                    let &Raw::$raw { keys, types } = raw else { return Continue(()); };
                    let Some(keys) = context.keys(keys.0, keys.1) else { return Continue(()); };
                    let Some(types) = $types(&context, types) else { return Continue(()); };
                    for r#type in types {
                        context.all(keys.iter().map(move |&key| Self { key, r#type }))?;
                    }
                    Continue(())
                }
            }

            impl<D: Datum> Event for $on_type<D> {
                #[inline]
                fn process(raw: &Raw, mut context: Context<Self>) -> ControlFlow<()> {
                    let &Raw::$raw { keys, types } = raw else { return Continue(()); };
                    let Some(true) = $has::<D>(&context, types) else { return Continue(()); };
                    context.one(Self {
                        keys: keys.1 as _,
                        _marker: PhantomData,
                    })
                }
            }

            impl<D: Datum> Event for $on_key_type<D> {
                #[inline]
                fn process(raw: &Raw, mut context: Context<Self>) -> ControlFlow<()> {
                    let &Raw::$raw { keys, types } = raw else { return Continue(()); };
                    let Some(keys) = context.keys(keys.0, keys.1) else { return Continue(()); };
                    let Some(true) = $has::<D>(&context, types) else { return Continue(()); };
                    context.all(keys.iter().map(|&key| Self {
                        key,
                        _marker: PhantomData,
                    }))
                }
            }
        };
    }

    #[inline]
    fn create_destroy_count(types: (u32, u32)) -> Option<usize> {
        Some(types.1 as _)
    }
    #[inline]
    fn create_destroy_types<'a>(
        context: &Context<'a, impl Event>,
        types: (u32, u32),
    ) -> Option<impl Iterator<Item = TypeId> + 'a> {
        Some(context.types(types.0, types.1)?.iter().copied())
    }
    #[inline]
    fn create_destroy_has<D: Datum>(
        context: &Context<impl Event>,
        types: (u32, u32),
    ) -> Option<bool> {
        let types = context.types(types.0, types.1)?;
        Some(types.binary_search(&TypeId::of::<D>()).is_ok())
    }
    event!(
        on_create,
        on_create_with,
        TypeId,
        usize,
        OnCreate,
        OnCreateKey,
        OnCreateType,
        OnCreateTypes,
        OnCreateKeyType,
        OnCreateKeyTypes,
        Create,
        create_destroy_count,
        create_destroy_types,
        create_destroy_has
    );
    event!(
        on_destroy,
        on_destroy_with,
        TypeId,
        usize,
        OnDestroy,
        OnDestroyKey,
        OnDestroyType,
        OnDestroyTypes,
        OnDestroyKeyType,
        OnDestroyKeyTypes,
        Destroy,
        create_destroy_count,
        create_destroy_types,
        create_destroy_has
    );

    #[inline]
    fn modify_count(types: (u32, u32, u32)) -> Option<Types> {
        Some(Types {
            add: types.1 as _,
            remove: types.2 as _,
        })
    }
    #[inline]
    fn modify_types<'a>(
        context: &Context<'a, impl Event>,
        types: (u32, u32, u32),
    ) -> Option<impl Iterator<Item = Type> + 'a> {
        let add = context.types(types.0, types.1)?;
        let remove = context.types(types.0 + types.1, types.2)?;
        Some(
            add.iter()
                .copied()
                .map(Type::Add)
                .chain(remove.iter().copied().map(Type::Remove)),
        )
    }
    #[inline]
    fn modify_has<D: Datum>(context: &Context<impl Event>, types: (u32, u32, u32)) -> Option<bool> {
        let add = context.types(types.0, types.1)?;
        let remove = context.types(types.0 + types.1, types.2)?;
        Some(
            add.binary_search(&TypeId::of::<D>()).is_ok()
                || remove.binary_search(&TypeId::of::<D>()).is_ok(),
        )
    }
    event!(
        on_modify,
        on_modify_with,
        Type,
        Types,
        OnModify,
        OnModifyKey,
        OnModifyType,
        OnModifyTypes,
        OnModifyKeyType,
        OnModifyKeyTypes,
        Modify,
        modify_count,
        modify_types,
        modify_has
    );

    #[inline]
    fn add_count(types: (u32, u32, u32)) -> Option<usize> {
        Some(NonZeroUsize::new(types.1 as _)?.get())
    }
    #[inline]
    fn add_types<'a>(
        context: &Context<'a, impl Event>,
        types: (u32, u32, u32),
    ) -> Option<impl Iterator<Item = TypeId> + 'a> {
        Some(context.types(types.0, types.1)?.iter().copied())
    }
    #[inline]
    fn add_has<D: Datum>(context: &Context<impl Event>, types: (u32, u32, u32)) -> Option<bool> {
        let types = context.types(types.0, types.1)?;
        Some(types.binary_search(&TypeId::of::<D>()).is_ok())
    }
    event!(
        on_add,
        on_add_with,
        TypeId,
        usize,
        OnAdd,
        OnAddKey,
        OnAddType,
        OnAddTypes,
        OnAddKeyType,
        OnAddKeyTypes,
        Modify,
        add_count,
        add_types,
        add_has
    );

    #[inline]
    fn remove_count(types: (u32, u32, u32)) -> Option<usize> {
        Some(NonZeroUsize::new(types.2 as _)?.get())
    }
    #[inline]
    fn remove_types<'a>(
        context: &Context<'a, impl Event>,
        types: (u32, u32, u32),
    ) -> Option<impl Iterator<Item = TypeId> + 'a> {
        Some(context.types(types.0 + types.1, types.2)?.iter().copied())
    }
    #[inline]
    fn remove_has<D: Datum>(context: &Context<impl Event>, types: (u32, u32, u32)) -> Option<bool> {
        let types = context.types(types.0 + types.1, types.2)?;
        Some(types.binary_search(&TypeId::of::<D>()).is_ok())
    }
    event!(
        on_remove,
        on_remove_with,
        TypeId,
        usize,
        OnRemove,
        OnRemoveKey,
        OnRemoveType,
        OnRemoveTypes,
        OnRemoveKeyType,
        OnRemoveKeyTypes,
        Modify,
        remove_count,
        remove_types,
        remove_has
    );
}
