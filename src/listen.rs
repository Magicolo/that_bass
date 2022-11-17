use crate::{
    core::{
        tuples,
        utility::{sorted_difference, sorted_symmetric_difference},
    },
    key::Key,
    table::{Table, Tables},
    Database, Datum,
};
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
    fn on_create(&self, keys: &[Key], table: &Table);
    fn on_destroy(&self, keys: &[Key], table: &Table);
    fn on_modify(&self, keys: &[Key], source: &Table, target: &Table);
}

pub trait Event: Sized {
    fn process(raw: &Raw, context: Context<Self>) -> ControlFlow<(), bool>;
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
    tables: &'a Tables,
}

#[derive(Clone, Copy)]
pub enum Raw {
    Create {
        index: u32,
        count: u32,
        table: u32,
    },
    Destroy {
        index: u32,
        count: u32,
        table: u32,
    },
    Modify {
        index: u32,
        count: u32,
        source: u32,
        target: u32,
    },
}

struct Inner {
    ready: RwLock<Ready>,
    pending: Mutex<Chunk>,
    tables: Arc<Tables>,
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
            tables: self.inner.tables.clone(),
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
    fn on_create(&self, keys: &[Key], table: &Table) {
        let mut pending = self.0.pending.lock();
        let index = pending.keys.len();
        pending.events.push(Raw::Create {
            index: index as _,
            count: keys.len() as _,
            table: table.index(),
        });
        pending.keys.extend_from_slice(keys);
    }
    #[inline]
    fn on_destroy(&self, keys: &[Key], table: &Table) {
        let mut pending = self.0.pending.lock();
        let index = pending.keys.len();
        pending.events.push(Raw::Destroy {
            index: index as _,
            count: keys.len() as _,
            table: table.index(),
        });
        pending.keys.extend_from_slice(keys);
    }
    #[inline]
    fn on_modify(&self, keys: &[Key], source: &Table, target: &Table) {
        let mut pending = self.0.pending.lock();
        let index = pending.keys.len();
        pending.events.push(Raw::Modify {
            index: index as _,
            count: keys.len() as _,
            source: source.index(),
            target: target.index(),
        });
        pending.keys.extend_from_slice(keys);
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
    pub fn table(&self, index: u32) -> Option<&'a Table> {
        self.tables.get(index as usize).ok()
    }

    pub fn own(&mut self) -> Context<T> {
        Context {
            items: self.items,
            keep: self.keep,
            keys: self.keys,
            tables: self.tables,
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
                tables: &self.inner.tables,
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
            if let Some(mut chunk) = ready
                .chunks
                .drain(..drain)
                .max_by_key(|chunk| chunk.events.capacity() + chunk.keys.capacity())
            {
                chunk.events.clear();
                chunk.keys.clear();
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
            fn on_create(&self, _keys: &[Key], _table: &Table) {
                $(self.$i.on_create(_keys, _table);)*
            }
            #[inline]
            fn on_destroy(&self, _keys: &[Key], _table: &Table) {
                $(self.$i.on_destroy(_keys, _table);)*
            }
            #[inline]
            fn on_modify(&self, _keys: &[Key], _source: &Table, _target: &Table) {
                $(self.$i.on_modify(_keys, _source, _target);)*
            }
        }
    };
}
tuples!(tuple);

pub mod events {
    use super::*;

    macro_rules! event {
        (
            $n:ident, $nw:ident,
            $on:ident, $on_key:ident, $on_type:ident, $on_types:ident, $on_key_type:ident, $on_key_types:ident,
            $raw:ident($($f:ident),*),
            $valid:ident,
            $types:ident,
            $check:ident
        ) => {
            #[derive(Clone, Debug)]
            pub struct $on {
                pub count: usize,
            }
            #[derive(Clone, Debug)]
            pub struct $on_key {
                pub key: Key,
            }
            #[derive(Clone, Debug)]
            pub struct $on_type<T> {
                pub count: usize,
                _marker: PhantomData<T>,
            }
            #[derive(Clone, Debug)]
            pub struct $on_types {
                pub count: usize,
                pub identifier: TypeId,
            }
            #[derive(Clone, Debug)]
            pub struct $on_key_type<T> {
                pub key: Key,
                _marker: PhantomData<T>,
            }
            #[derive(Clone, Debug)]
            pub struct $on_key_types {
                pub key: Key,
                pub identifier: TypeId,
            }

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


            impl Event for $on {
                #[inline]
                fn process(raw: &Raw, mut context: Context<Self>) -> ControlFlow<(), bool> {
                    let &Raw::$raw { count, $($f,)* .. } = raw else { return Continue(false); };
                    let Some(_) = $valid(&context $(,$f)*) else { return Continue(false); };
                    context.one(Self { count: count as _ })?;
                    Continue(true)
                }
            }

            impl Event for $on_key {
                #[inline]
                fn process(raw: &Raw, mut context: Context<Self>) -> ControlFlow<(), bool> {
                    let &Raw::$raw { index, count, $($f,)* .. } = raw else { return Continue(false); };
                    let Some(keys) = context.keys(index, count) else { return Continue(false); };
                    let Some(_) = $valid(&context $(,$f)*) else { return Continue(false); };
                    context.all(keys.iter().map(|&key| Self { key }))?;
                    Continue(true)
                }
            }

            impl Event for $on_types {
                #[inline]
                fn process(raw: &Raw, mut context: Context<Self>) -> ControlFlow<(), bool> {
                    let &Raw::$raw { count, $($f,)* .. } = raw else { return Continue(false); };
                    let Some(types) = $types(&context $(,$f)*) else { return Continue(false); };
                    context.all(types.map(|identifier| Self {
                        count: count as _,
                        identifier,
                    }))?;
                    Continue(true)
                }
            }

            impl Event for $on_key_types {
                #[inline]
                fn process(raw: &Raw, mut context: Context<Self>) -> ControlFlow<(), bool> {
                    let &Raw::$raw { index, count, $($f,)* .. } = raw else { return Continue(false); };
                    let Some(keys) = context.keys(index, count) else { return Continue(false); };
                    let Some(types) = $types(&context $(,$f)*) else { return Continue(false); };
                    for identifier in types {
                        context.all(keys.iter().map(move |&key| Self { key, identifier }))?;
                    }
                    Continue(true)
                }
            }

            impl<D: Datum> Event for $on_type<D> {
                #[inline]
                fn process(raw: &Raw, mut context: Context<Self>) -> ControlFlow<(), bool> {
                    let &Raw::$raw { count, $($f,)* .. } = raw else { return Continue(false); };
                    let Some(true) = $check::<D>(&context $(,$f)*) else { return Continue(false); };
                    context.one(Self { count: count as _, _marker: PhantomData, })?;
                    Continue(true)
                }
            }

            impl<D: Datum> Event for $on_key_type<D> {
                #[inline]
                fn process(raw: &Raw, mut context: Context<Self>) -> ControlFlow<(), bool> {
                    let &Raw::$raw { index, count, $($f,)* .. } = raw else { return Continue(false); };
                    let Some(keys) = context.keys(index, count) else { return Continue(false); };
                    let Some(true) = $check::<D>(&context $(,$f)*) else { return Continue(false); };
                    context.all(keys.iter().map(|&key| Self { key, _marker: PhantomData }))?;
                    Continue(true)
                }
            }
        };
    }

    #[inline]
    fn create_destroy_valid(_: &Context<impl Event>, _: u32) -> Option<()> {
        Some(())
    }
    #[inline]
    fn create_destroy_types<'a>(
        context: &Context<'a, impl Event>,
        table: u32,
    ) -> Option<impl Iterator<Item = TypeId> + 'a> {
        Some(context.table(table)?.types())
    }
    #[inline]
    fn create_destroy_check<D: Datum>(context: &Context<impl Event>, table: u32) -> Option<bool> {
        Some(context.table(table)?.has::<D>())
    }
    event!(
        on_create,
        on_create_with,
        OnCreate,
        OnCreateKey,
        OnCreateType,
        OnCreateTypes,
        OnCreateKeyType,
        OnCreateKeyTypes,
        Create(table),
        create_destroy_valid,
        create_destroy_types,
        create_destroy_check
    );
    event!(
        on_destroy,
        on_destroy_with,
        OnDestroy,
        OnDestroyKey,
        OnDestroyType,
        OnDestroyTypes,
        OnDestroyKeyType,
        OnDestroyKeyTypes,
        Destroy(table),
        create_destroy_valid,
        create_destroy_types,
        create_destroy_check
    );

    #[inline]
    fn modify_valid(context: &Context<impl Event>, source: u32, target: u32) -> Option<()> {
        let source = context.table(source)?;
        let target = context.table(target)?;
        (source.index() != target.index()).then_some(())
    }
    #[inline]
    fn modify_types<'a>(
        context: &Context<'a, impl Event>,
        source: u32,
        target: u32,
    ) -> Option<impl Iterator<Item = TypeId> + 'a> {
        let source = context.table(source)?;
        let target = context.table(target)?;
        Some(sorted_symmetric_difference(source.types(), target.types()))
    }
    #[inline]
    fn modify_check<D: Datum>(
        context: &Context<impl Event>,
        source: u32,
        target: u32,
    ) -> Option<bool> {
        let source = context.table(source)?;
        let target = context.table(target)?;
        Some(source.has::<D>() != target.has::<D>())
    }
    event!(
        on_modify,
        on_modify_with,
        OnModify,
        OnModifyKey,
        OnModifyType,
        OnModifyTypes,
        OnModifyKeyType,
        OnModifyKeyTypes,
        Modify(source, target),
        modify_valid,
        modify_types,
        modify_check
    );

    #[inline]
    fn add_valid(context: &Context<impl Event>, source: u32, target: u32) -> Option<()> {
        let source = context.table(source)?;
        let target = context.table(target)?;
        sorted_difference(target.types(), source.types()).next()?;
        Some(())
    }
    #[inline]
    fn add_types<'a>(
        context: &Context<'a, impl Event>,
        source: u32,
        target: u32,
    ) -> Option<impl Iterator<Item = TypeId> + 'a> {
        let source = context.table(source)?;
        let target = context.table(target)?;
        Some(sorted_difference(target.types(), source.types()))
    }
    #[inline]
    fn add_check<D: Datum>(
        context: &Context<impl Event>,
        source: u32,
        target: u32,
    ) -> Option<bool> {
        let source = context.table(source)?;
        let target = context.table(target)?;
        Some(!source.has::<D>() && target.has::<D>())
    }
    event!(
        on_add,
        on_add_with,
        OnAdd,
        OnAddKey,
        OnAddType,
        OnAddTypes,
        OnAddKeyType,
        OnAddKeyTypes,
        Modify(source, target),
        add_valid,
        add_types,
        add_check
    );

    #[inline]
    fn remove_valid(context: &Context<impl Event>, source: u32, target: u32) -> Option<()> {
        let source = context.table(source)?;
        let target = context.table(target)?;
        sorted_difference(source.types(), target.types()).next()?;
        Some(())
    }
    #[inline]
    fn remove_types<'a>(
        context: &Context<'a, impl Event>,
        source: u32,
        target: u32,
    ) -> Option<impl Iterator<Item = TypeId> + 'a> {
        let source = context.table(source)?;
        let target = context.table(target)?;
        Some(sorted_difference(source.types(), target.types()))
    }
    #[inline]
    fn remove_check<D: Datum>(
        context: &Context<impl Event>,
        source: u32,
        target: u32,
    ) -> Option<bool> {
        let source = context.table(source)?;
        let target = context.table(target)?;
        Some(source.has::<D>() && !target.has::<D>())
    }
    event!(
        on_remove,
        on_remove_with,
        OnRemove,
        OnRemoveKey,
        OnRemoveType,
        OnRemoveTypes,
        OnRemoveKeyType,
        OnRemoveKeyTypes,
        Modify(source, target),
        remove_valid,
        remove_types,
        remove_check
    );
}
