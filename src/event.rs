use crate::{
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
    ops::ControlFlow::{self, *},
    sync::atomic::{AtomicU64, AtomicUsize, Ordering::*},
};

pub trait Event: Sized {
    fn declare(context: DeclareContext);
    fn process(raw: &Raw, context: ProcessContext<Self>) -> ControlFlow<()>;
}

// pub trait Eventz: Sized {
//     fn declare(context: DeclareContext);
//     fn process<C: Collect<Self>>(
//         collect: &mut C,
//         raws: &[Raw],
//         keys: &[Key],
//         tables: &Tables,
//     ) -> ControlFlow<()>;
// }

// pub trait Collect<E> {
//     fn one(&mut self, event: E) -> ControlFlow<()>;
//     fn all<I: IntoIterator<Item = E>>(&mut self, events: I) -> ControlFlow<()>;
// }

// impl<E: Event> Collect<E> for Receive<'_, E> {
//     #[inline]
//     fn one(&mut self, event: E) -> ControlFlow<()> {
//         self.buffer.push_back(event);
//         if self.keep.apply(&mut self.buffer) {
//             Continue(())
//         } else {
//             Break(())
//         }
//     }

//     #[inline]
//     fn all<I: IntoIterator<Item = E>>(&mut self, events: I) -> ControlFlow<()> {
//         self.buffer.extend(events);
//         if self.keep.apply(&mut self.buffer) {
//             Continue(())
//         } else {
//             Break(())
//         }
//     }
// }

pub struct Events {
    ready: RwLock<Ready>,
    pending: Mutex<Chunk>,
    // 1 bit closed, 21 bits receivers, 21 bits requires keys, 21 bits requires types
    create: AtomicU64,
    destroy: AtomicU64,
    modify: AtomicU64,
}

pub struct Receive<'a, E: Event> {
    database: &'a Database,
    keep: Keep,
    buffer: VecDeque<E>,
    head: usize,
    version: usize,
}

#[derive(Clone, Copy, Default)]
pub enum Keep {
    #[default]
    All,
    First(usize),
    Last(usize),
}

pub struct DeclareContext<'a> {
    values: &'a mut [bool; 9],
}

pub struct ProcessContext<'a, T> {
    items: &'a mut VecDeque<T>,
    keep: Keep,
    keys: &'a [Key],
    tables: &'a Tables,
}

#[derive(Clone, Copy)]
pub enum Raw {
    Create {
        keys: (u32, u32),
        table: u32,
    },
    Destroy {
        keys: (u32, u32),
        table: u32,
    },
    Modify {
        keys: (u32, u32),
        tables: (u32, u32),
    },
}

enum Channel {
    Create,
    Destroy,
    Modify,
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

impl Database {
    pub fn on<E: Event>(&self) -> Receive<E> {
        Receive::new(self)
    }
}

impl Events {
    pub fn new() -> Self {
        Self {
            pending: Default::default(),
            ready: Default::default(),
            create: AtomicU64::new(0),
            destroy: AtomicU64::new(0),
            modify: AtomicU64::new(0),
        }
    }

    pub fn open(&self) -> bool {
        self.open_with(Channel::Create)
            | self.open_with(Channel::Destroy)
            | self.open_with(Channel::Modify)
    }

    pub fn close(&self) -> bool {
        self.close_with(Channel::Create)
            | self.close_with(Channel::Destroy)
            | self.close_with(Channel::Modify)
    }

    #[inline]
    pub(crate) fn emit_create(&self, keys: &[Key], table: &Table) {
        let values = decompose(self.create.load(Relaxed));
        if values.0 || values.1 == 0 {
            return;
        }

        let mut pending = self.pending.lock();
        let index = pending.keys.len();
        pending.events.push(Raw::Create {
            keys: (index as _, keys.len() as _),
            table: table.index(),
        });
        if values.2 > 0 {
            pending.keys.extend_from_slice(keys);
        }
    }

    #[inline]
    pub(crate) fn emit_destroy(&self, keys: &[Key], table: &Table) {
        let values = decompose(self.destroy.load(Relaxed));
        if values.0 || values.1 == 0 {
            return;
        }

        let mut pending = self.pending.lock();
        let index = pending.keys.len();
        pending.events.push(Raw::Destroy {
            keys: (index as _, keys.len() as _),
            table: table.index(),
        });
        if values.2 > 0 {
            pending.keys.extend_from_slice(keys);
        }
    }

    #[inline]
    pub(crate) fn emit_modify(&self, keys: &[Key], tables: (&Table, &Table)) {
        let values = decompose(self.modify.load(Relaxed));
        if values.0 || values.1 == 0 {
            return;
        }

        let mut pending = self.pending.lock();
        let index = pending.keys.len();
        pending.events.push(Raw::Modify {
            keys: (index as _, keys.len() as _),
            tables: (tables.0.index(), tables.1.index()),
        });
        if values.2 > 0 {
            pending.keys.extend_from_slice(keys);
        }
    }

    fn open_with(&self, channel: Channel) -> bool {
        let value = recompose(false, u32::MAX, u32::MAX, u32::MAX);
        decompose(match channel {
            Channel::Create => self.create.fetch_and(value, Relaxed),
            Channel::Destroy => self.destroy.fetch_and(value, Relaxed),
            Channel::Modify => self.modify.fetch_and(value, Relaxed),
        })
        .0 == true
    }

    fn close_with(&self, channel: Channel) -> bool {
        let value = recompose(true, 0, 0, 0);
        decompose(match channel {
            Channel::Create => self.create.fetch_or(value, Relaxed),
            Channel::Destroy => self.destroy.fetch_or(value, Relaxed),
            Channel::Modify => self.modify.fetch_or(value, Relaxed),
        })
        .0 == false
    }
}

impl Drop for Events {
    fn drop(&mut self) {
        self.close();
    }
}

impl DeclareContext<'_> {
    pub fn create(&mut self, keys: bool, types: bool) {
        self.values[0] |= true;
        self.values[1] |= keys;
        self.values[2] |= types;
    }

    pub fn destroy(&mut self, keys: bool, types: bool) {
        self.values[3] |= true;
        self.values[4] |= keys;
        self.values[5] |= types;
    }

    pub fn modify(&mut self, keys: bool, types: bool) {
        self.values[6] |= true;
        self.values[7] |= keys;
        self.values[8] |= types;
    }
}

impl<'a, T> ProcessContext<'a, T> {
    #[inline]
    pub fn one(&mut self, item: T) -> ControlFlow<()> {
        self.items.push_back(item);
        if self.keep.apply(self.items) {
            Continue(())
        } else {
            Break(())
        }
    }

    #[inline]
    pub fn all<I: IntoIterator<Item = T>>(&mut self, items: I) -> ControlFlow<()> {
        self.items.extend(items);
        if self.keep.apply(self.items) {
            Continue(())
        } else {
            Break(())
        }
    }

    #[inline]
    pub fn keys(&self, index: u32, count: u32) -> Option<&'a [Key]> {
        self.keys.get(index as usize..(index + count) as usize)
    }

    #[inline]
    pub fn table(&self, index: u32) -> Option<&'a Table> {
        self.tables.get(index as _).ok()
    }

    pub fn own(&mut self) -> ProcessContext<T> {
        ProcessContext {
            items: self.items,
            keep: self.keep,
            keys: self.keys,
            tables: self.tables,
        }
    }
}

impl<'a, E: Event> Receive<'a, E> {
    pub fn clear(&mut self) {
        let keep = self.keep(Keep::First(0));
        self.next();
        self.keep(keep);
    }

    pub fn keep(&mut self, keep: Keep) -> Keep {
        let keep = replace(&mut self.keep, keep);
        self.keep.apply(&mut self.buffer);
        keep
    }

    pub fn with<F: Event>(self) -> Receive<'a, F> {
        let receive = ManuallyDrop::new(self);
        Self::remove_declare(receive.database.events());
        Receive::<F>::add_declare(receive.database.events());
        Receive {
            keep: receive.keep,
            database: receive.database,
            head: receive.head,
            buffer: VecDeque::new(),
            version: receive.version,
        }
    }

    fn new(database: &'a Database) -> Self {
        Self::add_declare(database.events());
        let ready = database.events().ready.read();
        ready.buffers.fetch_add(1, Relaxed);
        ready.seen.fetch_add(1, Relaxed);
        Receive {
            keep: Keep::All,
            database,
            head: ready.chunks.len(),
            buffer: VecDeque::new(),
            version: ready.version,
        }
    }

    fn declare() -> [bool; 9] {
        let mut values = [false; 9];
        E::declare(DeclareContext {
            values: &mut values,
        });
        values
    }

    fn add_declare(events: &Events) {
        Self::update(events, Self::declare(), |source, target| {
            target.fetch_add(source, Relaxed);
        });
    }

    fn remove_declare(events: &Events) {
        Self::update(events, Self::declare(), |source, target| {
            target.fetch_sub(source, Relaxed);
        });
    }

    fn update(events: &Events, values: [bool; 9], update: impl Fn(u64, &AtomicU64)) {
        let mut index = 0;
        for channel in [&events.create, &events.destroy, &events.modify] {
            if values[index] {
                let keys = if values[index + 1] { 1 } else { 0 };
                let types = if values[index + 2] { 1 } else { 0 };
                update(recompose(false, 1, keys, types), channel);
            }
            index += 3;
        }
    }

    fn process(&mut self, ready: &Ready) -> ControlFlow<()> {
        let head = replace(&mut self.head, ready.chunks.len());
        for chunk in ready.chunks.range(head..) {
            let mut context = ProcessContext {
                items: &mut self.buffer,
                keep: self.keep,
                keys: &chunk.keys,
                tables: self.database.tables(),
            };
            for event in chunk.events.iter() {
                E::process(event, context.own())?;
            }
        }
        Continue(())
    }
}

impl<E: Event> Iterator for Receive<'_, E> {
    type Item = E;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(event) = self.buffer.pop_front() {
            return Some(event);
        }

        let ready = self.database.events().ready.read();
        debug_assert!(self.version + 1 >= ready.version);
        if self.version < ready.version {
            self.head -= ready.head;
            self.version = ready.version;
            ready.seen.fetch_add(1, Relaxed);
        }
        self.process(&ready);
        if let Some(event) = self.buffer.pop_front() {
            return Some(event);
        }

        drop(ready);

        let mut ready = self.database.events().ready.write();
        let chunk = ready.last.take().unwrap_or_default();
        let chunk = {
            // The time spent with the `pending` lock is minimized as much as possible since it may block other threads
            // that hold database locks. The tradeoff here is that the `ready` write lock may be taken in vain when `pending`
            // has no new events.
            let mut pending = self.database.events().pending.lock();
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
        self.buffer.pop_front()
    }
}

impl<E: Event> Drop for Receive<'_, E> {
    fn drop(&mut self) {
        Self::remove_declare(self.database.events());
        let ready = self.database.events().ready.read();
        ready.buffers.fetch_sub(1, Relaxed);
        if self.version == ready.version {
            ready.seen.fetch_sub(1, Relaxed);
        }
    }
}

impl Keep {
    fn apply<T>(&self, items: &mut VecDeque<T>) -> bool {
        match *self {
            Keep::All => true,
            Keep::First(0) | Keep::Last(0) => {
                items.clear();
                false
            }
            Keep::First(count) if items.len() < count => true,
            Keep::First(count) => {
                items.truncate(count);
                false
            }
            Keep::Last(count) if items.len() < count => true,
            Keep::Last(count) => {
                items.drain(..items.len() - count);
                true
            }
        }
    }
}

const MASK: u32 = 0x001FFFFF;

#[inline]
const fn decompose(value: u64) -> (bool, u32, u32, u32) {
    (
        value >> 63 == 1,
        (value >> 42) as u32 & MASK,
        (value >> 21) as u32 & MASK,
        (value as u32 & MASK),
    )
}

#[inline]
const fn recompose(close: bool, receivers: u32, keys: u32, types: u32) -> u64 {
    let close = if close { 1u8 } else { 0u8 };
    ((close as u64) << 63)
        | (((receivers & MASK) as u64) << 42)
        | (((keys & MASK) as u64) << 21)
        | ((types & MASK) as u64)
}

pub mod events {
    use crate::core::utility::{sorted_difference, sorted_symmetric_difference_by};

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

    impl Type {
        #[inline]
        pub fn identifier(&self) -> TypeId {
            match self {
                Type::Add(identifier) => *identifier,
                Type::Remove(identifier) => *identifier,
            }
        }
    }

    macro_rules! with {
        ($n:ident, $on:ident, $on_key:ident, $on_type:ident, $on_types:ident, $on_key_type:ident, $on_key_types:ident) => {
            impl Database {
                pub fn $n(&self) -> Receive<$on> {
                    self.on()
                }
            }
            impl<'a> Receive<'a, $on> {
                pub fn with_key(self) -> Receive<'a, $on_key> {
                    self.with()
                }
                pub fn with_type<D: Datum>(self) -> Receive<'a, $on_type<D>> {
                    self.with()
                }
                pub fn with_types(self) -> Receive<'a, $on_types> {
                    self.with()
                }
            }
            impl<'a> Receive<'a, $on_key> {
                pub fn with_type<D: Datum>(self) -> Receive<'a, $on_key_type<D>> {
                    self.with()
                }
                pub fn with_types(self) -> Receive<'a, $on_key_types> {
                    self.with()
                }
            }
            impl<'a, D: Datum> Receive<'a, $on_type<D>> {
                pub fn with_key(self) -> Receive<'a, $on_key_type<D>> {
                    self.with()
                }
            }
            impl<'a> Receive<'a, $on_types> {
                pub fn with_key(self) -> Receive<'a, $on_key_types> {
                    self.with()
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
            $n:ident, $t:ty, $ts:ty, $d:ident, $table:ident,
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
                $on,
                $on_key,
                $on_type,
                $on_types,
                $on_key_type,
                $on_key_types
            );

            impl Event for $on {
                fn declare(mut context: DeclareContext) {
                    context.$d(false, false)
                }

                #[inline]
                fn process(raw: &Raw, mut context: ProcessContext<Self>) -> ControlFlow<()> {
                    let &Raw::$raw { keys, $table } = raw else { return Continue(()); };
                    let Some(types) = $count(&context, $table) else { return Continue(()); };
                    context.one(Self {
                        keys: keys.1 as _,
                        types: types,
                    })
                }
            }

            impl Event for $on_key {
                fn declare(mut context: DeclareContext) {
                    context.$d(true, false)
                }

                #[inline]
                fn process(raw: &Raw, mut context: ProcessContext<Self>) -> ControlFlow<()> {
                    let &Raw::$raw { keys, $table } = raw else { return Continue(()); };
                    let Some(keys) = context.keys(keys.0, keys.1) else { return Continue(()); };
                    let Some(types) = $count(&context, $table) else { return Continue(()); };
                    context.all(keys.iter().map(|&key| Self { key, types }))
                }
            }

            impl Event for $on_types {
                fn declare(mut context: DeclareContext) {
                    context.$d(false, true)
                }

                #[inline]
                fn process(raw: &Raw, mut context: ProcessContext<Self>) -> ControlFlow<()> {
                    let &Raw::$raw { keys, $table } = raw else { return Continue(()); };
                    let Some(types) = $types(&context, $table) else { return Continue(()); };
                    context.all(types.map(|r#type| Self {
                        keys: keys.1 as _,
                        r#type,
                    }))
                }
            }

            impl Event for $on_key_types {
                fn declare(mut context: DeclareContext) {
                    context.$d(true, true)
                }

                #[inline]
                fn process(raw: &Raw, mut context: ProcessContext<Self>) -> ControlFlow<()> {
                    let &Raw::$raw { keys, $table } = raw else { return Continue(()); };
                    let Some(keys) = context.keys(keys.0, keys.1) else { return Continue(()); };
                    let Some(types) = $types(&context, $table) else { return Continue(()); };
                    for r#type in types {
                        context.all(keys.iter().map(move |&key| Self { key, r#type }))?;
                    }
                    Continue(())
                }
            }

            impl<D: Datum> Event for $on_type<D> {
                fn declare(mut context: DeclareContext) {
                    context.$d(false, true)
                }

                #[inline]
                fn process(raw: &Raw, mut context: ProcessContext<Self>) -> ControlFlow<()> {
                    let &Raw::$raw { keys, $table } = raw else { return Continue(()); };
                    let Some(true) = $has::<D, _>(&context, $table) else { return Continue(()); };
                    context.one(Self {
                        keys: keys.1 as _,
                        _marker: PhantomData,
                    })
                }
            }

            impl<D: Datum> Event for $on_key_type<D> {
                fn declare(mut context: DeclareContext) {
                    context.$d(true, true)
                }

                #[inline]
                fn process(raw: &Raw, mut context: ProcessContext<Self>) -> ControlFlow<()> {
                    let &Raw::$raw { keys, $table } = raw else { return Continue(()); };
                    let Some(keys) = context.keys(keys.0, keys.1) else { return Continue(()); };
                    let Some(true) = $has::<D, _>(&context, $table) else { return Continue(()); };
                    context.all(keys.iter().map(|&key| Self {
                        key,
                        _marker: PhantomData,
                    }))
                }
            }
        };
    }

    #[inline]
    fn create_destroy_count<T>(context: &ProcessContext<T>, table: u32) -> Option<usize> {
        Some(context.table(table)?.columns().len())
    }
    #[inline]
    fn create_destroy_types<'a, T>(
        context: &ProcessContext<'a, T>,
        table: u32,
    ) -> Option<impl Iterator<Item = TypeId> + 'a> {
        Some(context.table(table)?.types())
    }
    #[inline]
    fn create_destroy_has<D: Datum, T>(context: &ProcessContext<T>, table: u32) -> Option<bool> {
        Some(context.table(table)?.has::<D>())
    }
    event!(
        on_create,
        TypeId,
        usize,
        create,
        table,
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
        TypeId,
        usize,
        destroy,
        table,
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
    fn modify_count<T>(context: &ProcessContext<T>, tables: (u32, u32)) -> Option<Types> {
        let (add, remove) =
            modify_types(context, tables)?.fold((0, 0), |counts, r#type| match r#type {
                Type::Add(_) => (counts.0 + 1, counts.1),
                Type::Remove(_) => (counts.0, counts.1 + 1),
            });
        Some(Types { add, remove })
    }
    #[inline]
    fn modify_types<'a, T>(
        context: &ProcessContext<'a, T>,
        tables: (u32, u32),
    ) -> Option<impl Iterator<Item = Type> + 'a> {
        let source = context.table(tables.0)?;
        let target = context.table(tables.1)?;
        Some(sorted_symmetric_difference_by(
            |left, right| Ord::cmp(&left.identifier(), &right.identifier()),
            source.types().map(Type::Remove),
            target.types().map(Type::Add),
        ))
    }
    #[inline]
    fn modify_has<D: Datum, T>(context: &ProcessContext<T>, tables: (u32, u32)) -> Option<bool> {
        let source = context.table(tables.0)?;
        let target = context.table(tables.1)?;
        Some(source.has::<D>() != target.has::<D>())
    }
    event!(
        on_modify,
        Type,
        Types,
        modify,
        tables,
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
    fn add_count<T>(context: &ProcessContext<T>, tables: (u32, u32)) -> Option<usize> {
        Some(add_types(context, tables)?.count())
    }
    #[inline]
    fn add_types<'a, T>(
        context: &ProcessContext<'a, T>,
        tables: (u32, u32),
    ) -> Option<impl Iterator<Item = TypeId> + 'a> {
        let source = context.table(tables.0)?;
        let target = context.table(tables.1)?;
        Some(sorted_difference(target.types(), source.types()))
    }
    #[inline]
    fn add_has<D: Datum, T>(context: &ProcessContext<T>, tables: (u32, u32)) -> Option<bool> {
        let source = context.table(tables.0)?;
        let target = context.table(tables.1)?;
        Some(!source.has::<D>() && target.has::<D>())
    }
    event!(
        on_add,
        TypeId,
        usize,
        modify,
        tables,
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
    fn remove_count<T>(context: &ProcessContext<T>, tables: (u32, u32)) -> Option<usize> {
        Some(remove_types(context, tables)?.count())
    }
    #[inline]
    fn remove_types<'a, T>(
        context: &ProcessContext<'a, T>,
        tables: (u32, u32),
    ) -> Option<impl Iterator<Item = TypeId> + 'a> {
        let source = context.table(tables.0)?;
        let target = context.table(tables.1)?;
        Some(sorted_difference(source.types(), target.types()))
    }
    #[inline]
    fn remove_has<D: Datum, T>(context: &ProcessContext<T>, tables: (u32, u32)) -> Option<bool> {
        let source = context.table(tables.0)?;
        let target = context.table(tables.1)?;
        Some(source.has::<D>() && !target.has::<D>())
    }
    event!(
        on_remove,
        TypeId,
        usize,
        modify,
        tables,
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
