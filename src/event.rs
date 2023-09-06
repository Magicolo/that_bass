use crate::{
    core::utility::{get_unchecked, sorted_difference, sorted_symmetric_difference_by},
    key::Key,
    table::{self, Table},
    Database, Datum,
};
use parking_lot::{Mutex, RwLock, RwLockWriteGuard};
use std::{
    any::TypeId,
    array::from_ref,
    collections::VecDeque,
    marker::PhantomData,
    mem::replace,
    ops::ControlFlow::{self, *},
    sync::atomic::{AtomicU64, AtomicUsize, Ordering::*},
};

pub trait Event: Sized {
    type Events<'a>: Iterator<Item = Self> + 'a;

    fn declare(context: DeclareContext);
    fn process<'a>(events: &'a [Raw], context: ProcessContext<'a>) -> Self::Events<'a>;
}

#[derive(Clone)]
pub struct Events<'a>(&'a State, table::Tables<'a>);

pub(crate) struct State {
    ready: RwLock<Ready>,
    pending: Mutex<Chunk>,
    create: AtomicU64,
    destroy: AtomicU64,
    modify: AtomicU64,
}

pub struct Listen<'d, E: Event> {
    events: Events<'d>,
    cursor: Cursor,
    buffer: Buffer<E>,
}

pub struct Cursor {
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
    values: &'a mut [bool; 6],
}

#[derive(Clone)]
pub struct ProcessContext<'a> {
    keys: &'a [Key],
    tables: &'a table::Tables<'a>,
}

#[derive(Clone, Copy, Debug)]
pub struct Tables {
    pub source: u32,
    pub target: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Keys {
    pub index: u32,
    pub count: u32,
}

#[derive(Clone, Copy, Debug)]
pub enum Raw {
    Create { keys: Keys, table: u32 },
    Destroy { keys: Keys, table: u32 },
    Modify { keys: Keys, tables: Tables },
}

struct Buffer<T>(VecDeque<T>, Keep);

#[derive(Default)]
struct Ready {
    chunks: VecDeque<Chunk>,
    // TODO: Use something like this for messages.
    // - Listen could provide direct references into raw which could be processed in place into another message type.
    // - When a chunk has been seen by all, it becomes available for writes (i.e. reimplementing a queue).
    // - Only on resize will the indices need to be adjusted. This can be observed by readers with a 'version'?
    // c: Slice<Arc<UnsafeCell<[MaybeUninit<Raw>]>>>,
    last: Option<Chunk>,
    version: usize,
    head: usize,
    next: usize,
    seen: AtomicUsize,
    count: AtomicUsize,
}

#[derive(Default)]
struct Chunk {
    events: Vec<Raw>,
    keys: Vec<Key>,
}

impl Database {
    #[inline]
    pub fn events(&self) -> Events {
        Events(&self.events, self.tables())
    }
}

impl<'a> Events<'a> {
    pub fn listen<E: Event>(&self) -> Listen<'a, E> {
        Listen::new_with(self.clone())
    }

    #[inline]
    pub(crate) fn emit_create(&self, keys: &[Key], table: &Table) {
        self.emit(&self.0.destroy, keys, |index| Raw::Create {
            keys: Keys {
                index: index as _,
                count: keys.len() as _,
            },
            table: table.index(),
        })
    }

    #[inline]
    pub(crate) fn emit_destroy(&self, keys: &[Key], table: &Table) {
        self.emit(&self.0.destroy, keys, |index| Raw::Destroy {
            keys: Keys {
                index: index as _,
                count: keys.len() as _,
            },
            table: table.index(),
        })
    }

    #[inline]
    pub(crate) fn emit_modify(&self, keys: &[Key], tables: (&Table, &Table)) {
        self.emit(&self.0.modify, keys, |index| Raw::Modify {
            keys: Keys {
                index: index as _,
                count: keys.len() as _,
            },
            tables: Tables {
                source: tables.0.index(),
                target: tables.1.index(),
            },
        })
    }

    fn emit(&self, guard: &AtomicU64, keys: &[Key], event: impl FnOnce(usize) -> Raw) {
        let values = decompose(guard.load(Relaxed));
        if values.0 == 0 {
            return;
        }

        let mut pending = self.0.pending.lock();
        let index = pending.keys.len();
        pending.events.push(event(index));
        if values.1 == 0 {
            return;
        }
        pending.keys.extend_from_slice(keys);
    }
}

impl State {
    pub fn new() -> Self {
        Self {
            pending: Default::default(),
            ready: Default::default(),
            create: AtomicU64::new(0),
            destroy: AtomicU64::new(0),
            modify: AtomicU64::new(0),
        }
    }

    fn add_declare(&self, values: [bool; 6]) {
        self.update_declare(values, |source, target| {
            target.fetch_add(source, Relaxed);
        });
    }

    fn remove_declare(&self, values: [bool; 6]) {
        self.update_declare(values, |source, target| {
            target.fetch_sub(source, Relaxed);
        });
    }

    fn update_declare(&self, values: [bool; 6], update: impl Fn(u64, &AtomicU64)) {
        fn next(listen: bool, keys: bool, target: &AtomicU64, update: impl Fn(u64, &AtomicU64)) {
            if listen {
                update(recompose(1, if keys { 1 } else { 0 }), target);
            }
        }
        next(values[0], values[1], &self.create, &update);
        next(values[2], values[3], &self.destroy, &update);
        next(values[4], values[5], &self.modify, &update);
    }
}

impl<T> Buffer<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, items: I) -> ControlFlow<(), usize> {
        let count = self.0.len();
        self.0.extend(items);
        if self.1.apply(&mut self.0) {
            Continue(self.0.len() - count)
        } else {
            Break(())
        }
    }
}

impl DeclareContext<'_> {
    pub fn any(&mut self, keys: bool) {
        self.create(keys);
        self.modify(keys);
        self.destroy(keys);
    }

    pub fn create(&mut self, keys: bool) {
        self.values[0] |= true;
        self.values[1] |= keys;
    }

    pub fn destroy(&mut self, keys: bool) {
        self.values[2] |= true;
        self.values[3] |= keys;
    }

    pub fn modify(&mut self, keys: bool) {
        self.values[4] |= true;
        self.values[5] |= keys;
    }
}

impl<'a> DeclareContext<'a> {
    #[inline]
    pub fn own(&mut self) -> DeclareContext {
        DeclareContext {
            values: self.values,
        }
    }
}

impl<'a> ProcessContext<'a> {
    #[inline]
    pub fn keys(&self, keys: Keys) -> &'a [Key] {
        unsafe {
            get_unchecked(
                self.keys,
                keys.index as usize..(keys.index + keys.count) as usize,
            )
        }
    }

    #[inline]
    pub fn table(&self, index: u32) -> &'a Table {
        unsafe { self.tables.get_unchecked(index as _) }
    }
}

impl<'a, E: Event> Listen<'a, E> {
    pub fn clear(&mut self) {
        let keep = self.keep(Keep::First(0));
        self.next();
        self.keep(keep);
    }

    pub fn keep(&mut self, keep: Keep) -> Keep {
        let keep = replace(&mut self.buffer.1, keep);
        self.buffer.1.apply(&mut self.buffer.0);
        keep
    }

    pub fn with<F: Event>(self) -> Listen<'a, F> {
        self.events.0.add_declare(Listen::<F>::declare());
        let ready = self.events.0.ready.read();
        ready.count.fetch_add(1, Relaxed);
        ready.seen.fetch_add(1, Relaxed);
        Listen {
            events: self.events.clone(),
            cursor: Cursor {
                head: self.cursor.head,
                version: self.cursor.version,
            },
            buffer: Buffer(VecDeque::new(), Keep::All),
        }
    }

    pub fn update(&mut self) -> bool {
        let ready = self.events.0.ready.read();
        debug_assert!(self.cursor.version + 1 >= ready.version);
        if self.cursor.version < ready.version {
            self.cursor.head -= ready.head;
            self.cursor.version = ready.version;
            ready.seen.fetch_add(1, Relaxed);
        }

        let count = self.process(&ready);
        if count > 0 {
            return true;
        }

        // TODO: There should be some kind of version check with `self.pending` that would allow to return early without taking a
        // write lock if there are no new events.
        drop(ready);

        let mut ready = self.events.0.ready.write();
        let chunk = {
            let chunk = ready.last.take().unwrap_or_default();
            // The time spent with the `pending` lock is minimized as much as possible since it may block other threads
            // that hold database locks. The trade-off here is that the `ready` write lock may be taken in vain when `pending`
            // has no new events.
            let mut pending = self.events.0.pending.lock();
            if pending.events.is_empty() {
                ready.last = Some(chunk);
                return false;
            }
            replace(&mut *pending, chunk)
        };

        ready.chunks.push_back(chunk);
        let drain = ready.next;
        let buffers = *ready.count.get_mut();
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
            self.cursor.head -= ready.head;
            self.cursor.version = ready.version;
        }

        self.process(&RwLockWriteGuard::downgrade(ready)) > 0
    }

    fn process(&mut self, ready: &Ready) -> usize {
        self.events.1.update();
        let head = replace(&mut self.cursor.head, ready.chunks.len());
        let mut sum = 0;
        for chunk in ready.chunks.range(head..) {
            let context = ProcessContext {
                keys: &chunk.keys,
                tables: &self.events.1,
            };
            match self.buffer.extend(E::process(&chunk.events, context)) {
                Continue(count) => sum += count,
                Break(_) => break,
            }
        }
        sum
    }

    fn new_with(events: Events<'a>) -> Self {
        events.0.add_declare(Self::declare());
        let ready = events.0.ready.read();
        ready.count.fetch_add(1, Relaxed);
        ready.seen.fetch_add(1, Relaxed);
        Listen {
            events,
            cursor: Cursor {
                head: ready.chunks.len(),
                version: ready.version,
            },
            buffer: Buffer(VecDeque::new(), Keep::All),
        }
    }

    fn declare() -> [bool; 6] {
        let mut values = [false; 6];
        E::declare(DeclareContext {
            values: &mut values,
        });
        values
    }
}

impl<E: Event> Iterator for Listen<'_, E> {
    type Item = E;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(event) = self.buffer.0.pop_front() {
            return Some(event);
        }
        self.update();
        self.buffer.0.pop_front()
    }
}

impl<E: Event> Drop for Listen<'_, E> {
    fn drop(&mut self) {
        self.events.0.remove_declare(Listen::<E>::declare());
        let ready = self.events.0.ready.read();
        ready.count.fetch_sub(1, Relaxed);
        if self.cursor.version == ready.version {
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

#[inline]
const fn decompose(value: u64) -> (u32, u32) {
    ((value >> 32) as u32, value as u32)
}

#[inline]
const fn recompose(listeners: u32, keys: u32) -> u64 {
    ((listeners as u64) << 32) | (keys as u64)
}

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
        impl<'a> Events<'a> {
            pub fn $n(&self) -> Listen<'a, $on> {
                self.listen()
            }
        }

        impl<'a> Listen<'a, $on> {
            pub fn with_key(self) -> Listen<'a, $on_key> {
                self.with()
            }
            pub fn with_type<D: Datum>(self) -> Listen<'a, $on_type<D>> {
                self.with()
            }
            pub fn with_types(self) -> Listen<'a, $on_types> {
                self.with()
            }
        }

        impl<'a> Listen<'a, $on_key> {
            pub fn with_type<D: Datum>(self) -> Listen<'a, $on_key_type<D>> {
                self.with()
            }
            pub fn with_types(self) -> Listen<'a, $on_key_types> {
                self.with()
            }
        }

        impl<'a, D: Datum> Listen<'a, $on_type<D>> {
            pub fn with_key(self) -> Listen<'a, $on_key_type<D>> {
                self.with()
            }
        }

        impl<'a> Listen<'a, $on_types> {
            pub fn with_key(self) -> Listen<'a, $on_key_types> {
                self.with()
            }
        }
    };
}

macro_rules! event {
    (
        $n:ident, $t:ty, $ts:ty, $d:ident, $table_f:ident, $table_t:ty,
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
            pub $table_f: $table_t,
        }
        #[derive(Clone, Debug)]
        pub struct $on_key {
            pub key: Key,
            pub types: $ts,
            pub $table_f: $table_t,
        }
        #[derive(Clone, Debug)]
        pub struct $on_type<T> {
            pub keys: usize,
            pub $table_f: $table_t,
            _marker: PhantomData<T>,
        }
        #[derive(Clone, Debug)]
        pub struct $on_types {
            pub keys: usize,
            pub r#type: $t,
            pub $table_f: $table_t,
        }
        #[derive(Clone, Debug)]
        pub struct $on_key_type<T> {
            pub key: Key,
            pub $table_f: $table_t,
            _marker: PhantomData<T>,
        }
        #[derive(Clone, Debug)]
        pub struct $on_key_types {
            pub key: Key,
            pub r#type: $t,
            pub $table_f: $table_t,
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

        impl From<$on_key> for Key {
            #[inline]
            fn from(value: $on_key) -> Key {
                value.key
            }
        }

        impl<T> From<$on_key_type<T>> for Key {
            #[inline]
            fn from(value: $on_key_type<T>) -> Key {
                value.key
            }
        }

        impl From<$on_key_types> for Key {
            #[inline]
            fn from(value: $on_key_types) -> Key {
                value.key
            }
        }

        impl Event for $on {
            type Events<'a> = impl Iterator<Item = Self> + 'a;

            fn declare(mut context: DeclareContext) {
                context.$d(false)
            }

            fn process<'a>(events: &'a [Raw], context: ProcessContext<'a>) -> Self::Events<'a> {
                events.iter().filter_map(move |event| {
                    let &Raw::$raw { keys, $table_f } = event else {
                        return None;
                    };
                    let types = $count(&context, $table_f);
                    Some(Self {
                        keys: keys.count as _,
                        types,
                        $table_f,
                    })
                })
            }
        }

        impl Event for $on_key {
            type Events<'a> = impl Iterator<Item = Self> + 'a;

            fn declare(mut context: DeclareContext) {
                context.$d(true)
            }

            fn process<'a>(events: &'a [Raw], context: ProcessContext<'a>) -> Self::Events<'a> {
                events
                    .iter()
                    .filter_map(move |event| {
                        let &Raw::$raw { keys, $table_f } = event else {
                            return None;
                        };
                        let keys = context.keys(keys);
                        let types = $count(&context, $table_f);
                        Some(keys.iter().map(move |&key| Self {
                            key,
                            types,
                            $table_f,
                        }))
                    })
                    .flatten()
            }
        }

        impl Event for $on_types {
            type Events<'a> = impl Iterator<Item = Self> + 'a;

            fn declare(mut context: DeclareContext) {
                context.$d(false)
            }

            fn process<'a>(events: &'a [Raw], context: ProcessContext<'a>) -> Self::Events<'a> {
                events
                    .iter()
                    .filter_map(move |event| {
                        let &Raw::$raw { keys, $table_f } = event else {
                            return None;
                        };
                        Some($types(&context, $table_f).map(move |r#type| Self {
                            keys: keys.count as _,
                            r#type,
                            $table_f,
                        }))
                    })
                    .flatten()
            }
        }

        impl Event for $on_key_types {
            type Events<'a> = impl Iterator<Item = Self> + 'a;

            fn declare(mut context: DeclareContext) {
                context.$d(true)
            }

            fn process<'a>(events: &'a [Raw], context: ProcessContext<'a>) -> Self::Events<'a> {
                events
                    .iter()
                    .filter_map(move |event| {
                        let &Raw::$raw { keys, $table_f } = event else {
                            return None;
                        };
                        let keys = context.keys(keys);
                        Some($types(&context, $table_f).flat_map(move |r#type| {
                            keys.iter().map(move |&key| Self {
                                key,
                                r#type,
                                $table_f,
                            })
                        }))
                    })
                    .flatten()
            }
        }

        impl<D: Datum> Event for $on_type<D> {
            type Events<'a> = impl Iterator<Item = Self> + 'a;

            fn declare(mut context: DeclareContext) {
                context.$d(false)
            }

            fn process<'a>(events: &'a [Raw], context: ProcessContext<'a>) -> Self::Events<'a> {
                events.iter().filter_map(move |event| {
                    let &Raw::$raw { keys, $table_f } = event else {
                        return None;
                    };
                    if $has::<D>(&context, $table_f) {
                        Some(Self {
                            keys: keys.count as _,
                            $table_f,
                            _marker: PhantomData,
                        })
                    } else {
                        None
                    }
                })
            }
        }

        impl<D: Datum> Event for $on_key_type<D> {
            type Events<'a> = impl Iterator<Item = Self> + 'a;

            fn declare(mut context: DeclareContext) {
                context.$d(true)
            }

            fn process<'a>(events: &'a [Raw], context: ProcessContext<'a>) -> Self::Events<'a> {
                events
                    .iter()
                    .filter_map(move |event| {
                        let &Raw::$raw { keys, $table_f } = event else {
                            return None;
                        };
                        let keys = context.keys(keys);
                        if $has::<D>(&context, $table_f) {
                            Some(keys.iter().map(move |&key| Self {
                                key,
                                $table_f,
                                _marker: PhantomData,
                            }))
                        } else {
                            None
                        }
                    })
                    .flatten()
            }
        }
    };
}

#[inline]
fn create_destroy_count(context: &ProcessContext, table: u32) -> usize {
    context.table(table).columns().len()
}
#[inline]
fn create_destroy_types<'a>(
    context: &ProcessContext<'a>,
    table: u32,
) -> impl Iterator<Item = TypeId> + 'a {
    context.table(table).types()
}
#[inline]
fn create_destroy_has<D: Datum>(context: &ProcessContext, table: u32) -> bool {
    context.table(table).has::<D>()
}
event!(
    on_create,
    TypeId,
    usize,
    create,
    table,
    u32,
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
    u32,
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
fn modify_count(context: &ProcessContext, tables: Tables) -> Types {
    let (add, remove) = modify_types(context, tables).fold((0, 0), |counts, r#type| match r#type {
        Type::Add(_) => (counts.0 + 1, counts.1),
        Type::Remove(_) => (counts.0, counts.1 + 1),
    });
    Types { add, remove }
}
#[inline]
fn modify_types<'a>(
    context: &ProcessContext<'a>,
    tables: Tables,
) -> impl Iterator<Item = Type> + 'a {
    let source = context.table(tables.source);
    let target = context.table(tables.target);
    sorted_symmetric_difference_by(
        |left, right| Ord::cmp(&left.identifier(), &right.identifier()),
        source.types().map(Type::Remove),
        target.types().map(Type::Add),
    )
}
#[inline]
fn modify_has<D: Datum>(context: &ProcessContext, tables: Tables) -> bool {
    let source = context.table(tables.source);
    let target = context.table(tables.target);
    source.has::<D>() != target.has::<D>()
}
event!(
    on_modify,
    Type,
    Types,
    modify,
    tables,
    Tables,
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
fn add_count(context: &ProcessContext, tables: Tables) -> usize {
    add_types(context, tables).count()
}
#[inline]
fn add_types<'a>(
    context: &ProcessContext<'a>,
    tables: Tables,
) -> impl Iterator<Item = TypeId> + 'a {
    let source = context.table(tables.source);
    let target = context.table(tables.target);
    sorted_difference(target.types(), source.types())
}
#[inline]
fn add_has<D: Datum>(context: &ProcessContext, tables: Tables) -> bool {
    let source = context.table(tables.source);
    let target = context.table(tables.target);
    !source.has::<D>() && target.has::<D>()
}
event!(
    on_add,
    TypeId,
    usize,
    modify,
    tables,
    Tables,
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
fn remove_count(context: &ProcessContext, tables: Tables) -> usize {
    remove_types(context, tables).count()
}
#[inline]
fn remove_types<'a>(
    context: &ProcessContext<'a>,
    tables: Tables,
) -> impl Iterator<Item = TypeId> + 'a {
    let source = context.table(tables.source);
    let target = context.table(tables.target);
    sorted_difference(source.types(), target.types())
}
#[inline]
fn remove_has<D: Datum>(context: &ProcessContext, tables: Tables) -> bool {
    let source = context.table(tables.source);
    let target = context.table(tables.target);
    source.has::<D>() && !target.has::<D>()
}
event!(
    on_remove,
    TypeId,
    usize,
    modify,
    tables,
    Tables,
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

pub enum OnAny {
    Create(OnCreate),
    Modify(OnModify),
    Destroy(OnDestroy),
}
pub enum OnAnyKey {
    Create(OnCreateKey),
    Modify(OnModifyKey),
    Destroy(OnDestroyKey),
}
pub enum OnAnyType<T> {
    Create(OnCreateType<T>),
    Modify(OnModifyType<T>),
    Destroy(OnDestroyType<T>),
}
pub enum OnAnyTypes {
    Create(OnCreateTypes),
    Modify(OnModifyTypes),
    Destroy(OnDestroyTypes),
}
pub enum OnAnyKeyType<T> {
    Create(OnCreateKeyType<T>),
    Modify(OnModifyKeyType<T>),
    Destroy(OnDestroyKeyType<T>),
}
pub enum OnAnyKeyTypes {
    Create(OnCreateKeyTypes),
    Modify(OnModifyKeyTypes),
    Destroy(OnDestroyKeyTypes),
}

with!(
    on_any,
    OnAny,
    OnAnyKey,
    OnAnyType,
    OnAnyTypes,
    OnAnyKeyType,
    OnAnyKeyTypes
);

#[inline]
fn map<'a, T: Event, U>(
    map: impl FnMut(T) -> U + 'a,
    event: &'a Raw,
    context: ProcessContext<'a>,
) -> impl Iterator<Item = U> + 'a {
    T::process(from_ref(event), context).map(map)
}

enum One<T1, T2, T3> {
    A(T1),
    B(T2),
    C(T3),
}

impl<I1: Iterator, I2: Iterator<Item = I1::Item>, I3: Iterator<Item = I1::Item>> Iterator
    for One<I1, I2, I3>
{
    type Item = I1::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            One::A(iterator) => iterator.next(),
            One::B(iterator) => iterator.next(),
            One::C(iterator) => iterator.next(),
        }
    }
}

macro_rules! body {
    ($keys:expr) => {
        type Events<'a> = impl Iterator<Item = Self> + 'a;

        fn declare(mut context: DeclareContext) {
            context.any($keys)
        }

        fn process<'a>(events: &'a [Raw], context: ProcessContext<'a>) -> Self::Events<'a> {
            events.iter().flat_map(move |event| match event {
                Raw::Create { .. } => One::A(map(Self::Create, event, context.clone())),
                Raw::Destroy { .. } => One::B(map(Self::Destroy, event, context.clone())),
                Raw::Modify { .. } => One::C(map(Self::Modify, event, context.clone())),
            })
        }
    };
}

macro_rules! from {
    ($t:ty, $i:ident) => {
        #[inline]
        fn from(value: $t) -> Key {
            match value {
                $i::Create(event) => event.into(),
                $i::Modify(event) => event.into(),
                $i::Destroy(event) => event.into(),
            }
        }
    };
}

impl Event for OnAny {
    body!(false);
}

impl Event for OnAnyKey {
    body!(true);
}

impl<T: Datum> Event for OnAnyType<T> {
    body!(false);
}

impl Event for OnAnyTypes {
    body!(false);
}

impl<T: Datum> Event for OnAnyKeyType<T> {
    body!(true);
}

impl Event for OnAnyKeyTypes {
    body!(true);
}

impl From<OnAnyKey> for Key {
    from!(OnAnyKey, OnAnyKey);
}

impl<T> From<OnAnyKeyType<T>> for Key {
    from!(OnAnyKeyType<T>, OnAnyKeyType);
}

impl From<OnAnyKeyTypes> for Key {
    from!(OnAnyKeyTypes, OnAnyKeyTypes);
}
