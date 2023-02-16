use crate::{
    core::utility::get_unchecked,
    key::Key,
    table::{self, Table},
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
    fn process<C: Collect<Self>>(collect: &mut C, context: ProcessContext) -> ControlFlow<()>;
}

pub trait Collect<T> {
    fn one(&mut self, item: T) -> ControlFlow<()>;
    fn all<I: IntoIterator<Item = T>>(&mut self, items: I) -> ControlFlow<()>;
    fn next(&mut self) -> Option<T>;
}

pub struct Events {
    ready: RwLock<Ready>,
    pending: Mutex<Chunk>,
    create: AtomicU64,
    destroy: AtomicU64,
    modify: AtomicU64,
}

pub struct Listen<'d, E: Event> {
    guard: Guard<'d, E>,
    tables: table::Tables<'d>,
    head: usize,
    buffer: Buffer<E>,
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

pub struct ProcessContext<'a> {
    events: &'a [Raw],
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

struct Guard<'a, E: Event>(&'a Events, usize, PhantomData<E>);
struct Buffer<T>(VecDeque<T>, Keep);

#[derive(Default)]
struct Ready {
    chunks: VecDeque<Chunk>,
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
    pub const fn events(&self) -> &Events {
        &self.events
    }

    pub fn listen<E: Event>(&self) -> Listen<E> {
        Listen::new(self)
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

    pub(crate) fn emit_create(&self, keys: &[Key], table: &Table) {
        let values = decompose(self.create.load(Relaxed));
        if values.0 == 0 {
            return;
        }

        let mut pending = self.pending.lock();
        let index = pending.keys.len();
        pending.events.push(Raw::Create {
            keys: Keys {
                index: index as _,
                count: keys.len() as _,
            },
            table: table.index(),
        });
        if values.1 == 0 {
            return;
        }
        pending.keys.extend_from_slice(keys);
    }

    pub(crate) fn emit_destroy(&self, keys: &[Key], table: &Table) {
        let values = decompose(self.destroy.load(Relaxed));
        if values.0 == 0 {
            return;
        }

        let mut pending = self.pending.lock();
        let index = pending.keys.len();
        pending.events.push(Raw::Destroy {
            keys: Keys {
                index: index as _,
                count: keys.len() as _,
            },
            table: table.index(),
        });
        if values.1 == 0 {
            return;
        }
        pending.keys.extend_from_slice(keys);
    }

    pub(crate) fn emit_modify(&self, keys: &[Key], tables: (&Table, &Table)) {
        let values = decompose(self.modify.load(Relaxed));
        if values.0 == 0 {
            return;
        }

        let mut pending = self.pending.lock();
        let index = pending.keys.len();
        pending.events.push(Raw::Modify {
            keys: Keys {
                index: index as _,
                count: keys.len() as _,
            },
            tables: Tables {
                source: tables.0.index(),
                target: tables.1.index(),
            },
        });
        if values.1 == 0 {
            return;
        }
        pending.keys.extend_from_slice(keys);
    }

    fn update<E: Event, C: Collect<E>>(
        &self,
        collect: &mut C,
        head: &mut usize,
        version: &mut usize,
        tables: &mut table::Tables,
    ) -> Option<E> {
        if let Some(event) = collect.next() {
            return Some(event);
        }

        let ready = self.ready.read();
        debug_assert!(*version + 1 >= ready.version);
        if *version < ready.version {
            *head -= ready.head;
            *version = ready.version;
            ready.seen.fetch_add(1, Relaxed);
        }
        self.process(collect, &ready, head, tables);
        if let Some(event) = collect.next() {
            return Some(event);
        }

        // TODO: There should be some kind of version check with `self.pending` that would allow to return early without taking a
        // write lock if there are no new events.
        drop(ready);

        let mut ready = self.ready.write();
        let chunk = {
            let chunk = ready.last.take().unwrap_or_default();
            // The time spent with the `pending` lock is minimized as much as possible since it may block other threads
            // that hold database locks. The trade-off here is that the `ready` write lock may be taken in vain when `pending`
            // has no new events.
            let mut pending = self.pending.lock();
            if pending.events.len() == 0 {
                ready.last = Some(chunk);
                return None;
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
            *head -= ready.head;
            *version = ready.version;
        }

        self.process(collect, &RwLockWriteGuard::downgrade(ready), head, tables);
        collect.next()
    }

    fn process<E: Event, C: Collect<E>>(
        &self,
        collect: &mut C,
        ready: &Ready,
        head: &mut usize,
        tables: &mut table::Tables,
    ) {
        tables.update();
        let head = replace(head, ready.chunks.len());
        for chunk in ready.chunks.range(head..) {
            let context = ProcessContext {
                events: &chunk.events,
                keys: &chunk.keys,
                tables: &tables,
            };
            if let Break(_) = E::process(collect, context) {
                break;
            }
        }
    }
}

impl<T> Collect<T> for Buffer<T> {
    fn one(&mut self, item: T) -> ControlFlow<()> {
        self.0.push_back(item);
        self.1.apply(&mut self.0)
    }

    fn all<I: IntoIterator<Item = T>>(&mut self, items: I) -> ControlFlow<()> {
        self.0.extend(items);
        self.1.apply(&mut self.0)
    }

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.0.pop_front()
    }
}

impl DeclareContext<'_> {
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

impl<'a> ProcessContext<'a> {
    #[inline]
    pub const fn events(&self) -> &'a [Raw] {
        self.events
    }

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
        // Dropping the guard only to create a new one right after would be useless ceremony. `ManuallyDrop` helps skip some of it.
        let guard = ManuallyDrop::new(self.guard);
        Self::remove_declare(guard.0);
        Listen::<F>::add_declare(guard.0);
        Listen {
            guard: Guard(guard.0, guard.1, PhantomData),
            tables: self.tables,
            head: self.head,
            buffer: Buffer(VecDeque::new(), self.buffer.1),
        }
    }

    fn new(database: &'a Database) -> Self {
        Self::add_declare(database.events());
        let ready = database.events().ready.read();
        ready.count.fetch_add(1, Relaxed);
        ready.seen.fetch_add(1, Relaxed);
        Listen {
            guard: Guard(database.events(), ready.version, PhantomData),
            tables: database.tables(),
            head: ready.chunks.len(),
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

    fn update(events: &Events, values: [bool; 6], update: impl Fn(u64, &AtomicU64)) {
        let mut index = 0;
        for channel in [&events.create, &events.destroy, &events.modify] {
            if values[index] {
                update(recompose(1, if values[index + 1] { 1 } else { 0 }), channel);
            }
            index += 2;
        }
    }
}

impl<E: Event> Iterator for Listen<'_, E> {
    type Item = E;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.guard.0.update(
            &mut self.buffer,
            &mut self.head,
            &mut self.guard.1,
            &mut self.tables,
        )
    }
}

impl<E: Event> Drop for Guard<'_, E> {
    fn drop(&mut self) {
        Listen::<E>::remove_declare(self.0);
        let ready = self.0.ready.read();
        ready.count.fetch_sub(1, Relaxed);
        if self.1 == ready.version {
            ready.seen.fetch_sub(1, Relaxed);
        }
    }
}

impl Keep {
    fn apply<T>(&self, items: &mut VecDeque<T>) -> ControlFlow<()> {
        match *self {
            Keep::All => Continue(()),
            Keep::First(0) | Keep::Last(0) => {
                items.clear();
                Break(())
            }
            Keep::First(count) if items.len() < count => Continue(()),
            Keep::First(count) => {
                items.truncate(count);
                Break(())
            }
            Keep::Last(count) if items.len() < count => Continue(()),
            Keep::Last(count) => {
                items.drain(..items.len() - count);
                Continue(())
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

pub mod events {
    use super::*;
    use crate::core::utility::{sorted_difference, sorted_symmetric_difference_by};

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
                pub fn $n(&self) -> Listen<$on> {
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

            impl Event for $on {
                fn declare(mut context: DeclareContext) {
                    context.$d(false)
                }

                #[inline]
                fn process<C: Collect<Self>>(
                    collect: &mut C,
                    context: ProcessContext,
                ) -> ControlFlow<()> {
                    for event in context.events() {
                        let &Raw::$raw { keys, $table_f } = event else { continue; };
                        let types = $count(&context, $table_f);
                        collect.one(Self {
                            keys: keys.count as _,
                            types: types,
                            $table_f,
                        })?;
                    }
                    Continue(())
                }
            }

            impl Event for $on_key {
                fn declare(mut context: DeclareContext) {
                    context.$d(true)
                }

                #[inline]
                fn process<C: Collect<Self>>(
                    collect: &mut C,
                    context: ProcessContext,
                ) -> ControlFlow<()> {
                    for event in context.events() {
                        let &Raw::$raw { keys, $table_f } = event else { continue; };
                        let keys = context.keys(keys);
                        let types = $count(&context, $table_f);
                        collect.all(keys.iter().map(|&key| Self {
                            key,
                            types,
                            $table_f,
                        }))?;
                    }
                    Continue(())
                }
            }

            impl Event for $on_types {
                fn declare(mut context: DeclareContext) {
                    context.$d(false)
                }

                #[inline]
                fn process<C: Collect<Self>>(
                    collect: &mut C,
                    context: ProcessContext,
                ) -> ControlFlow<()> {
                    for event in context.events() {
                        let &Raw::$raw { keys, $table_f } = event else { continue; };
                        collect.all($types(&context, $table_f).map(|r#type| Self {
                            keys: keys.count as _,
                            r#type,
                            $table_f,
                        }))?;
                    }
                    Continue(())
                }
            }

            impl Event for $on_key_types {
                fn declare(mut context: DeclareContext) {
                    context.$d(true)
                }

                #[inline]
                fn process<C: Collect<Self>>(
                    collect: &mut C,
                    context: ProcessContext,
                ) -> ControlFlow<()> {
                    for event in context.events() {
                        let &Raw::$raw { keys, $table_f } = event else { continue; };
                        let keys = context.keys(keys);
                        for r#type in $types(&context, $table_f) {
                            collect.all(keys.iter().map(move |&key| Self {
                                key,
                                r#type,
                                $table_f,
                            }))?;
                        }
                    }
                    Continue(())
                }
            }

            impl<D: Datum> Event for $on_type<D> {
                fn declare(mut context: DeclareContext) {
                    context.$d(false)
                }

                #[inline]
                fn process<C: Collect<Self>>(
                    collect: &mut C,
                    context: ProcessContext,
                ) -> ControlFlow<()> {
                    for event in context.events() {
                        let &Raw::$raw { keys, $table_f } = event else { continue; };
                        if $has::<D>(&context, $table_f) {
                            collect.one(Self {
                                keys: keys.count as _,
                                $table_f,
                                _marker: PhantomData,
                            })?;
                        }
                    }
                    Continue(())
                }
            }

            impl<D: Datum> Event for $on_key_type<D> {
                fn declare(mut context: DeclareContext) {
                    context.$d(true)
                }

                #[inline]
                fn process<C: Collect<Self>>(
                    collect: &mut C,
                    context: ProcessContext,
                ) -> ControlFlow<()> {
                    for event in context.events() {
                        let &Raw::$raw { keys, $table_f } = event else { continue; };
                        let keys = context.keys(keys);
                        if $has::<D>(&context, $table_f) {
                            collect.all(keys.iter().map(|&key| Self {
                                key,
                                $table_f,
                                _marker: PhantomData,
                            }))?;
                        }
                    }
                    Continue(())
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
        let (add, remove) =
            modify_types(context, tables).fold((0, 0), |counts, r#type| match r#type {
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
}
