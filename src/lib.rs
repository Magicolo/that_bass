#![feature(alloc_layout_extra)]
#![feature(once_cell)]

pub mod core;
pub mod create;
mod defer;
pub mod destroy;
pub mod event;
pub mod filter;
pub mod key;
pub mod modify;
pub mod query;
pub mod resources;
pub mod row;
pub mod table;
pub mod template;
mod trace;

#[cfg(test)]
mod test;

use event::Events;
use key::Key;
use resources::Resources;
use std::{
    alloc::{alloc, dealloc, Layout},
    any::{type_name, TypeId},
    borrow::Cow,
    error, fmt,
    mem::{needs_drop, size_of, ManuallyDrop},
    num::NonZeroUsize,
    ptr::{copy, drop_in_place, slice_from_raw_parts_mut, NonNull},
    sync::LazyLock,
};
pub use that_base_derive::{Datum, Filter, Template};

/*
    TODO: Implement an interpreter.
        - This would allow sending dynamic requests to the database in the form of `String`s and listen response also a `String`s.
        - With this feature, the database becomes usable from any ecosystem and can serve similar purposes as other in-memory
        databases such as Redis.
        - Would be nice if the query language could be expressed in json understandably and efficiently; otherwise, a custom query
        language may be considered.
        - Queries may look something like:
            { "Position": { "x": null, "y": 1 } }:
            - Queries the first row that has a `Position` column with at least the field `x`.
            - If the field `y` is present, it is filled, otherwise the provided value is used.
            - If `Position` has more fields, they are projected out.
            - The response will yield an object with the same structure.
            - To query all rows, wrap this query in an array `[{ ... }]`.
    TODO: Implement a `Trace` that keeps a history for every key.
        - struct Trace; impl Listen for Trace {}
    TODO: Remove empty tables when calling `Tables::shrink`.
        - Requires to stop using `Table::index`.
    TODO: Queries don't prevent visiting a key twice if another thread resolves a move operation from a visited table to an unvisited
    one while the query is iterating.
        - It may be resonnable to require from users (or a scheduler) to resolve deferred operations at an appropriate time.
    TODO: Implement a table `If` operation.
        - The if checks some condition on tables with an upgradable lock and executes its body with a write lock on success.
        - This ensures that no structural operation has occured on the table between the check and the body.
        - The motivating scenario is `FindOrCreate` which tries to match a given row and creates the row if it is missing.
            - Usage could look like (arguments to `If` should be inferred): `|mut a: If| {
                // fn find_or_create<F, T: Template>(&mut self, find: impl FnMut(T::Read) -> Option<F>, create: impl FnMut() -> T) -> Result<Option<F>, Error>;
                a.find_or_create(|p| p.x == 0, || Position(0, 0, 0));
            }`
    TODO: Implement `Defer`:
        - Will order the resolution of deferred operations such that coherence is maintained.
    TODO: Implement `Keep<T: Template>/Trim<T: Template>`:
        - Removes all datum from a key except to ones provided by `T`.
    TODO: Implement `Change<T: Template>/Convert<T: Template>/Conform<T: Template>`:
        - Converts a key such that it corresponds to the `T: Template` (adds missing datum, removes exceeding datum).
    TODO: Implement `Template for Set<D: Datum>`:
        - Sets the datum `D` for a key only if the key already has the datum.
    TODO: Implement `Row` for `Get<D: Datum + Copy>`:
        - At the beginning of iteration, `Get` makes a copy of the whole column to a temporary buffer, then the column lock can be
        released immediately.
    TODO: The `Table::commit` mechanism allows for an incoherent state where a create operation has been resolved, but the
    created keys are reported to not be present in a corresponding query.
        - This happens because `Table::commit` can technically fail for an unlimited amount of time...
        - Would require to force a successful commit at key moments.
    TODO: Share some query state using a COW pattern.
        - Investigate `arc-swap`.
        - Gains are likely marginal and memory will remain in use for the lifetime of the database...
    TODO: Implement `Permute`.
        - Returns all permutations (with repetitions) of two queries.
        - Order of items matters, so (A, B) is considered different than (B, A), thus both will be returned.
        - May be unified with `Combine`.
    TODO: Implement `Combine`.
        - Returns all combinations (no repetitions) of two queries.
        - Order of items does not matter, so (A, B) is the same as (B, A), thus only one of those will be returned.
        - Use a filter similar to `A.key() < B.key()` to eliminate duplicates.
    TODO: Implement compile-time checking of `Columns`, if possible.
    TODO: Prevent using a query within a query using an auto trait `Nest` that is negatively implemented for `Query`.
    TODO: Test the database with generative tests.
    TODO: Make `Table::columns` inline rather than with an indirection.
        - See `thin-dst` and/or `slice-dst` libraries.

    TODO (POSTPONED): Allow querying with a struct or enum with a `#[derive(Row)]`.
    TODO (POSTPONED): Add an option to split large table stores in chunks of fixed size.
        - Large stores have better locality but might cause more contention on their locks.
        - Small stores cause less contention but have worse locality.
        - Imposes a design dillema on the user:
            - Use more static templates with dynamic values (larger stores).
            - Use more dynamic templates (using `Add/Remove` operations) with static datum (smaller stores).
        - Tables may have a `size: u32` that defaults to `u32::MAX` that defines the maximum store size.
        - After the first chunk has grown to full size (ex: 1024), change growing strategy to allocating
        full chunks with separate locks.
        - This removes the dillema from the user in favor of more static templates since they'll have similar
        locality and parallelism properties compared to more dynamic templates, but in the dynamic case, there'll be more
        move operations.
        - Should offer some benefits in terms in parallelism (parallelise more granularly by chunk).
        - Should offer some opportunities when moving/destroying rows such that only one chunk will need to be locked.
            - Each chunk may have its own `count` such that `squash` can be used within a chunk.
        - This solution will add a lot of complexity.
        - Will this require table + chunk + column locks vs table + column locks (currently)?
        *** It seems that locality should be prioritise since it allows work on columns do be completed sooner which reduces
        contention. Locality also allows a more efficient use of SIMD instructions.
    TODO (POSTPONED): Implement `Row` for `Add<T: Template>`?
        - The added constraints on `Query`, the added locks, the added complexity and the added confusion in the API tend
        to outweigh the potential benefits of being able to make more assumptions about queued keys...

        - Has only one method `Add::add(self, T)` (consuming `self` ensures that there are no duplicate keys).
        - Since a table lock will be held while iterating the query up to the resolving of `Add` (inclusively), keys can be assumed
        to be valid and to not have been moved.
        - Keys can also be assumed to be ordered in the table!
        - `Add` would be resolved right after a table is done being iterated.
        - The query will need to take upgradable locks on its tables rather than read locks.
        - Upgradable locks will also need to be taken for source tables that have a target table with a lower index (such that table
        locks can be taken in index order).
        - To prevent visiting keys twice, some ordering must apply on the query tables:
            - Group tables by overlap, order them from most to least columns, groups can be reordered freely within themselves.
                - The overlap is the size of the intersection of all of the combined `Add<T>` (or similar operator) metas.
            - With `Add<A>`, all tables with an `A` must be visited first (in any order within themselves).
            - With `Add<A>, Add<B>`: `A & B`, `A | B`, others.
            - With `Add<A>, Add<B>, Add<C>`: `A & B & C`, (`A & B`) | (`A & C`) | (`B & C`), A | B | C, others.
        - Because of this ordering, `Add<T>` will not be allowed with `Remove<U>` in the same query.
    TODO (POSTPONED): Implement `Row` for `Remove<T: Template>`:
        - Same benefits and inconvenients as `Add<T>`.
        - Requires the reverse ordering as `Add<T>`.
        - Do not allow `Add`-like and `Remove`-like operators in the same query.
    TODO (POSTPONED): Implement nest operations for query:
        - Forces the outer query to take additionnal locks in case the inner query needs to produce the same item as the outer query:
            - An outer read that conflicts with an inner write becomes an upgradable read.
            - An inner read that is missing in outer becomes a read if its index is lower than the last outer column.
            - An inner write that is missing in outer becomes an upgradable read if its index is lower than the last outer column.
        - Always skip the current outer key in the inner query.
        - When inner takes column locks from a table:
            - GREATER than outer => hard lock the columns directly as usual
            - EQUAL to outer => add/upgrade the missing locks
            - LESS than outer => first try to lock the columns and on failure, drop the columns
                locks (while keeping the table locks) from the outer query and hard lock all the column locks in order.
            - FIX: this strategy may allow an outer immutable reference to be modified elsewhere while holding on to it...
        database
            .query::<(&mut A, &CopyFrom)>()
            .nest::<&A>()
            .each(|((a1, copy), nest)| nest.find(copy.0, |a2| a1.0 = a2.0));
        database
            .query::<&A>()
            .nest::<&mut A>()
            .each(|(a1, nest)| nest.each(|a2| a2.0 = a1.0));
*/

/*
Scheduler library:
    - Can be completely independent from this library.
    - A `Run<I, O>` holds an `fn(I) -> O` where `I: Depend`.
    - With depend, the scheduler will be able to parallelize efficiently and maintain a chosen level of coherence.
    - A scheduler will need to decide when to resolve deferred operations such as `Create/Destroy/Add/Remove`.
    - A scheduler would be allowed to reorder chunks of work to minimize contention as long as coherence is maintained.
    - Might need to prevent combining operations that require a dynamic ordering to define a proper outcome:
        - `Create` operations are always resolved first (compatible with all other operations).
        - `Destroy` operations are always resolved second (compatible with all other operations).
        - `Add<T>` and `Remove<U>` operations must be dynamically ordered if `T` overlaps `U` (otherwise compatible).
        - `Add` and `Add` don't require a resolve order.
        - `Remove` and `Remove` don't require a resolve order.

    let database = Database::new();
    // Scheduler::new(): Scheduler<()>
    // Scheduler::with(&database): Scheduler<&Database>
    database
        .scheduler()
        .add(a_system) // impl FnOnce(T) -> Run<impl Depend>
        .schedule()?;

    fn a_system(database: &Database) -> Run<impl Depend> {
        Run::new(
            (database.create(), database.query()), // impl Depend; declares dependencies to the scheduler
            |(create, query)| {
                let key = create.one(());
                query.find(key, |item| {});
            }) // impl FnMut(&mut S) -> O + 'static
    }
*/

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    DuplicateMeta,
    InvalidKey(Key),
    InvalidType(TypeId),
    KeyNotInQuery(Key),
    KeyNotInSplit(Key),
    MissingColumn(TypeId),
    MissingTable(usize),
    ReadWriteConflict(TypeId),
    WriteWriteConflict(TypeId),
    TablesMustDiffer(u32),
    TableDoesNotMatchFilter(u32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self, f)
    }
}
impl error::Error for Error {}

pub struct Database {
    keys: key::State,
    tables: table::State,
    resources: Resources,
    filters: filter::State,
    events: Events,
}

pub struct Meta {
    identifier: fn() -> TypeId,
    name: fn() -> &'static str,
    layout: fn() -> Layout,
    new: unsafe fn(usize) -> NonNull<()>,
    free: unsafe fn(NonNull<()>, usize, usize),
    copy: unsafe fn((NonNull<()>, usize), (NonNull<()>, usize), NonZeroUsize),
    drop: (fn() -> bool, unsafe fn(NonNull<()>, usize, NonZeroUsize)),
}

pub trait Datum: Sized + 'static {
    #[inline]
    fn meta() -> &'static Meta {
        &Meta {
            identifier: TypeId::of::<Self>,
            name: type_name::<Self>,
            layout: Layout::new::<Self>,
            new: |capacity| unsafe {
                let mut items = ManuallyDrop::new(Vec::<Self>::with_capacity(capacity));
                debug_assert!(items.capacity() == capacity || items.capacity() == usize::MAX);
                NonNull::new_unchecked(items.as_mut_ptr()).cast::<()>()
            },
            free: |data, count, capacity| unsafe {
                Vec::from_raw_parts(data.as_ptr().cast::<Self>(), count, capacity);
            },
            copy: |source, target, count| unsafe {
                if size_of::<Self>() > 0 {
                    let source = source.0.as_ptr().cast::<Self>().add(source.1);
                    let target = target.0.as_ptr().cast::<Self>().add(target.1);
                    copy(source, target, count.get());
                }
            },
            drop: (needs_drop::<Self>, |data, index, count| unsafe {
                if needs_drop::<Self>() {
                    let data = data.as_ptr().cast::<Self>().add(index);
                    drop_in_place(slice_from_raw_parts_mut(data, count.get()));
                }
            }),
        }
    }
}

impl Meta {
    #[inline]
    pub fn identifier(&self) -> TypeId {
        (self.identifier)()
    }

    #[inline]
    pub fn name(&self) -> &'static str {
        (self.name)()
    }

    #[inline]
    pub fn layout(&self) -> Layout {
        (self.layout)()
    }
}

impl Database {
    pub fn new() -> Self {
        Database {
            keys: key::State::new(),
            tables: table::State::new(),
            resources: Resources::new(),
            filters: filter::State::new(),
            events: Events::new(),
        }
    }
}

mod dynamic {
    use super::*;
    use parking_lot::RwLock;
    use std::{collections::BTreeMap, string};

    impl Database {
        pub fn run(
            &self,
            operations: impl IntoIterator<Item = &Operation>,
        ) -> Result<Value, Error> {
            todo!()
        }
    }

    pub enum Node {
        Null,
        Number(Number),
        String(String),
        Increment(Number),
        Pair(Key, Box<Node>),
        List(Vec<Node>),
    }

    pub enum Operation {
        Create(Node),
        Destroy(Key),
        Modify(Key, Node),
    }

    pub enum String {
        Borrow(&'static str),
        Own(string::String),
    }

    pub enum Number {
        Integer(i64),
        Rational(f64),
    }

    pub enum Key {
        Meta(&'static Meta),
        Name(String),
    }

    impl Into<Node> for &'static str {
        fn into(self) -> Node {
            Node::String(String::from(self))
        }
    }

    impl Into<Box<Node>> for &'static str {
        fn into(self) -> Box<Node> {
            Box::new(Node::from(self))
        }
    }

    impl Into<String> for &'static str {
        fn into(self) -> String {
            String::Borrow(self)
        }
    }

    impl Into<Key> for &'static str {
        fn into(self) -> String {
            Key::Name(String::from(self))
        }
    }

    impl Into<Node> for i64 {
        fn into(self) -> Node {
            Node::Number(Number::Integer(self))
        }
    }

    impl Into<Box<Node>> for i64 {
        fn into(self) -> Node {
            Box::new(Node::Number(Number::Integer(self)))
        }
    }

    impl Into<Node> for f64 {
        fn into(self) -> Node {
            Node::Number(Number::Rational(self))
        }
    }

    impl Into<Box<Node>> for f64 {
        fn into(self) -> Node {
            Box::new(Node::Number(Number::Rational(self)))
        }
    }

    impl Into<Node> for Vec<Node> {
        fn into(self) -> Node {
            Node::List(self)
        }
    }

    impl Into<Key> for &'static Meta {
        fn into(self) -> Key {
            Key::Meta(self)
        }
    }

    fn pair(key: impl Into<Key>, value: impl Into<Node>) -> Node {
        Node::Pair(key.into(), Box::new(value.into()))
    }

    fn create(node: impl Into<Node>) -> Operation {
        Operation::Create(node.into())
    }

    fn destroy(key: key::Key) -> Operation {
        Operation::Destroy(key)
    }

    fn modify(key: key::Key, node: impl Into<Node>) -> Operation {
        Operation::Modify(key, node.into())
    }

    fn increment(node: impl Into<Number>) -> Node {
        Node::Increment(node.into())
    }

    #[test]
    fn database_run_create() {
        use Node::*;

        let database = Database::new();
        database.run([
            create(vec![
                pair(
                    "Position",
                    vec![pair("x", 0.0), pair("y", 1.0), pair("z", 2.0)],
                ),
                pair(
                    velocity(),
                    vec![pair("x", 0.0), pair("y", 1.0), pair("z", 2.0)],
                ),
                pair("Mass", vec![Node::from(0.0)]),
            ]),
            destroy(Key::NULL),
            modify(Key::NULL, vec![pair("Position", pair("x", increment(5.0)))]),
        ]);
    }

    fn position() -> &'static Meta {
        dynamic(
            "component::Position",
            [field("x", r#static::<f64>), field("y", r#static::<f64>)],
        )
    }

    fn velocity() -> &'static Meta {
        dynamic(
            "component::Velocity",
            [field("x", r#static::<f64>), field("y", r#static::<f64>)],
        )
    }

    struct Field {
        offset: usize,
        name: String,
        meta: LazyLock<&'static Meta, Box<dyn FnOnce() -> &'static Meta + Send + Sync + 'static>>,
    }

    struct Meta {
        index: usize,
        name: String,
        layout: Layout,
        fields: Box<[Field]>,
    }

    enum Identifier {
        Type(TypeId),
        Index(usize),
    }

    impl Meta {
        pub unsafe fn new(&self, capacity: usize) -> NonNull<u8> {
            if self.layout.size() == 0 {
                self.layout.dangling().cast()
            } else {
                let (layout, _) = self.layout.repeat(capacity).unwrap_unchecked();
                NonNull::new_unchecked(alloc(layout).cast())
            }
        }

        pub unsafe fn copy(
            &self,
            source: (NonNull<u8>, usize),
            target: (NonNull<u8>, usize),
            count: NonZeroUsize,
        ) {
            if self.layout.size() > 0 {
                let source = source
                    .0
                    .as_ptr()
                    .cast::<u8>()
                    .add(source.1 * self.layout.size());
                let target = target
                    .0
                    .as_ptr()
                    .cast::<u8>()
                    .add(target.1 * self.layout.size());
                copy(source, target, count.get() * self.layout.size());
            }
        }

        pub unsafe fn free(&self, data: NonNull<u8>, capacity: usize) {
            let (layout, _) = self.layout.repeat(capacity).unwrap_unchecked();
            dealloc(data.as_ptr().cast(), layout)
        }
    }

    fn field(
        name: impl Into<String>,
        meta: impl FnOnce() -> &'static Meta + Send + Sync + 'static,
    ) -> Field {
        Field {
            offset: 0,
            name: name.into(),
            meta: LazyLock::new(Box::new(meta)),
        }
    }

    static REGISTRY: RwLock<BTreeMap<String, &'static Meta>> = RwLock::new(BTreeMap::new());

    fn r#static<T: 'static>() -> &'static Meta {
        let name = type_name::<T>().to_string();
        if let Some(meta) = REGISTRY.read().get(&name) {
            return meta;
        }

        let mut registry = REGISTRY.write();
        let index = registry.len();
        registry.entry(name).or_insert_with_key(|key| {
            Box::leak(Box::new(Meta {
                index,
                name: key.clone(),
                layout: Layout::new::<T>(),
                fields: Box::new([]),
            }))
        })
    }

    fn dynamic(name: impl Into<String>, fields: impl IntoIterator<Item = Field>) -> &'static Meta {
        let name = name.into();
        if let Some(meta) = REGISTRY.read().get(&name) {
            return meta;
        }

        let mut registry = REGISTRY.write();
        let index = registry.len();
        registry.entry(name.into()).or_insert_with_key(|key| {
            let mut layout = Layout::new::<()>();
            let fields = fields
                .into_iter()
                .map(|mut field| {
                    let pair = layout.extend(field.meta.layout).unwrap();
                    layout = pair.0;
                    field.offset = pair.1;
                    field
                })
                .collect();
            Box::leak(Box::new(Meta {
                index,
                name: key.clone(),
                layout: layout.pad_to_align(),
                fields,
            }))
        })
    }
}

// mod locks {
//     use std::marker::PhantomData;

//     pub trait Datum {}
//     pub trait Item {}
//     impl<D: Datum> Item for &D {}
//     impl<D: Datum> Item for &mut D {}
//     impl Item for () {}
//     impl<I1: Item> Item for (I1,) {}
//     impl<I1: Item, I2: Item + Allow<I1>> Item for (I1, I2) {}
//     impl<I1: Item, I2: Item + Allow<I1>, I3: Item + Allow<I1> + Allow<I2>> Item for (I1, I2, I3) {}
//     impl Datum for bool {}
//     impl Datum for char {}

//     pub auto trait Safe {}
//     impl<T, U> Safe for (&T, &U) {}
//     impl<T> !Safe for (&mut T, &T) {}
//     impl<T> !Safe for (&mut T, &mut T) {}
//     impl<T> !Safe for (&T, &mut T) {}

//     pub trait Allow<T> {}
//     impl<T, U> Allow<U> for &T where for<'a> (&'a T, U): Safe {}
//     impl<T, U> Allow<U> for &mut T where for<'a> (&'a mut T, U): Safe {}
//     impl<U> Allow<U> for () {}
//     impl<T1, U: Allow<T1>> Allow<U> for (T1,) {}
//     impl<T1, T2, U: Allow<T1> + Allow<T2>> Allow<U> for (T1, T2) {}
//     impl<T1, T2, T3, U: Allow<T1> + Allow<T2> + Allow<T3>> Allow<U> for (T1, T2, T3) {}

//     struct Query<I: Item>(PhantomData<I>);
//     impl<I: Item, U: Allow<I>> Allow<U> for Query<I> {}
//     impl<I: Item> Query<I> {
//         pub fn new() -> Self {
//             todo!()
//         }
//         pub fn each(&mut self, each: impl FnMut(I) + 'static) {}
//         pub fn each_with<S: Allow<I>>(&mut self, state: S, each: impl FnMut(I, S) + 'static) {}
//     }

//     fn boba<T, U>(a: T, b: U)
//     where
//         (T, U): Item,
//     {
//     }
//     fn jango<T: Item>(a: T) {}
//     fn karl(mut query1: Query<&mut bool>) {
//         let a = &mut 'a';
//         let c = &mut true;
//         let e = true;
//         let query2 = Query::<&mut char>::new();
//         query1.each(move |b| *b = e);
//         query1.each_with(a, move |b, c| *b = *c == '0');
//         query1.each_with(query2, move |b, mut query| {
//             query.each_with(b, |c, b| *b = *c == '0')
//         });
//         // query.each_with(c, move |b, c| *b = *c);
//         // query.each(|b| *b = *c);
//         // karl([a], move |b: &bool, a| {
//         //     *a[0] = *b;
//         //     // *c = *a;
//         //     d += 1;
//         // });
//     }
//     fn fett<T: Datum>(value: &T) {
//         // Succeed

//         jango(&true);
//         jango(&mut true);
//         jango((&true, &true));
//         jango((&mut 'a', &true));
//         jango((&true, &mut 'a'));

//         boba(value, &true);
//         boba(&'a', &mut true);
//         boba(&mut 'a', &mut true);
//         boba(&true, (&'a', &true));
//         boba((&'a', &true), &true);
//         boba(value, value);

//         // Fail
//         // boba(&mut 'a', value);
//         // jango(((&true, ((),)), (&mut 'a', (), &mut true)));
//         // jango((&mut true, &true));
//         // jango((&true, &mut true));
//         // jango((&mut true, &mut true));
//         // boba(&false, &mut true);
//         // boba(&mut false, &true);
//         // boba(&true, (&mut 'b', &mut true));
//         // boba(
//         //     (&'a', (&true, &'b', ((), ()))),
//         //     ((), &'b', (&'a', (&mut true, ((), ())))),
//         // );
//     }
// }

// mod push_vec {
//     use std::{
//         mem::{forget, MaybeUninit},
//         ptr::{copy_nonoverlapping, null_mut},
//         sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering::*},
//     };

//     use crate::core::utility::get_unchecked;

//     const SHIFT: usize = 8;
//     const CHUNK: usize = 1 << SHIFT;
//     const MASK: u32 = (CHUNK - 1) as u32;

//     pub struct PushVec<T> {
//         count: AtomicU64,
//         uses: AtomicUsize,
//         chunks: AtomicPtr<Box<[MaybeUninit<T>; CHUNK]>>,
//         pending: AtomicPtr<Box<[MaybeUninit<T>; CHUNK]>>,
//     }

//     impl<T> PushVec<T> {
//         pub fn get(&self, index: u32) -> Option<&T> {
//             let (count, _, _) = decompose_count(self.count.load(Acquire));
//             if index >= count as _ {
//                 return None;
//             }

//             let (chunk, item) = decompose_index(index);
//             let (count, _) = decompose_index(count);
//             self.increment_use(count - 1);
//             let chunks = self.chunks.load(Acquire);
//             let chunk = unsafe { &**chunks.add(chunk as usize) };
//             self.decrement_use(count - 1);
//             Some(unsafe { get_unchecked(chunk, item as usize).assume_init_ref() })
//         }

//         pub fn push(&self, item: T) {
//             let (mut count, ended, begun) = decompose_count(self.count.fetch_add(1, AcqRel));
//             let index = count + ended as u32 + begun as u32;
//             let (mut old_count, _) = decompose_index(count);
//             let new_count = decompose_index(index);
//             self.increment_use(old_count);
//             let mut old_chunks = self.chunks.load(Acquire);

//             debug_assert_eq!(new_count.0 - old_count, 1);
//             if old_count < new_count.0 {
//                 // TODO: Re-read the count here? In a loop?
//                 let new_chunks = {
//                     let mut chunks = Vec::with_capacity(new_count.0 as usize);
//                     let new_chunks = chunks.as_mut_ptr();
//                     unsafe { copy_nonoverlapping(old_chunks, new_chunks, old_count as usize) };
//                     forget(chunks);
//                     new_chunks
//                 };

//                 match self
//                     .chunks
//                     .compare_exchange(old_chunks, new_chunks, AcqRel, Acquire)
//                 {
//                     Ok(chunks) => {
//                         let chunk = Box::new([(); CHUNK].map(|_| MaybeUninit::<T>::uninit()));
//                         unsafe { new_chunks.add(old_count as usize).write(chunk) };
//                         // It should be extremely unlikely that this call returns `true`.
//                         self.try_free(old_count, old_chunks);
//                         old_chunks = chunks;
//                     }
//                     Err(chunks) => {
//                         // Another thread won the race; free this allocation.
//                         drop(unsafe { Vec::from_raw_parts(new_chunks, 0, new_count.0 as usize) });
//                         old_chunks = chunks;
//                     }
//                 }
//                 (count, _, _) = decompose_count(self.count.load(Acquire));
//                 (old_count, _) = decompose_index(count);
//             }

//             let chunk = unsafe { &mut **old_chunks.add(new_count.0 as usize) };
//             self.decrement_use(old_count - 1);
//             let item = MaybeUninit::new(item);
//             unsafe { chunk.as_mut_ptr().add(new_count.1 as usize).write(item) };
//             let result = self.count.fetch_update(AcqRel, Acquire, |count| {
//                 let (count, ended, begun) = decompose_count(count);
//                 Some(if begun == 1 {
//                     recompose_count(count + ended as u32 + begun as u32, 0, 0)
//                 } else {
//                     debug_assert!(begun > 1);
//                     recompose_count(count, ended + 1, begun - 1)
//                 })
//             });
//             debug_assert!(result.is_ok());
//         }

//         #[inline]
//         fn increment_use(&self, count: u32) {
//             if self.uses.fetch_add(1, Relaxed) == 0 {
//                 self.try_free(count, null_mut());
//             }
//         }

//         #[inline]
//         fn decrement_use(&self, count: u32) {
//             if self.uses.fetch_sub(1, Relaxed) == 1 {
//                 self.try_free(count, null_mut());
//             }
//         }

//         #[inline]
//         fn try_free(&self, count: u32, swap: *mut Box<[MaybeUninit<T>; CHUNK]>) -> bool {
//             let pending = self.pending.swap(swap, AcqRel);
//             if pending.is_null() {
//                 false
//             } else {
//                 drop(unsafe { Vec::from_raw_parts(pending, 0, count as usize) });
//                 true
//             }
//         }
//     }

//     #[inline]
//     const fn decompose_index(index: u32) -> (u32, u32) {
//         (index >> SHIFT, index & MASK)
//     }

//     #[inline]
//     const fn recompose_count(count: u32, ended: u16, begun: u16) -> u64 {
//         (count as u64) << 32 | (ended as u64) << 16 | (begun as u64)
//     }

//     #[inline]
//     const fn decompose_count(count: u64) -> (u32, u16, u16) {
//         ((count >> 32) as u32, (count >> 16) as u16, count as u16)
//     }
// }
