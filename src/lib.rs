#![feature(impl_trait_in_assoc_type)]

pub mod core;
pub mod create;
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

use key::Key;
use parking_lot::Mutex;
use resources::Resources;
use std::{
    alloc::{Layout, LayoutError},
    any::{TypeId, type_name},
    collections::BTreeMap,
    error, fmt,
    mem::{needs_drop, size_of},
    num::NonZeroUsize,
    ptr::{NonNull, copy, drop_in_place, slice_from_raw_parts_mut},
};
pub use that_base_derive::{Datum, Filter, Template};

/*
    TODO: Implement async versions of each operation?
    TODO: Remove an indirection in table to the columns since tables are always kept in an 'Arc'. See 'slice-dst' crate.
        struct Header {
            index: u32,
            pub(crate) count: AtomicUsize,
            pub(crate) keys: RwLock<Keys>,
        }
        type Table = SliceWithHeader<Header, Column>;
    TODO: Use the 'erasable' crate to represent a pointer in a column?
    TODO: A parallel version of 'fold_swap' such that many query operations can be executed in parallel (ex: 'each' -> 'par_each').
    TODO: Dynamic disk/network save/load of data.
        - In `enum Data { Unloaded, Loaded(NonNull<()>) }` such that `Column::data` is of type `RwLock<Data>`.
        - Tables will be able to save/load their data to some target. Whenever the data is requested, the table will load it.
        - Add `memory_pressure_threshold` parameter to `Database`.
            - It will represent the memory usage at which the `Database` should start offloading its data to some external storage.
            - Memory usage may temporarily go over the threshold, but when the `Database` is "at rest" (i.e. no operations are ongoing),
            its memory usage should remain `<= memory_pressure_threshold`.
        - Add `maximum_memory_size` parameter to `Database`.
            - It represents the absolute memory limit that the `Database` may reach. If the `Database` were to have more in memory data
            than `maximum_memory_size`, it will use more drastic measures to always stay `<= maximum_memory_size`.
            - `assert!(maximum_memory_size >= memory_pressure_threshold);`
        - Track current memory size of the `Database` and expose it.
        - Add `maximum_table_size` parameter to `Database`.
            - `assert!(maximum_table_size >= table_item_size);`
            - `maximum_table_size` must always be greater than one table item (unlikely a problem, but still).
            - When a table grows, it can only grow up to `maximum_table_size / table_item_size` item count. If there is an overflow,
            create a new table with the same schema.
            - It may be worth factoring out the schema from the tables since many tables may share the same schema.
        - Memory policy must be applied on table creation, table growth and on table load.
            - Ex: tables that are less accessed can be dumped to storage

    TODO: Tables vs Chunks architecture.
        - Tables:
            - Have a 1 to 1 mapping between schemas and storage.
            - Have a variable size from 0 to u32::MAX.
            - Slots need 84-96 bits to represent the location of keys (32 bits: generation, 20-32 bits: table, 32 bits: row).
            - When creating a row, there is no ambiguity as to which table will hold it. As such, `Create` operations can keep
            a single reference to the target table.
            - Very cache friendly.
        - Chunks
            - Storage grows in 'chunks' of 256.
            - Keys can be held in an array RwLock<[Key; 256]>; thus one less indirection.
            - Slots need 64 bits to hold the location of keys (32 bits: generation, 24 bits: table/chunk, 8 bits: row).
            - Allows for more parallelism since there are more locks.
            - May allow for more optimization around locking especially when moving a row. After taking the source lock,
            if none of the target locks can be taken, a new chunk can be created.
            - Will likely be wasteful of memory.
            - Most operations already operate on multiple tables, so they would need little adaptation.
            - Will complexify the logic of `Create`.

    TODO: Add filters to events.
        - database.listen::<OnCreate>().filter::<Has<Mass>>();
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
        - It may be reasonable to require from users (or a scheduler) to resolve deferred operations at an appropriate time.
    TODO: Implement a table `If` operation.
        - The if checks some condition on tables with an upgradable lock and executes its body with a write lock on success.
        - This ensures that no structural operation has occurred on the table between the check and the body.
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
        - Imposes a design dilemma on the user:
            - Use more static templates with dynamic values (larger stores).
            - Use more dynamic templates (using `Add/Remove` operations) with static datum (smaller stores).
        - Tables may have a `size: u32` that defaults to `u32::MAX` that defines the maximum store size.
        - After the first chunk has grown to full size (ex: 1024), change growing strategy to allocating
        full chunks with separate locks.
        - This removes the dilemma from the user in favor of more static templates since they'll have similar
        locality and parallelism properties compared to more dynamic templates, but in the dynamic case, there'll be more
        move operations.
        - Should offer some benefits in terms in parallelism (parallelize more granularly by chunk).
        - Should offer some opportunities when moving/destroying rows such that only one chunk will need to be locked.
            - Each chunk may have its own `count` such that `squash` can be used within a chunk.
        - This solution will add a lot of complexity.
        - Will this require table + chunk + column locks vs table + column locks (currently)?
        *** It seems that locality should be prioritize since it allows work on columns do be completed sooner which reduces
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
        - Same benefits and inconveniences as `Add<T>`.
        - Requires the reverse ordering as `Add<T>`.
        - Do not allow `Add`-like and `Remove`-like operators in the same query.
    TODO (POSTPONED): Implement nest operations for query:
        - Forces the outer query to take additional locks in case the inner query needs to produce the same item as the outer query:
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
    events: event::State,
    resources: Resources,
}

type MetaCopy = unsafe fn((NonNull<u8>, usize), (NonNull<u8>, usize), NonZeroUsize);
type MetaDrop = unsafe fn(NonNull<u8>, usize, NonZeroUsize);
pub struct Meta {
    identifier: TypeId,
    name: &'static str,
    size: usize,
    layout: fn(usize) -> Result<Layout, LayoutError>,
    copy: Option<MetaCopy>,
    drop: Option<MetaDrop>,
}

pub trait Datum: Sized + 'static {}

impl Meta {
    #[inline]
    pub fn get<T: Sized + 'static>() -> &'static Meta {
        static METAS: Mutex<BTreeMap<TypeId, &'static Meta>> = Mutex::new(BTreeMap::new());
        METAS
            .lock()
            .entry(TypeId::of::<T>())
            .or_insert_with_key(|&key| {
                Box::leak(Box::new(Meta {
                    identifier: key,
                    name: type_name::<T>(),
                    size: size_of::<T>(),
                    layout: Layout::array::<T>,
                    copy: if size_of::<T>() > 0 {
                        Some(|source, target, count| unsafe {
                            let source = source.0.as_ptr().cast::<T>().add(source.1);
                            let target = target.0.as_ptr().cast::<T>().add(target.1);
                            copy(source, target, count.get());
                        })
                    } else {
                        None
                    },
                    drop: if needs_drop::<T>() {
                        Some(|data, index, count| unsafe {
                            let data = data.as_ptr().cast::<T>().add(index);
                            drop_in_place(slice_from_raw_parts_mut(data, count.get()));
                        })
                    } else {
                        None
                    },
                }))
            })
    }

    #[inline]
    pub const fn identifier(&self) -> TypeId {
        self.identifier
    }

    #[inline]
    pub const fn name(&self) -> &'static str {
        self.name
    }

    #[inline]
    pub const fn size(&self) -> usize {
        self.size
    }
}

impl Database {
    pub fn new() -> Self {
        Self {
            keys: key::State::new(),
            tables: table::State::new(),
            resources: Resources::new(),
            events: event::State::new(),
        }
    }
}

impl Default for Database {
    fn default() -> Self {
        Self::new()
    }
}

mod next_table_based {
    /*
        DESIGN:
        - Keys are static in the table. This implies that fragmentation will happen.
        - Fragments of a table can be un/locked independently, similarly to a chunk.
        - A fragment spans from the first valid row to the last invalid row.
        - A list of fragments is maintained in each tables as pairs of (index: usize, count: usize).
        - Tables try to defragment the data when possible.
        - Query iterators iterate over fragments by default (giving out slices [T]), but a wrapper chunks iterator might
        try to split/combine fragments to spread the load more evenly when parallelizing.
    */

    pub struct Table {
        fragments: Vec<(usize, usize)>,
    }
}

mod next_chunk_based {
    /*
        DESIGN:
        - Make layouts maximally vertical. This means that every leaf/primitive (supported) type has
        its own column and is addressable.
            - A 'Chunk' is a big 2D matrix of bytes: (header, [rows x columns; u8]).
            - Position { x: f64, y: f64, z: f64 } would be layed out as ([f64], [f64], [f64]).
            - A macro could generate the individual queries:
            impl Position {
                pub const X: Query<f64> = ...;
                pub const Y: Query<f64> = ...;
                pub const Z: Query<f64> = ...;
                pub const ALL: (Query<f64>, Query<f64>, Query<f64>) = (X, Y, Z);
            }
        - Records are static once created in the sense that they cannot change structure.
            - This allows storing more direct information in the 'Key' structure such as the 'Chunk' index.
            - There still needs to be an indirection in the 'Chunk' to defragment columns when records are deleted.
            - Thus, 'Key' would have the following content: { generation: u32, chunk: u24, index: u8 }.
            - The 'index' points to a row in the 'Chunk' where another u8 is stored that points to the actual row of the record.
            - This setup can improve the performance of joins since the structure of a record can be checked very efficiently
            using its 'Chunk' index against a set of valid 'Chunk' indices.
        - Implement structure dynamism of data using 'Key' graph edges (such as '[#derive(Edge)] Parent(Key)').
            - When a static structure in not dynamic enough, links between records can be created and broken using the graph.
            - Given the query '(&mut Position, Parent<&Position>)':
                - The set of valid 'Chunk' indices for the base query and the parent query are cached in the query state.
                - Collect all parent 'Key' from its base 'Chunk's where their 'Chunk' index is in the parent set.
                - Sort the collected keys by the base then parent 'Chunk' indices.
                - Run the query for each group of (base, parent) indices by locking at most 2 'Chunk's at a time.
                - If a `Parent` link has changed between the collection of `Key`s and the query execution, skip this item.
            - The 'Child' query may exist for convenience, but they would be converted to its equivalent 'Parent' query.
        - 'Key' may have configurable layouts to allow more smaller/larger 'Chunk's or allow smaller/larger 'Key's.
            - Key<u32, u24, u8> is the default.
            - Key<u16, u[8-12], u[4-8]> is a small 'Key'.
            - Key<u32, u[16-24], u[u8-u16]> is a standard 'Key'.
            - Key<u64, u[32-56], u[u8-u32]> is a large 'Key'.
            - The chosen 'Key' layout needs to be consistent across the database.
        - A 'Chunk' can be saved to disk and loaded from disk whenever memory pressure is too high.
            - Memory constraints are configurable at the database level.
            - Different dump policy may be chosen:
                - Dump the oldest 'Chunk'.
                - Dump the largest 'Chunk'.
                - Dump the least frequently accessed 'Chunk'.
                - Combine multiple policies.
            - A dump policy would simply calculate a score for each 'Chunk' and dump 'Chunk's with the highest score
            until memory pressure is within bounds.
            - The dump process would run whenever the database needs to make a new allocation and realizes that memory
            constraints would not be respected.
            - In order to dump data that contains externally allocated data (such as with 'Box', 'Vec', 'Arc', 'String', etc.),
            the dump would save the raw pointer address without running the 'drop' code, effectively 'leaking' that memory.
                - When restoring the pointers, it would be assumed that their pointed-to memory is still there.
                - This means that *any* 'Box<T>/etc.' is supported in the database.
                - Note that this forces a difference between a dump and serialization of the data.
                - Serialization stores the pointed-to memory and will need to store shared references links.
                - When clearing or dropping a dumped chunk, it will need to be loaded from disk and then dropped.

        QUESTIONS:
        - How can 'Chunk' try to make the use of its 'generation' budget as uniform as possible?
            - Maintain a list of free rows sorted by their 'generation'? Add to that list on deletion?
        - How can 'Sibling', 'Ancestor' and 'Descendant' queries be implemented efficiently?
            - The query would need to run for each sibling/ancestor/descendant with the base record.
            - If the base record and all records for the descendants would be accessed in a single iteration
            of the query, this could result in locking an arbitrary number of 'Chunk's (at words, one lock per descendant).
        - What if a query has many family queries (ex: (Parent, Descendant, Child))?
            - One `Chunk` lock per family query could be allowed (ex: (Parent, Descendant, Child) would lock at most 3 'Chunk's at once).
        - Integrate asynchronicity in the database?
            - Would likely add a cost to every operation.
            - Most useful parallel operations can be accomplished outside of the database through iterators (ex: the 'Splits' iterator).
        - How to make the dump to disk process efficient?
        - Would the memory pressure calculation consider external allocations (such as Box<T> or Vec<T>)?
            - Since external allocations are not dumped to disk and depend on their persistence in memory, I think the answer is no.
        - Can this database be distributed?
            - Since this database allows for arbitrary operations on its data by given direct access to its memory,
            it is going to be very tricky to maintain a history of operations that would allow to replicate the
            database's state on every node.
            - When a write lock is taken on a column, it has to be assumed that every value has been set to anything,
            thus when synchronizing, all values need to be sent to every node. Even with compression, this doesn't
            scale nicely.
            - The database may support SQL or GraphQL queries that would allow for easier synchronization by taking
            advantage of the knowledge of the mutation embedded in those queries.
                - Ex: { $add: { "position.x": 1 } } would add the value of 1 to each record's 'position.x' and would
                only require synchronizing this operation rather than all the values.
            - The generational index mechanism would not work well for distribution.
            - Using 'uuid's would cause 'Key's to be 16 bytes in size and likely require an indirection that maps
            a 'Key' to its 'Chunk' because it is going to be almost impossible to guarantee the same 'Chunk' ordering
            between instances of the database, thus the same 'Key' must be allowed to point to different 'Chunk's
            depending on the specific database instance it is running on.
    */

    use core::{
        alloc::Layout,
        marker::PhantomData,
        mem::{ManuallyDrop, forget},
        ops::Deref,
        ptr::{NonNull, drop_in_place, null_mut},
        slice::{from_raw_parts, from_ref},
        sync::atomic::{AtomicPtr, AtomicU8, AtomicU32, Ordering},
    };
    use parking_lot::RwLock;
    use static_assertions::{const_assert, const_assert_eq};
    use std::{
        alloc::{alloc, dealloc},
        sync::{Arc, Weak},
    };

    pub trait Keyed {
        type Generation;
        type Chunk;
        type Index;
        const NULL: Self;

        fn generation(&self) -> Self::Generation;
        fn chunk(&self) -> Self::Chunk;
        fn index(&self) -> Self::Index;
    }

    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub struct Key<T = u64, const G: usize = 64, const C: usize = 24, const I: usize = 8>(T) 
        where T: Keyed<Generation = G, Chunk = C, Index = I>;

    macro_rules! key {
        ($type: ident <$gs: literal : $gt: ty, $cs: literal : $ct: ty, $is: literal : $it: ty>) => {
            const_assert!(size_of::<usize>() * 8 >= $gs);
            const_assert!(size_of::<usize>() * 8 >= $cs);
            const_assert!(size_of::<usize>() * 8 >= $is);
            const_assert_eq!(size_of::<$type>() * 8, { $gs + $cs + $is });

            impl Keyed for Key<$type, $gs, $cs, $is> {
                type Chunk = $ct;
                type Generation = $gt;
                type Index = $it;
                const NULL: Self = Self($type::MAX);

                #[inline]
                fn generation(&self) -> Self::Generation {
                    let shift = 0;
                    let mask = 1 << $gs;
                    (self.0 >> shift) as $gt & mask
                }

                #[inline]
                fn chunk(&self) -> Self::Chunk {
                    let shift = $gs + $is;
                    let mask = 1 << $cs;
                    (self.0 >> shift) as $ct & mask
                }

                #[inline]
                fn index(&self) -> Self::Index {
                    let shift = $gs;
                    let mask = 1 << $is;
                    (self.0 >> shift) as $it & mask
                }
            }
        };
    }

    key!(u128<64: u64, 56: u64, 8: u8>);
    key!(u128<64: u64, 55: u64, 9: u16>);
    key!(u128<64: u64, 54: u64, 10: u16>);
    key!(u128<64: u64, 53: u64, 11: u16>);
    key!(u128<64: u64, 52: u64, 12: u16>);
    key!(u128<64: u64, 51: u64, 13: u16>);
    key!(u128<64: u64, 50: u64, 14: u16>);
    key!(u128<64: u64, 49: u64, 15: u16>);
    key!(u128<64: u64, 48: u64, 16: u16>);
    key!(u128<64: u64, 47: u64, 17: u32>);
    key!(u128<64: u64, 46: u64, 18: u32>);
    key!(u128<64: u64, 45: u64, 19: u32>);
    key!(u128<64: u64, 44: u64, 20: u32>);
    key!(u128<64: u64, 43: u64, 21: u32>);
    key!(u128<64: u64, 42: u64, 22: u32>);
    key!(u128<64: u64, 41: u64, 23: u32>);
    key!(u128<64: u64, 40: u64, 24: u32>);
    key!(u128<64: u64, 39: u64, 25: u32>);
    key!(u128<64: u64, 38: u64, 26: u32>);
    key!(u128<64: u64, 37: u64, 27: u32>);
    key!(u128<64: u64, 36: u64, 28: u32>);
    key!(u128<64: u64, 35: u64, 29: u32>);
    key!(u128<64: u64, 34: u64, 30: u32>);
    key!(u128<64: u64, 33: u64, 31: u32>);
    key!(u128<64: u64, 32: u32, 32: u32>);

    key!(u64<32: u32, 24: u32, 8: u8>);
    key!(u64<32: u32, 23: u32, 9: u16>);
    key!(u64<32: u32, 22: u32, 10: u16>);
    key!(u64<32: u32, 21: u32, 11: u16>);
    key!(u64<32: u32, 20: u32, 12: u16>);
    key!(u64<32: u32, 19: u32, 13: u16>);
    key!(u64<32: u32, 18: u32, 14: u16>);
    key!(u64<32: u32, 17: u32, 15: u16>);
    key!(u64<32: u32, 16: u16, 16: u16>);

    key!(u32<16: u16, 12: u16, 4: u8>);
    key!(u32<16: u16, 11: u16, 5: u8>);
    key!(u32<16: u16, 10: u16, 6: u8>);
    key!(u32<16: u16, 9: u16, 7: u8>);
    key!(u32<16: u16, 8: u8, 8: u8>);

    key!(u16<8: u8, 6: u8, 2: u8>);
    key!(u16<8: u8, 5: u8, 3: u8>);
    key!(u16<8: u8, 4: u8, 4: u8>);

    /// Store the state of the keys in the 'K::Chunk' bits.
    struct Rows<K>(usize, RwLock<NonNull<K>>);
    struct Columns(usize, NonNull<Column>);
    struct Column(RwLock<NonNull<u8>>);

    pub struct Chunk<K: Keyed> {
        index: usize,
        rows: Rows<K>,
        columns: Columns,
    }

    pub struct Store<K: Keyed> {
        chunks: RwLock<Vec<Arc<Chunk<K>>>>,
    }

    impl<K: Keyed> Chunk<K> {
        pub fn new(index: usize) -> Box<Self> {
            todo!()
        }
    }
}

// mod karl {

//     use core::ptr::NonNull;

//     #[derive(Debug, Clone, PartialEq, Eq)]
//     enum Layout {
//         Unit,
//         U8,
//         U16,
//         U32,
//         U64,
//         U128,
//         USize,
//         I8,
//         I16,
//         I32,
//         I64,
//         I128,
//         ISize,
//         F32,
//         F64,
//         Bool,
//         Char,
//         Array(Box<Layout>, usize), // [item; count]
//         All(Vec<Layout>),          // (items,*)
//         Any(Vec<Layout>),          // (u8|u16|u32|u64|u128, max(items))
//         Name(String, Box<Layout>), // item
//     }

//     trait Lay {
//         fn layout() -> Layout;
//     }

//     struct Pointer(NonNull<u8>, Layout);

//     impl Pointer {
//         pub unsafe fn get<T: Lay>(&self, path: &str) -> Option<&T> {
//             let pair = self.1.offset(path)?;
//             if pair.1 == &T::layout() {
//                 Some(&*self.0.as_ptr().add(pair.0).cast())
//             } else {
//                 None
//             }
//         }

//         pub unsafe fn get_mut<T: Lay>(&mut self, path: &str) -> Option<&mut
// T> {             let pair = self.1.offset(path)?;
//             if pair.1 == &T::layout() {
//                 Some(&mut *self.0.as_ptr().add(pair.0).cast())
//             } else {
//                 None
//             }
//         }
//     }

//     impl Layout {
//         pub fn offset(&self, path: &str) -> Option<(usize, &Self)> {
//             if path.is_empty() {
//                 return Some((0, self));
//             }

//             let (part, path) = path.split_once('.').unwrap_or((path, ""));
//             match self {
//                 Layout::Array(item, count) => match path.parse::<usize>() {
//                     Ok(index) if index < *count => {
//                         let pair = item.offset(path)?;
//                         Some((pair.0 + item.size() * index, pair.1))
//                     }
//                     _ => None,
//                 },
//                 Layout::All(items) => match path.parse::<usize>() {
//                     Ok(mut index) => {
//                         let mut offset = 0;
//                         for item in items {
//                             if index == 0 {
//                                 let pair = item.offset(path)?;
//                                 return Some((pair.0 + offset, pair.1));
//                             }
//                             offset += item.size();
//                             index -= 1;
//                         }
//                         None
//                     }
//                     Err(_) => {
//                         let mut offset = 0;
//                         for item in items {
//                             if let Layout::Name(name, item) = item {
//                                 if part == name {
//                                     let pair = item.offset(path)?;
//                                     return Some((pair.0 + offset, pair.1));
//                                 }
//                             }
//                             offset += item.size();
//                         }
//                         None
//                     }
//                 },
//                 _ => None,
//             }
//         }

//         fn size(&self) -> usize {
//             match self {
//                 Layout::Unit => size_of::<()>(),
//                 Layout::U8 => size_of::<u8>(),
//                 Layout::U16 => size_of::<u16>(),
//                 Layout::U32 => size_of::<u32>(),
//                 Layout::U64 => size_of::<u64>(),
//                 Layout::U128 => size_of::<u128>(),
//                 Layout::USize => size_of::<usize>(),
//                 Layout::I8 => size_of::<i8>(),
//                 Layout::I16 => size_of::<i16>(),
//                 Layout::I32 => size_of::<i32>(),
//                 Layout::I64 => size_of::<i64>(),
//                 Layout::I128 => size_of::<i128>(),
//                 Layout::ISize => size_of::<isize>(),
//                 Layout::F32 => size_of::<f32>(),
//                 Layout::F64 => size_of::<f64>(),
//                 Layout::Bool => size_of::<bool>(),
//                 Layout::Char => size_of::<char>(),
//                 Layout::Array(item, count) => item.size() * *count,
//                 // TODO: Think about padding.
//                 Layout::All(items) => items.iter().map(|item|
// item.size()).sum(),                 Layout::Any(items) if items.len() < (1 <<
// 8) => {                     size_of::<u8>() + items.iter().map(|item|
// item.size()).max().unwrap_or(0)                 }
//                 Layout::Any(items) if items.len() < (1 << 16) => {
//                     size_of::<u16>() + items.iter().map(|item|
// item.size()).max().unwrap_or(0)                 }
//                 Layout::Any(items) if items.len() < (1 << 32) => {
//                     size_of::<u32>() + items.iter().map(|item|
// item.size()).max().unwrap_or(0)                 }
//                 Layout::Any(items) if items.len() < (1 << 64) => {
//                     size_of::<u64>() + items.iter().map(|item|
// item.size()).max().unwrap_or(0)                 }
//                 Layout::Any(items) => {
//                     size_of::<u128>() + items.iter().map(|item|
// item.size()).max().unwrap_or(0)                 }
//                 Layout::Name(_, item) => item.size(),
//             }
//         }
//     }

//     /*
//         [repr(transparent)]
//         struct Position([f64; 3]);

//         impl Lay for Position {
//             fn layout() -> Layout {
//                 Layout::Array { item: Box::new(Layout::F64), count: 3 }
//             }
//         }

//         impl Lay for () {
//             fn layout() -> Layout {
//                 Layout::Unit
//             }
//         }

//         impl Lay for usize {
//             fn layout() -> Layout {
//                 Layout::USize
//             }
//         }

//         impl<T: Lay> Lay for [T] {
//             fn layout() -> Layout {
//                 Layout::Slice { item: Box::new(T::layout()) }
//             }
//         }

//         impl<T: Lay, const N: usize> for [T; N] {
//             fn layout() -> Layout {
//                 Layout::Array { item: Box::new(T::layout()), count: N }
//             }
//         }

//         let store = Store::new();
//         let key = store.insert([(1usize, 2usize), (3usize, 4usize)]);
//         if let Ok(value) = store.get_mut<usize>(key, (0, 1)) {
//             *value += 1;
//         }

//         for values in store.get_chunks_mut((0, 1)) {
//             for value in values {
//                 *value += 1;
//             }
//         }
//     */
// }

mod store {
    use core::{
        alloc::{Layout, LayoutError},
        any::{TypeId, type_name},
        cell::OnceCell,
        marker::PhantomData,
        mem::{ManuallyDrop, needs_drop},
        num::NonZeroUsize,
        ops::{Deref, DerefMut},
        ptr::{NonNull, copy, drop_in_place, slice_from_raw_parts_mut},
        slice::{from_raw_parts, from_raw_parts_mut},
        sync::atomic::{AtomicU8, AtomicU32},
    };
    use parking_lot::{Mutex, RwLock};
    use std::{
        alloc::{alloc, dealloc},
        collections::BTreeMap,
        sync::{Arc, OnceLock},
    };

    #[repr(transparent)]
    struct U24([u8; 3]);

    pub struct Schema {
        identifier: Option<TypeId>,
        name: &'static str,
        path: &'static str,
        layout: Layout,
        drop: Option<unsafe fn(NonNull<u8>, usize, NonZeroUsize)>,
        content: Content,
    }

    #[derive(Debug, Clone)]
    pub enum Content {
        Unit,
        U8,
        U16,
        U32,
        U64,
        U128,
        USize,
        I8,
        I16,
        I32,
        I64,
        I128,
        ISize,
        F32,
        F64,
        Bool,
        Char,
        String,
        All(Vec<Content>),
        Any(Vec<Content>),
    }

    #[repr(C)]
    struct PositionI2D(i32, i32);

    #[repr(C)]
    enum Status {
        Frozen(usize),
        Dead(bool),
        Shrunk(f64, f64),
    }

    impl PositionI2D {
        pub fn schema() -> &'static Schema {
            static SCHEMA: OnceLock<Schema> = OnceLock::new();
            SCHEMA.get_or_init(|| Schema {
                identifier: Some(TypeId::of::<Self>()),
                name: "PositionI2D",
                path: type_name::<Self>(),
                layout: Layout::new::<Self>(),
                drop: if needs_drop::<Self>() {
                    Some(|data, index, count| unsafe {
                        let data = data.as_ptr().cast::<Self>().add(index);
                        drop_in_place(slice_from_raw_parts_mut(data, count.get()));
                    })
                } else {
                    None
                },
                content: Content::All(vec![Content::I32, Content::I32]),
            })
        }
    }

    impl Status {
        pub fn schema() -> &'static Schema {
            static SCHEMA: OnceLock<Schema> = OnceLock::new();
            SCHEMA.get_or_init(|| {
                let content = Content::Any(vec![
                    Content::USize,
                    Content::Bool,
                    Content::All(vec![Content::F64, Content::F64]),
                ]);
                Schema {
                    identifier: None,
                    name: "Status",
                    path: "::that_bass::store::Status",
                    layout: content.layout().unwrap(),
                    drop: None,
                    content,
                }
            })
        }
    }

    impl Content {
        pub fn layout(&self) -> Result<Layout, LayoutError> {
            Ok(match self {
                Content::Unit => Layout::new::<()>(),
                Content::U8 => Layout::new::<u8>(),
                Content::U16 => Layout::new::<u16>(),
                Content::U32 => Layout::new::<u32>(),
                Content::U64 => Layout::new::<u64>(),
                Content::U128 => Layout::new::<u128>(),
                Content::USize => Layout::new::<usize>(),
                Content::I8 => Layout::new::<i8>(),
                Content::I16 => Layout::new::<i16>(),
                Content::I32 => Layout::new::<i32>(),
                Content::I64 => Layout::new::<i64>(),
                Content::I128 => Layout::new::<i128>(),
                Content::ISize => Layout::new::<isize>(),
                Content::F32 => Layout::new::<f32>(),
                Content::F64 => Layout::new::<f64>(),
                Content::Bool => Layout::new::<bool>(),
                Content::Char => Layout::new::<char>(),
                Content::String => Layout::new::<String>(),
                Content::All(items) => {
                    let mut layout = Layout::from_size_align(0, 1)?;
                    for item in items {
                        (layout, _) = layout.extend(item.layout()?)?;
                    }
                    layout.pad_to_align()
                }
                Content::Any(items) => {
                    let layout = if items.len() <= u8::MAX as _ {
                        Content::U8
                    } else if items.len() <= u16::MAX as _ {
                        Content::U16
                    } else if items.len() <= u32::MAX as _ {
                        Content::U32
                    } else if items.len() <= u64::MAX as _ {
                        Content::U64
                    } else {
                        Content::U128
                    }
                    .layout()?;

                    let mut size = 0;
                    let mut align = 0;
                    for item in items {
                        let layout = item.layout()?;
                        size = size.max(layout.size());
                        align = align.max(layout.align());
                    }
                    let (layout, _) = layout.extend(Layout::from_size_align(size, align)?)?;
                    layout.pad_to_align()
                }
            })
        }
    }

    struct GrowVec<T>(PhantomData<T>);

    #[repr(C)]
    struct Key {
        generation: u32,
        chunk: U24,
        index: u8,
    }

    #[repr(C)]
    struct Rows {
        count: AtomicU8,
        capacity: u8,
        generations: NonNull<AtomicU32>,
        indices: RwLock<NonNull<u8>>,
    }

    #[repr(C)]
    struct Column {
        schema: Schema,
        data: RwLock<NonNull<u8>>,
    }

    #[repr(C)]
    struct Columns {
        count: usize,
        items: NonNull<Column>,
    }

    #[repr(C)]
    struct Chunk {
        layout: Layout,
        rows: Rows,
        columns: Columns,
    }

    impl Schema {
        #[inline]
        pub fn get<T: Sized + 'static>() -> &'static Self {
            static STATICS: Mutex<BTreeMap<TypeId, &'static Schema>> = Mutex::new(BTreeMap::new());
            STATICS
                .lock()
                .entry(TypeId::of::<T>())
                .or_insert_with_key(|&key| {
                    Box::leak(Box::new(Schema {
                        identifier: Some(key),
                        name: type_name::<T>(),
                        path: type_name::<T>(),
                        layout: Layout::new::<T>(),
                        drop: if needs_drop::<T>() {
                            Some(|data, index, count| unsafe {
                                let data = data.as_ptr().cast::<T>().add(index);
                                drop_in_place(slice_from_raw_parts_mut(data, count.get()));
                            })
                        } else {
                            None
                        },
                        content: Content::Unit,
                    }))
                })
        }
    }

    impl Deref for Columns {
        type Target = [Column];

        fn deref(&self) -> &Self::Target {
            unsafe { from_raw_parts(self.items.as_ptr(), self.count) }
        }
    }

    impl DerefMut for Columns {
        fn deref_mut(&mut self) -> &mut Self::Target {
            unsafe { from_raw_parts_mut(self.items.as_ptr(), self.count) }
        }
    }

    impl Chunk {
        pub unsafe fn new<I: IntoIterator<Item = Schema>>(
            rows: u8,
            columns: I,
        ) -> Result<Box<Self>, LayoutError> {
            let schemas = columns.into_iter().collect::<Vec<_>>();
            // schemas.sort_unstable_by_key(|meta| meta.identifier);
            // schemas.dedup_by_key(|meta| meta.identifier);

            let count = schemas.len();
            let layout = Layout::new::<Chunk>();
            let (layout, columns) = layout.extend(Layout::array::<Column>(count)?)?;
            let (layout, generations) = layout.extend(Layout::array::<AtomicU32>(rows as _)?)?;
            let (layout, indices) = layout.extend(Layout::array::<u8>(rows as _)?)?;
            let mut layout = layout;
            let mut pairs = Vec::with_capacity(count);
            for schema in schemas {
                let array = Layout::from_size_align(
                    schema.layout.size() * rows as usize,
                    schema.layout.align(),
                )?;
                let pair = layout.extend(array)?;
                layout = pair.0;
                pairs.push((schema, pair.1));
            }
            let layout = layout.pad_to_align();
            let pointer = NonNull::new_unchecked(alloc(layout));
            let chunk = pointer.cast::<Chunk>();
            let generations = pointer.add(generations).cast();
            let indices = pointer.add(indices);
            let columns = pointer.add(columns).cast();
            from_raw_parts_mut(generations.cast::<u32>().as_ptr(), rows as _).fill(0);
            from_raw_parts_mut(indices.as_ptr(), rows as _).fill(0);
            for (i, (meta, offset)) in pairs.into_iter().enumerate() {
                columns.add(i).write(Column {
                    schema: meta,
                    data: pointer.add(offset).into(),
                });
            }
            chunk.write(Chunk {
                layout,
                rows: Rows {
                    count: AtomicU8::new(0),
                    capacity: rows,
                    generations,
                    indices: indices.into(),
                },
                columns: Columns {
                    count,
                    items: columns,
                },
            });
            Ok(Box::from_raw(chunk.as_ptr()))
        }
    }

    impl Drop for Chunk {
        fn drop(&mut self) {
            // let count = *self.rows.count.get_mut();
            // if let Some(count) = NonZeroUsize::new(count as _) {
            //     for column in self.columns.iter_mut() {
            //         if let Some(drop) = column.schema.drop {
            //             let data = *column.data.get_mut();
            //             unsafe { drop(data, 0, count) };
            //         }
            //     }
            // }
            // unsafe { dealloc(self as *mut _ as *mut _, self.layout) };
        }
    }

    struct Store {
        chunks: GrowVec<Arc<Chunk>>,
    }

    trait Datum {}
    trait Template {}

    struct Insert<'a, T: Template>(PhantomData<&'a T>);
    struct Remove<'a>(PhantomData<&'a ()>);

    impl Store {
        pub fn insert<T: Template>(&self) -> Insert<T> {
            todo!()
        }

        pub fn remove(&self) -> Remove {
            todo!()
        }
    }

    impl<T: Template> Insert<'_, T> {
        pub fn one(&mut self, template: T) -> Key {
            todo!()
        }

        pub fn all<I: IntoIterator<Item = T>, F: FromIterator<Key>>(&mut self, templates: I) -> F {
            todo!()
        }

        pub fn all_in<I: IntoIterator<Item = T>, E: Extend<Key>>(
            &mut self,
            keys: &mut E,
            templates: I,
        ) -> usize {
            todo!()
        }

        pub fn all_n<const N: usize>(&mut self, templates: [T; N]) -> [Key; N] {
            todo!()
        }
    }

    impl Remove<'_> {
        pub fn one(&mut self, key: Key) -> bool {
            todo!()
        }

        pub fn all_n<const N: usize>(&mut self, keys: [Key; N]) -> [bool; N] {
            todo!()
        }

        pub fn all<I: IntoIterator<Item = Key>>(&mut self, keys: I) -> usize {
            todo!()
        }
    }
}

// mod dynamic {
//     use super::*;
//     use parking_lot::RwLock;
//     use std::{collections::BTreeMap, string};

//     impl Database {
//         pub fn run(
//             &self,
//             operations: impl IntoIterator<Item = &Operation>,
//         ) -> Result<Value, Error> {
//             todo!()
//         }
//     }

//     pub enum Node {
//         Null,
//         Number(Number),
//         String(String),
//         Increment(Number),
//         Pair(Key, Box<Node>),
//         List(Vec<Node>),
//     }

//     pub enum Operation {
//         Create(Node),
//         Destroy(Key),
//         Modify(Key, Node),
//     }

//     pub enum String {
//         Borrow(&'static str),
//         Own(string::String),
//     }

//     pub enum Number {
//         Integer(i64),
//         Rational(f64),
//     }

//     pub enum Key {
//         Meta(&'static Meta),
//         Name(String),
//     }

//     impl Into<Node> for &'static str {
//         fn into(self) -> Node {
//             Node::String(String::from(self))
//         }
//     }

//     impl Into<Box<Node>> for &'static str {
//         fn into(self) -> Box<Node> {
//             Box::new(Node::from(self))
//         }
//     }

//     impl Into<String> for &'static str {
//         fn into(self) -> String {
//             String::Borrow(self)
//         }
//     }

//     impl Into<Key> for &'static str {
//         fn into(self) -> String {
//             Key::Name(String::from(self))
//         }
//     }

//     impl Into<Node> for i64 {
//         fn into(self) -> Node {
//             Node::Number(Number::Integer(self))
//         }
//     }

//     impl Into<Box<Node>> for i64 {
//         fn into(self) -> Node {
//             Box::new(Node::Number(Number::Integer(self)))
//         }
//     }

//     impl Into<Node> for f64 {
//         fn into(self) -> Node {
//             Node::Number(Number::Rational(self))
//         }
//     }

//     impl Into<Box<Node>> for f64 {
//         fn into(self) -> Node {
//             Box::new(Node::Number(Number::Rational(self)))
//         }
//     }

//     impl Into<Node> for Vec<Node> {
//         fn into(self) -> Node {
//             Node::List(self)
//         }
//     }

//     impl Into<Key> for &'static Meta {
//         fn into(self) -> Key {
//             Key::Meta(self)
//         }
//     }

//     fn pair(key: impl Into<Key>, value: impl Into<Node>) -> Node {
//         Node::Pair(key.into(), Box::new(value.into()))
//     }

//     fn create(node: impl Into<Node>) -> Operation {
//         Operation::Create(node.into())
//     }

//     fn destroy(key: key::Key) -> Operation {
//         Operation::Destroy(key)
//     }

//     fn modify(key: key::Key, node: impl Into<Node>) -> Operation {
//         Operation::Modify(key, node.into())
//     }

//     fn increment(node: impl Into<Number>) -> Node {
//         Node::Increment(node.into())
//     }

//     #[test]
//     fn database_run_create() {
//         use Node::*;

//         let database = Database::new();
//         database.run([
//             create(vec![
//                 pair(
//                     "Position",
//                     vec![pair("x", 0.0), pair("y", 1.0), pair("z", 2.0)],
//                 ),
//                 pair(
//                     velocity(),
//                     vec![pair("x", 0.0), pair("y", 1.0), pair("z", 2.0)],
//                 ),
//                 pair("Mass", vec![Node::from(0.0)]),
//             ]),
//             destroy(Key::NULL),
//             modify(Key::NULL, vec![pair("Position", pair("x",
// increment(5.0)))]),         ]);
//     }

//     fn position() -> &'static Meta {
//         dynamic(
//             "component::Position",
//             [field("x", r#static::<f64>), field("y", r#static::<f64>)],
//         )
//     }

//     fn velocity() -> &'static Meta {
//         dynamic(
//             "component::Velocity",
//             [field("x", r#static::<f64>), field("y", r#static::<f64>)],
//         )
//     }

//     struct Field {
//         offset: usize,
//         name: String,
//         meta: LazyLock<&'static Meta, Box<dyn FnOnce() -> &'static Meta +
// Send + Sync + 'static>>,     }

//     struct Meta {
//         index: usize,
//         name: String,
//         layout: Layout,
//         fields: Box<[Field]>,
//     }

//     enum Identifier {
//         Type(TypeId),
//         Index(usize),
//     }

//     impl Meta {
//         pub unsafe fn new(&self, capacity: usize) -> NonNull<u8> {
//             if self.layout.size() == 0 {
//                 self.layout.dangling().cast()
//             } else {
//                 let (layout, _) =
// self.layout.repeat(capacity).unwrap_unchecked();
// NonNull::new_unchecked(alloc(layout).cast())             }
//         }

//         pub unsafe fn copy(
//             &self,
//             source: (NonNull<u8>, usize),
//             target: (NonNull<u8>, usize),
//             count: NonZeroUsize,
//         ) {
//             if self.layout.size() > 0 {
//                 let source = source
//                     .0
//                     .as_ptr()
//                     .cast::<u8>()
//                     .add(source.1 * self.layout.size());
//                 let target = target
//                     .0
//                     .as_ptr()
//                     .cast::<u8>()
//                     .add(target.1 * self.layout.size());
//                 copy(source, target, count.get() * self.layout.size());
//             }
//         }

//         pub unsafe fn free(&self, data: NonNull<u8>, capacity: usize) {
//             let (layout, _) =
// self.layout.repeat(capacity).unwrap_unchecked();
// dealloc(data.as_ptr().cast(), layout)         }
//     }

//     fn field(
//         name: impl Into<String>,
//         meta: impl FnOnce() -> &'static Meta + Send + Sync + 'static,
//     ) -> Field {
//         Field {
//             offset: 0,
//             name: name.into(),
//             meta: LazyLock::new(Box::new(meta)),
//         }
//     }

//     static REGISTRY: RwLock<BTreeMap<String, &'static Meta>> =
// RwLock::new(BTreeMap::new());

//     fn r#static<T: 'static>() -> &'static Meta {
//         let name = type_name::<T>().to_string();
//         if let Some(meta) = REGISTRY.read().get(&name) {
//             return meta;
//         }

//         let mut registry = REGISTRY.write();
//         let index = registry.len();
//         registry.entry(name).or_insert_with_key(|key| {
//             Box::leak(Box::new(Meta {
//                 index,
//                 name: key.clone(),
//                 layout: Layout::new::<T>(),
//                 fields: Box::new([]),
//             }))
//         })
//     }

//     fn dynamic(name: impl Into<String>, fields: impl IntoIterator<Item =
// Field>) -> &'static Meta {         let name = name.into();
//         if let Some(meta) = REGISTRY.read().get(&name) {
//             return meta;
//         }

//         let mut registry = REGISTRY.write();
//         let index = registry.len();
//         registry.entry(name.into()).or_insert_with_key(|key| {
//             let mut layout = Layout::new::<()>();
//             let fields = fields
//                 .into_iter()
//                 .map(|mut field| {
//                     let pair = layout.extend(field.meta.layout).unwrap();
//                     layout = pair.0;
//                     field.offset = pair.1;
//                     field
//                 })
//                 .collect();
//             Box::leak(Box::new(Meta {
//                 index,
//                 name: key.clone(),
//                 layout: layout.pad_to_align(),
//                 fields,
//             }))
//         })
//     }
// }

// mod events_by_table {
//     use super::*;

//     /*
//         TODO: Have the events by stored by table.
//         - Listeners would pull events from their eligible tables (ex:
//           `OnAdd<T>` would only pull from tables where `table.has::<T>()`).
//         - PRO: Allows using `Filter` to pull only events from filtered
//           tables.
//         - PRO: Much more granular `create/destroy/modify` counters to turn
//           on/off event collection.
//         - PRO: `fold_swap` can be used to pull events rather than blocking on
//           a single mutex.
//         - CON: Adds a fair amount of complexity and many atomic operations
//           which may not be necessary if contention is not an issue in
//         the first place.
//         - Listen pulls will need to be very quick (probably a `swap`) such
//           that they hold the lock for a very short time.
//         - All table operations will not contend on the same mutex; in fact
//           they will only contend on listeners since only one table
//         operation can (currently) take place at a time (guaranteed by the
// upgradable lock held during these operations).
//         - Events will be ordered within a table.
//         - To get a global ordering of events, a global atomic counter will
//           need to be used.
//         - Tables will need to initialize their event collection counter with
//           `Events` when created.
//         - `Events` will need to maintain a `Vec<Box<dyn Fn(&Table)>>` which
//           initialize a newly created table's event counters.
//     */
//     struct TableEvents {
//         index: u32,
//         counts: (AtomicU64, AtomicU64, AtomicU64),
//         events: RwLock<(Vec<TableRaw>, Vec<Key>)>,
//     }
//     #[derive(Clone, Copy, Debug)]
//     struct TableRaw(usize, Keys, TableKind);
//     #[derive(Clone, Copy, Debug)]
//     enum TableKind {
//         Create,
//         Modify(u32),
//         Destroy,
//     }
//     fn from_table_to_key_history<'a>(
//         collect: &mut (Vec<TableRaw>, Vec<Key>),
//         indices: &mut [u32],
//         tables: &[TableEvents],
//     ) {
//         let start = collect.0.len();
//         fold_swap(
//             indices,
//             (),
//             &mut *collect,
//             |_, collect, index| {
//                 let index = *index as usize;
//                 let table = unsafe { get_unchecked(tables, index) };
//                 let Some(events) = table.events.try_read() else { return
// Err(()); };                 Ok(extend(collect, &events))
//             },
//             |_, collect, index| {
//                 let index = *index as usize;
//                 let table = unsafe { get_unchecked(tables, index) };
//                 let events = table.events.read();
//                 extend(collect, &events);
//             },
//         );
//         collect.0[start..].sort_by_key(|event| event.0);

//         #[inline]
//         const fn decompose(value: u64) -> (u32, u32) {
//             ((value >> 32) as u32, value as u32)
//         }

//         #[inline]
//         const fn compose(read: u32, key: u32) -> u64 {
//             ((read as u64) << 32) | (key as u64)
//         }

//         #[inline]
//         fn extend(collect: &mut (Vec<TableRaw>, Vec<Key>), events:
// &(Vec<TableRaw>, Vec<Key>)) {
// collect.0.extend(events.0.iter().map(|&(mut event)| {
// event.1.index = event.1.index.saturating_add(collect.1.len() as _);
//                 event
//             }));
//             collect.1.extend_from_slice(&events.1);
//         }

//         fn emit(
//             table: &TableEvents,
//             keys: &[Key],
//             kind: TableKind,
//             local: &AtomicU64,
//             global: &AtomicUsize,
//         ) {
//             let values = decompose(local.load(Ordering::Relaxed));
//             if values.0 > 0 {
//                 let order = global.fetch_add(1, Ordering::Relaxed);
//                 let mut events = table.events.write();
//                 if values.1 > 0 {
//                     let index = events.1.len();
//                     events.0.push(TableRaw(
//                         order,
//                         Keys {
//                             index: index as _,
//                             count: keys.len() as _,
//                         },
//                         kind,
//                     ));
//                     events.1.extend_from_slice(keys);
//                 } else {
//                     events.0.push(TableRaw(
//                         order,
//                         Keys {
//                             index: u32::MAX,
//                             count: keys.len() as _,
//                         },
//                         kind,
//                     ));
//                 }
//             }
//         }

//         fn get_keys<'a>(event: &TableRaw, keys: &'a [Key]) -> Option<&'a
// [Key]> {             keys.get(event.1.index as usize..(event.1.index +
// event.1.count) as usize)         }

//         fn history<'a>(
//             key: Key,
//             map: &'a HashMap<(Key, u32), u32>,
//         ) -> impl Iterator<Item = u32> + 'a {
//             let mut last = map.get(&(key, u32::MAX));
//             from_fn(move || {
//                 let &table = last?;
//                 last = map.get(&(key, table));
//                 Some(table)
//             })
//         }
//     }
// }

// mod locks {
//     use std::marker::PhantomData;

//     pub trait Datum {}
//     pub trait Item {}
//     impl<D: Datum> Item for &D {}
//     impl<D: Datum> Item for &mut D {}
//     impl Item for () {}
//     impl<I1: Item> Item for (I1,) {}
//     impl<I1: Item, I2: Item + Allow<I1>> Item for (I1, I2) {}
//     impl<I1: Item, I2: Item + Allow<I1>, I3: Item + Allow<I1> + Allow<I2>>
// Item for (I1, I2, I3) {}     impl Datum for bool {}
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
//     impl<T1, T2, T3, U: Allow<T1> + Allow<T2> + Allow<T3>> Allow<U> for (T1,
// T2, T3) {}

//     struct Query<I: Item>(PhantomData<I>);
//     impl<I: Item, U: Allow<I>> Allow<U> for Query<I> {}
//     impl<I: Item> Query<I> {
//         pub fn new() -> Self {
//             todo!()
//         }
//         pub fn each(&mut self, each: impl FnMut(I) + 'static) {}
//         pub fn each_with<S: Allow<I>>(&mut self, state: S, each: impl
// FnMut(I, S) + 'static) {}     }

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
//             Some(unsafe { get_unchecked(chunk, item as
// usize).assume_init_ref() })         }

//         pub fn push(&self, item: T) {
//             let (mut count, ended, begun) =
// decompose_count(self.count.fetch_add(1, AcqRel));             let index =
// count + ended as u32 + begun as u32;             let (mut old_count, _) =
// decompose_index(count);             let new_count = decompose_index(index);
//             self.increment_use(old_count);
//             let mut old_chunks = self.chunks.load(Acquire);

//             debug_assert_eq!(new_count.0 - old_count, 1);
//             if old_count < new_count.0 {
//                 // TODO: Re-read the count here? In a loop?
//                 let new_chunks = {
//                     let mut chunks = Vec::with_capacity(new_count.0 as
// usize);                     let new_chunks = chunks.as_mut_ptr();
//                     unsafe { copy_nonoverlapping(old_chunks, new_chunks,
// old_count as usize) };                     forget(chunks);
//                     new_chunks
//                 };

//                 match self
//                     .chunks
//                     .compare_exchange(old_chunks, new_chunks, AcqRel,
// Acquire)                 {
//                     Ok(chunks) => {
//                         let chunk = Box::new([(); CHUNK].map(|_|
// MaybeUninit::<T>::uninit()));                         unsafe {
// new_chunks.add(old_count as usize).write(chunk) };                         //
// It should be extremely unlikely that this call returns `true`.
// self.try_free(old_count, old_chunks);                         old_chunks =
// chunks;                     }
//                     Err(chunks) => {
//                         // Another thread won the race; free this allocation.
//                         drop(unsafe { Vec::from_raw_parts(new_chunks, 0,
// new_count.0 as usize) });                         old_chunks = chunks;
//                     }
//                 }
//                 (count, _, _) = decompose_count(self.count.load(Acquire));
//                 (old_count, _) = decompose_index(count);
//             }

//             let chunk = unsafe { &mut **old_chunks.add(new_count.0 as usize)
// };             self.decrement_use(old_count - 1);
//             let item = MaybeUninit::new(item);
//             unsafe { chunk.as_mut_ptr().add(new_count.1 as usize).write(item)
// };             let result = self.count.fetch_update(AcqRel, Acquire, |count|
// {                 let (count, ended, begun) = decompose_count(count);
//                 Some(if begun == 1 {
//                     recompose_count(count + ended as u32 + begun as u32, 0,
// 0) } else { debug_assert!(begun > 1); recompose_count(count, ended + 1, begun
//    - 1) }) }); debug_assert!(result.is_ok()); }

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
//         fn try_free(&self, count: u32, swap: *mut Box<[MaybeUninit<T>;
// CHUNK]>) -> bool {             let pending = self.pending.swap(swap, AcqRel);
//             if pending.is_null() {
//                 false
//             } else {
//                 drop(unsafe { Vec::from_raw_parts(pending, 0, count as usize)
// });                 true
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
