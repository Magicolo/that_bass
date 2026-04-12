# Chunking, Locks, And Query Plans

This document covers the best path forward if `that_bass` wants to keep a direct database API but push much harder on parallelism and locality.

## Core Thesis

The single most useful storage-level redesign is:

- keep archetype/table storage,
- but split each table into fixed-size chunks,
- make queries and structural operations operate on chunk plans rather than whole-table plans.

This is the pure-storage redesign with the best payoff.

## Why The Current Table Model Hits A Ceiling

Today the main unit of coordination is effectively the whole table:

- query splits are table-level,
- add/remove/destroy coordinate around table key locks,
- lock hold times scale with table size,
- large hot tables become contention hotspots,
- create/modify/destroy all compete on the same table-level control points.

Even with sorted column locking and deferred resolution, this limits parallelism when one schema dominates the frame.

That is exactly what happens in many real games:

- one giant movement archetype,
- one giant render archetype,
- one giant physics-contact archetype.

## Proposal: Archetype Chunks

Each logical table becomes:

- table metadata,
- many chunks,
- each chunk holding a fixed number of rows,
- per-chunk keys and per-chunk column slices.

Suggested chunk sizes to explore:

- 64 rows for very dynamic/high-contention workloads,
- 128 or 256 rows as a good general default,
- 512 rows for very dense numeric kernels.

Do not start at 1024+ without profiling; that begins drifting back toward table-level contention.

## What Changes

### Before

- `Table` owns one key array and one array per column.

### After

- `Archetype` owns schema-level metadata.
- `Chunk` owns:
  - key slice,
  - count,
  - capacity,
  - one data slice per column,
  - lock/version state.

This mirrors both:

- chunk ECS designs used in game engines,
- vectorized `DataChunk` execution in column engines like DuckDB.

## Benefits

## 1. Much More Parallelism

The scheduler or query engine can split work across chunks of the same archetype.

Instead of:

- one giant `(&mut Position, &Velocity)` split for the entire archetype,

you get:

- dozens or hundreds of chunk jobs.

This is far better for:

- work stealing,
- short task scheduling,
- avoiding one giant job monopolizing worker threads.

## 2. Better Lock Granularity

If the library stays lock-based for direct use, locks become chunk-local.

That means:

- shorter lock hold times,
- more concurrent creates/modifies/destroys on the same schema,
- lower blast radius for hot archetypes.

## 3. Better Structural Mutation Locality

Create/add/remove/destroy can often resolve within one or a few chunks:

- appending into partially filled chunks,
- compacting within one chunk,
- migrating only a handful of chunk pairs.

This is much cheaper than whole-table compaction.

## 4. Better Execution-Plan Caching

A query can cache:

- matching archetypes,
- lock plans,
- column projections,
- chunk iterators.

Then each frame or update only needs to refresh:

- chunk list freshness,
- counts/versions,
- maybe filter-specific membership.

## Chunk Model Variants

## Variant A: Fixed-Size Chunks

Every chunk has the same capacity.

### Good

- Very simple.
- Good for execution planning and vectorization.
- Key encoding is straightforward.

### Bad

- Some wasted space.
- Potentially more chunks than ideal.

### Verdict

Best starting point.

## Variant B: Small-Then-Grow Chunks

Early chunks grow geometrically until a stable size.

### Good

- Better behavior for tiny tables.

### Bad

- More complexity in the planner and allocator.

### Verdict

Potential later optimization, not a first design.

## Variant C: Fragment Lists

Keep tables conceptually large, but lock/iterate fragments.

### Good

- More continuity with the current code.

### Bad

- Less clean than real chunk ownership.
- Harder to reason about long-term.

### Verdict

Interesting transitional design, but inferior to full chunk ownership.

## Lock Strategy Options

## Option 1: Chunk-Key Lock + Chunk-Column Locks

This is the closest evolution of today's model.

### Good

- Easiest migration.
- Reuses many current invariants.

### Bad

- Runtime locking cost still exists.
- Queries over many chunks still pay per-chunk lock overhead.

## Option 2: Chunk Versioning + Copy-On-Write Structural Control

Readers take optimistic snapshots based on chunk versions.
Writers replace chunk metadata or reserve new chunks.

### Good

- Lower read overhead.

### Bad

- Significantly more complex.
- Harder to make row references safe.

## Option 3: Readonly Frame + Command Buffers

No hot-path locks during scheduled execution; only sync points and merges.

### Good

- Best long-term answer.

### Bad

- Requires scheduler integration.

### Verdict

If the project remains storage-only, use option 1 first.
If the project embraces a scheduler, jump to option 3.

## Query Plan Caching

The next major missing piece is persistent query plans.

## What to cache

Per query/filter shape:

- access set,
- matching archetypes,
- required column indices,
- chunk iteration order,
- maybe keyed-join helpers,
- maybe SIMD-friendly execution traits.

## Invalidating plans

Use generations:

- archetype schema generation,
- chunk list generation per archetype,
- maybe component layout generation for hot/cold changes.

This is similar in spirit to:

- hecs archetype freshness tracking,
- Flecs cached queries.

## Why it matters

Right now, the library is very dynamic, which is nice, but repeated queries still spend effort rediscovering table state.

Games run the same systems every frame.
The engine should behave like it knows that.

## Key Encoding In A Chunk World

The current key is:

- `index`
- `generation`

and slot tells you:

- table
- row

Possible chunk-world forms:

### Option A: Keep slot indirection

Key stays small and stable.
Slot now points to:

- archetype id,
- chunk id,
- row id.

### Option B: Encode chunk directly into key

Reduces one layer of indirection for some paths.

### Verdict

Keep the slot indirection first.
It preserves flexibility for compaction and chunk movement.

## Bring In Vector Concepts From DuckDB

DuckDB's execution model suggests a few good ideas:

- fixed-size execution vectors,
- multiple physical formats for logically columnar data,
- chunk-oriented operators.

Directly useful adaptations:

- constant chunks for repeated/default values,
- dictionary-like encoding for low-cardinality tags or small enums,
- sequence vectors for row ids or generated data,
- unified chunk iteration kernels.

This matters because game data often has:

- repeated default values,
- many identical tags/flags,
- stable IDs and counters,
- large scans over simple numeric columns.

## Recommended Pure-Storage Roadmap

If the project wants maximum gains before a scheduler rewrite:

1. Replace table-wide execution with fixed-size chunks.
2. Make `split()` chunk-based, not table-based.
3. Add persistent query plans keyed by row/filter shape.
4. Add archetype/chunk generations to invalidate plans cheaply.
5. Keep direct API and lock-based execution, but reduce lock scope to chunks.
6. Add optional chunk-local parallel iterators.

## Strong Conclusion

If the project wants a storage-centric redesign that still preserves the current database feel, chunked archetypes are the best next move.

This is the design that:

- improves parallelism immediately,
- keeps locality high,
- fits the current mental model,
- and composes naturally with a later scheduler-first lockless mode.
