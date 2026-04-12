# Recommended Roadmap

This is the decision document.

If I were steering `that_bass` toward its stated goals, this is the path I would choose.

## The Short Answer

Recommended target architecture:

- chunked archetype storage,
- keyless storage primitives plus identity extensions such as ordinary `Key` columns, `Keys`, and later user keys,
- scheduler-backed frame execution,
- deferred structural commands with sync points,
- optional hot/cold and fine-grained storage classes,
- advanced layout specialization only after the basics are stable.

In other words:

- not "just improve the locks",
- not "fully flatten everything immediately",
- but "move the whole engine toward chunked, scheduled, lockless hot execution".

## Why This Is The Right Balance

This path keeps what is already excellent in `that_bass`:

- generational keys,
- archetype-style locality,
- direct trait-driven query ergonomics,
- explicit structural operations,
- a storage engine that still makes sense outside a full engine framework.

And it addresses the real ceilings:

- coarse parallel splits,
- hot-path lock cost,
- unnecessary identity bookkeeping for bag-like data,
- missing schedule semantics,
- repeated runtime planning,
- structural visibility complexity.

## Proposed Phases

## Phase 0: Correctness, Instrumentation, And Visibility

Before major redesign, strengthen observability.

Do first:

- fix the create-event emission quirk and expand event tests,
- add archetype/table/chunk generation counters,
- add profiling hooks around query update, lock wait, resolve cost, and event cost,
- add explicit documentation of visibility windows and frame semantics,
- add microbenchmarks for:
  - dense query scans,
  - keyed joins,
  - create/add/remove/destroy under contention,
  - one huge archetype vs many small archetypes.

Why first:

- major storage redesign without measurements is guesswork.

## Phase 1: Chunk The Tables

Turn today's table into:

- table metadata,
- many fixed-size chunks.

Do in this phase:

- chunk-level `split()`,
- chunk-local counts and key slices,
- chunk-level create/modify/destroy resolution,
- chunk-aware key slots,
- optional identity extensions:
  - ordinary `Key` columns plus the `Keys` resource,
  - later user-provided keys,
  - keyless rows as the default,
- benchmark different chunk sizes.

Expected result:

- immediate improvement in parallelism,
- better scalability for dominant archetypes,
- less synchronization and memory overhead for rows with no external identity,
- shorter lock hold times even before scheduler integration.

## Phase 2: Persistent Query Plans

Once archetypes are chunked, cache plans aggressively.

Add:

- query plan cache keyed by row/filter/access shape,
- archetype and chunk freshness tracking,
- precomputed column projection and lock plans,
- chunk iteration order plans,
- optional keyed-join cache structures,
- a proper query algebra for nested cases:
  - lookup,
  - self-join,
  - cartesian product,
  - exclude-self,
  - pairwise nested execution plans.

Also decide in this phase how plans differ for:

- keyed tables with direct lookup,
- user-keyed tables with indexed lookup,
- keyless tables that only support iteration or explicit value-based joins.

Expected result:

- repeated frame queries become much cheaper,
- system-like execution starts to look natural even before a full scheduler.

## Phase 3: Scheduler Layer

Build a separate scheduling layer on top of the chunked storage core.

Add:

- systems,
- resources in the access graph,
- command buffers,
- phase execution,
- dependency graph,
- chunk jobization,
- sync points,
- an answer for nested conflicting queries:
  - planned relational operators where possible,
  - nest-aware reborrow execution for direct mode,
  - scheduler diagnostics when a requested nest cannot be made live and sound.

This should likely ship as a sibling crate or module boundary, not a full rewrite of the storage core.

Expected result:

- lockless hot iteration in scheduled mode,
- clearer user story for games,
- better frame predictability.

## Phase 4: Hot/Cold Storage Classes

After chunking plus scheduling, introduce differentiated storage.

Add:

- dense hot chunked columns,
- sparse side storage for volatile/rare data,
- cold/blob sidecar storage for big heap-rich datums,
- maybe opt-in split datums for POD-heavy structs.

Expected result:

- less structural movement cost,
- less cache pollution,
- much better fit for real game data.

## Phase 5: Advanced Layout Specialization

Only now should the project explore:

- owning-group style hot packs,
- profile-guided layout suggestions,
- field-level expert query APIs,
- compressed vector formats,
- GPU mirrors,
- transaction log / rollback integration.

Why so late:

- these features depend on stable execution and storage primitives.

## What I Would Not Do First

## Do not start with full field flattening

It is tempting, but it multiplies complexity before the bigger wins are secured.

## Do not chase lock cleverness as the main strategy

Better RW locks, sharded locks, or hand-tuned locking protocols can help, but they are not the real endgame for games.

## Do not force everything into a framework immediately

The current direct-database identity is a strength.
Preserve it as the low-level mode.

## API Recommendation

The project should become two-tiered.

## Tier 1: Direct API

Like today:

- explicit `create`, `query`, `add`, `remove`, `destroy`,
- plus future identity extensions where relevant,
- direct access from tools/tests/one-off code,
- flexible, lower ceremony.

## Tier 2: Scheduled API

New:

- systems,
- schedules,
- phases,
- resources,
- commands,
- lockless scheduled iteration.

This lets the library serve both:

- engine code that needs peak throughput,
- and users that want a dynamic state database.

## The Best Version Of `that_bass`

The best future version of the library would feel like this:

- simple systems and queries for users,
- chunked SoA storage underneath,
- rows only pay for identity when identity is actually needed,
- scheduler handles parallelism,
- structural operations are command-buffered,
- most hot iteration is lockless,
- layout specialization exists, but only when the user asks for it or the profiler strongly suggests it.

That would give the project a very strong identity:

- more locality-driven than sparse-set ECS,
- more usable as a standalone crate than a full engine runtime,
- more game-optimized than a generic in-memory database,
- and more scalable than the current lock-centric execution path.

## Final Conclusion

If I had to choose one slogan for the future architecture, it would be:

> Chunked archetypes plus scheduled readonly execution.

That is the redesign with the best combination of:

- performance ceiling,
- API sanity,
- multicore scalability,
- and strategic coherence with what the codebase already wants to be.

The selected direction is now expanded into a concrete implementation plan under `future/plan/`.
