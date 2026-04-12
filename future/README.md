# Future Proposals

This folder contains ambitious design proposals for where `that_bass` could go if the project fully optimized for realtime state management on modern multicore hardware.

These are not patch-sized ideas. Most of them imply incompatible APIs, storage rewrites, or a scheduler/runtime layer on top of the current database.

## Design Lens

I evaluated future directions against five criteria:

1. Frame-time predictability.
2. CPU cache locality.
3. Parallelism scaling as core count increases.
4. API simplicity for gameplay and simulation code.
5. How much runtime locking and structural coordination remains in the hot path.

The current crate is already strong on:

- archetype-style schema locality,
- explicit deferred structural operations,
- lightweight query syntax,
- generational keys,
- the ability to use the database directly without a framework.

The current crate is weaker on:

- runtime lock overhead in the query/resolve path,
- coarse table-level parallelism,
- repeated query discovery and lock planning,
- structural sync semantics for frame-oriented games,
- making the "fast path" obvious to users.

## Files

- `01-competitive-landscape.md`
  - Similar systems and what they get right.
- `02-field-granularity-and-layout.md`
  - More granular columns, hot/cold layouts, and query-driven packing.
- `03-chunking-locks-and-query-plans.md`
  - Chunked archetypes, smaller locking domains, and cached execution plans.
- `04-scheduler-first-lockless-mode.md`
  - Turning `that_bass` into a scheduler-backed lockless runtime for frame execution.
- `05-radical-ideas.md`
  - Bigger bets: self-tuning layouts, GPU mirrors, MVCC, and more.
- `06-recommended-roadmap.md`
  - What I would actually do, in order.
- `07-keyless-and-user-keyed-tables.md`
  - Making row identity optional instead of mandatory.
- `08-nested-query-soundness.md`
  - Making overlapping nested queries sound and deadlock-free.
- `plan/`
  - The concrete rewrite specification and ordered implementation tasks selected from these proposals.

## Executive Summary

The strongest long-term direction is not "slightly better locking". It is:

- keep an archetype/table mental model,
- split storage into fixed-size chunks,
- make row identity optional: no key by default, ordinary `Key` columns when stable identity is needed, and later user keys,
- add a scheduler/runtime that turns most frame execution into readonly iteration plus deferred command buffers,
- make queries and systems cacheable and schedulable,
- optionally add finer-grained field-level storage for hot POD-style data.

That is the path most aligned with the library goal: a simple, high-level API that still drives hardware hard.

## The Big Fork In The Road

There are really three plausible futures:

### Future A: Keep the current database model, but sharpen it

This means:

- query plan caching,
- chunked tables,
- optional keyless and user-keyed tables,
- better lock granularity,
- optional hot/cold storage classes,
- better event/filtering behavior,
- more efficient keyed joins.

This is the lowest-risk path. It preserves the current "database you can call from anywhere" feel.

### Future B: Build a scheduler-first runtime on top of the database

This means:

- systems become first-class,
- access declarations drive scheduling,
- the frame runs mostly lock-free,
- structural edits go through command buffers and sync points,
- direct ad hoc access still exists as an escape hatch.

This is the best fit for videogames and hard realtime simulation. It should be considered the default recommendation.

### Future C: Fully re-architect around generated data layouts

This means:

- field-level or leaf-level columns,
- generated query kernels,
- profile-guided hot group packing,
- perhaps even dual CPU/GPU layouts or self-tuning storage.

This has the highest upside, but also the highest chance of producing a brilliant but too-complex system.

## My Conclusion

If the goal is "simple API, extreme performance, high core utilization", the right answer is:

- adopt chunked archetypes,
- make identity optional so hot bag-like data stops paying the key tax,
- add scheduler-backed lockless frame execution,
- then selectively introduce finer-grained field storage where it helps.

Do not start with full leaf-column decomposition. Start with execution model and chunk topology first.
