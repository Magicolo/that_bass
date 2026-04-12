# Competitive Landscape

This file compares `that_bass` with adjacent systems and extracts design lessons.

I focused on systems that matter for the target problem:

- ECS engines used in games,
- schedulers that derive safety from declared access,
- data-oriented runtimes that minimize lock contention,
- vectorized engines that show how to structure work in fixed-size chunks.

## Current `that_bass` In One Sentence

`that_bass` today is a lock-coordinated archetype database with generational keys, deferred structural edits, and a lightweight query API.

That puts it in an unusual spot:

- closer to an archetype ECS than to a generic in-memory database,
- more dynamic and direct than scheduler-first runtimes,
- more concurrency-oriented than minimalist single-world ECS crates,
- but not yet as lock-free or schedule-driven as the fastest frame-oriented solutions.

## Comparison Matrix

| System | Storage model | Parallelism model | Structural change model | Main lesson |
| --- | --- | --- | --- | --- |
| `that_bass` | table/archetype SoA | runtime locks + deferred ops | explicit `resolve` | strong locality, but hot path still pays for coordination |
| Bevy ECS | tables + sparse sets | scheduler from system params | commands/deferred world ops | make access declarations first-class and drive parallelism from them |
| Legion | packed archetypes | automatic system scheduling | batch-oriented archetype movement | packed storage plus scheduler is a good baseline |
| Shipyard | sparse set default | workloads + parallel iterators | borrow-based scheduling | user-facing simplicity can coexist with aggressive schedule analysis |
| hecs | minimalist archetype ECS | little built-in scheduling | direct world access, command buffer available | keep the API tiny; let scheduler live outside if needed |
| yaks | scheduler on top of hecs | auto multithreading via executor | externalized from world | separating storage from scheduler is viable |
| Flecs | table ECS + cached queries + staging | pipelines + per-thread command queues | readonly stage + sync points | staging and sync-point insertion are the most directly useful ideas for `that_bass` |
| EnTT | sparse sets + owning groups | mostly user-managed | no full thread-safe registry by default | explicit hot-path groups are a powerful opt-in layout tool |
| Unity Jobs/Entities | chunk ECS + job graph | dependency-driven jobs | command buffers + job handles | lockless frame execution wins if the scheduler owns access |
| DuckDB | vector/chunk column engine | operator pipelines | immutable-ish chunks during execution | fixed-size chunks and vector formats can improve both locality and parallel work distribution |

## System By System

## Bevy ECS

Relevant references:

- https://docs.rs/bevy_ecs/latest/bevy_ecs/

Key traits:

- Bevy explicitly sells itself as simple, ergonomic, fast, and massively parallel.
- System function signatures declare data access.
- The schedule uses that access information to run systems in parallel when safe.
- It supports both table storage and sparse-set storage.

Why it matters:

- This is the clearest example of the API shape many Rust game developers now expect.
- It treats scheduling as a feature of the core model, not a separate afterthought.
- It gives users a very light API while still extracting access metadata from types.

Pros relative to `that_bass`:

- Much stronger story for frame execution.
- Easier path to lockless hot loops.
- Hybrid storage classes avoid forcing all components into one locality/fragmentation tradeoff.

Cons relative to `that_bass`:

- More framework-like.
- Less natural for ad hoc "database-style" direct access from arbitrary threads.
- Command/deferred semantics are simpler for games, but not always simpler for generic use.

Takeaway:

- If `that_bass` wants to stay game-first, it should probably learn from Bevy's "system params imply access graph" model.

## Legion

Relevant references:

- https://docs.rs/legion/latest/legion/

Key traits:

- Marketed as a feature-rich, high-performance ECS with minimal boilerplate.
- Uses packed archetype storage.
- Exposes automatic query scheduling and parallel execution.

Why it matters:

- It is very close to the storage instincts already visible in `that_bass`.
- It validates the combination of packed archetypes plus scheduler-driven system execution.

Pros relative to `that_bass`:

- More complete scheduler story.
- Stronger batch/archetype orientation.

Cons relative to `that_bass`:

- Less interesting around dynamic locking questions, because it mostly solves the problem by moving work into scheduled execution.

Takeaway:

- Archetype-locality plus auto-scheduling is not speculative; it is a proven direction.

## Shipyard

Relevant references:

- https://docs.rs/shipyard/latest/shipyard/
- https://leudz.github.io/shipyard/guide/master/going-further/parallelism.html

Key traits:

- Default storage is sparse-set based.
- Workloads store systems and their scheduling.
- The guide makes a clean distinction between outer parallelism and inner parallelism.
- Workloads analyze conflicts and build batches of non-conflicting systems.

Why it matters:

- Shipyard shows how much ergonomics can come from a borrow-driven API.
- It also shows that a scheduler can surface batch structure to users without making them think in lock terms.

Pros relative to `that_bass`:

- Easier scheduler model for users.
- Nice split between system-level parallelism and parallel iterators.
- Simpler reasoning about conflicting accesses.

Cons relative to `that_bass`:

- Sparse-set default is typically weaker for dense multi-component scans than packed archetype storage.
- Less aligned with the "maximize locality first" goal.

Takeaway:

- The workload model is worth borrowing.
- The default sparse-set storage is not the right end state for `that_bass`, but sparse side-stores may still be useful for cold/volatile components.

## hecs

Relevant references:

- https://docs.rs/hecs/latest/hecs/
- https://docs.rs/yaks/latest/yaks/

Key traits:

- hecs is proudly minimalist.
- It prioritizes fast traversals and a simple interface.
- It exposes archetypes, columns, archetype freshness generations, batched iteration, and command buffers.
- yaks layers automatic multithreading on top of hecs rather than fusing scheduler and storage together.

Why it matters:

- This is the cleanest proof that "storage library" and "scheduler" can be separate products.
- It is a strong model if `that_bass` wants to remain usable outside an engine loop.

Pros relative to `that_bass`:

- Cleaner conceptual boundary.
- Easier to keep the core library lightweight.

Cons relative to `that_bass`:

- If the scheduler stays external, the fastest path may not become the default path.
- Users may fail to compose it into a maximally efficient frame model.

Takeaway:

- Consider a two-layer architecture:
  - `that_bass_core`: storage and query engine
  - `that_bass_schedule`: frame scheduler and command buffers

## Flecs

Relevant references:

- https://www.flecs.dev/flecs/md_docs_2Queries.html
- https://www.flecs.dev/flecs/md_docs_2Systems.html
- https://www.flecs.dev/flecs/md_docs_2DesignWithFlecs.html

Key traits:

- Queries can be cached, uncached, or mixed.
- Systems run in a readonly stage where structural operations become commands.
- Each thread gets its own command queue.
- Sync points are inserted when queued writes must become visible before later reads.
- Systems can annotate extra reads/writes not obvious from the matched query.
- Immediate systems exist, but they lose some parallel properties.

Why it matters:

- This is the closest conceptual match to what `that_bass` could become.
- It directly attacks the same issue that `that_bass` currently handles with locks: structural mutation during iteration.

Pros relative to `that_bass`:

- Stronger lockless multithreading story.
- Better query caching story.
- Better distinction between "hot iteration" and "queued structural change".
- Automatic sync point insertion is a major usability win.

Cons relative to `that_bass`:

- The semantics become frame/pipeline oriented.
- Deferred visibility rules are more subtle.
- Scheduler awareness has to be deeply integrated.

Takeaway:

- If only one external system heavily informs the future of `that_bass`, it should be Flecs.

## EnTT

Relevant references:

- https://skypjack.github.io/entt/md_docs_2md_2entity.html

Key traits:

- Full-owning groups are extremely fast for multi-component iteration.
- They effectively arrange pools so multiple component arrays are co-packed and identically ordered.
- More performance comes with more constraints.
- The registry is not fully thread-safe by default, and the docs explicitly argue that performance can justify that.

Why it matters:

- EnTT is the strongest argument for explicit opt-in hot query layouts.
- It also validates the idea that not every registry operation needs to be thread-safe if the scheduler/runtime ensures correctness.

Pros relative to `that_bass`:

- Hot loops can be faster than generic queries.
- The tradeoff is explicit rather than hidden.

Cons relative to `that_bass`:

- More special cases in the data model.
- Harder to keep API and semantics uniform.

Takeaway:

- `that_bass` should consider "hot query packs" or "owning groups" as an advanced layout feature.

## Unity Jobs / Entities

Relevant references:

- https://docs.unity3d.com/2022.3/Documentation/Manual/JobSystemCreatingJobs.html

Key traits:

- Work is described as jobs with dependencies.
- Scheduled job data is isolated so multiple threads do not mutate the same data unsafely.
- Long jobs can starve the job graph; breaking work into smaller pieces matters.

Why it matters:

- It captures the execution discipline that high-end games tend to converge on.
- The important lesson is not Unity's API details, but the bias toward dependency graphs, command buffers, and phase-local readonly access.

Takeaway:

- A `that_bass` scheduler should think in terms of many short chunk jobs, not a few giant table jobs.

## DuckDB

Relevant references:

- https://duckdb.org/docs/lts/internals/vector

Key traits:

- Query execution is vectorized.
- Operators work on fixed-size vectors.
- `DataChunk` is a group of column vectors.
- Multiple vector formats exist: flat, constant, dictionary, sequence.

Why it matters:

- This is not an ECS, but it is a very relevant data-oriented engine.
- It shows how fixed-size chunks improve both CPU efficiency and work partitioning.
- It also shows that one stored type does not need a single fixed column representation forever.

Takeaway:

- `that_bass` should seriously consider fixed-size chunk execution and specialized vector formats for repeated/default data.

## Lessons For `that_bass`

### What to copy

- Bevy: access-derived schedule.
- Shipyard: batch reporting and inner vs outer parallelism vocabulary.
- hecs/yaks: clean split between world and scheduler.
- Flecs: staging, per-thread command queues, sync points, cached queries.
- EnTT: explicit hot query packs / owning groups.
- DuckDB: fixed execution chunks and multiple vector representations.

### What not to copy blindly

- Shipyard's sparse-set default. It is not aligned with `that_bass`'s locality-first goal.
- Full framework capture of the whole app. `that_bass` should still be usable as a storage engine.
- Immediate visibility everywhere. That tends to drag locks back into the hot path.

## Bottom-Line Conclusion

The strongest future for `that_bass` is not "be a better lock-based database".

It is:

- archetype-local storage,
- chunk-sized execution units,
- cached query/system plans,
- readonly frame execution,
- deferred structural commands,
- scheduler-inserted sync points,
- optional hot-path layout specialization.

That would place it in a unique and valuable spot:

- as direct and lightweight as hecs in spirit,
- as schedulable as Bevy or Shipyard,
- as structurally disciplined as Flecs,
- and as locality-conscious as packed-archetype ECS plus vectorized engines.
