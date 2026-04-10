# Rewrite Specification

This document is the concrete plan for the next major architecture of `that_bass`.

It is not a description of the current crate. The current crate is documented in `AGENTS.md` and in the existing `future/*.md` proposal set. This file selects a direction from those proposals, records the decisions already taken, and names the remaining open questions that must stay visible during the rewrite.

This document is architectural. Implementation standards live in `future/plan/standards.md` and are equally mandatory for rewrite work.

Primary references:

- `AGENTS.md`
- `future/plan/standards.md`
- `future/03-chunking-locks-and-query-plans.md`
- `future/04-scheduler-first-lockless-mode.md`
- `future/07-keyless-and-user-keyed-tables.md`
- `future/08-nested-query-soundness.md`
- Current implementation reference points:
  - `src/lib.rs`
  - `src/table.rs`
  - `src/query.rs`
  - `src/modify.rs`
  - `src/destroy.rs`
  - `src/key.rs`

## Status

This is a rewrite plan, not a promise that all details are final or already implemented.

The selected direction is:

- scheduler-owned safety instead of hot-path runtime locking,
- chunked table storage with dense packed rows,
- query execution over chunk slices, not per-row yielded references,
- keyless tables by default,
- optional managed keys when stable identity is required,
- single-allocation chunks,
- same-frame visibility through scheduled resolve jobs and happens-before edges,
- conservative rejection of overlapping conflicting live queries unless disjointness can be proven.

## Goals

The rewrite exists to optimize for these goals:

1. Maximize useful parallelism for realtime workloads.
2. Maximize CPU cache locality.
3. Keep the API light, obvious, and regular.
4. Make the fast path the default path.
5. Move coordination work out of the hot execution path and into schedule construction.
6. Let users choose whether rows pay the cost of stable identity.
7. Support applications that push hardware hard, especially games and simulations.

## Non-Goals

The rewrite is not trying to become:

- a transactional database,
- a deterministic-by-default executor,
- a full relational engine in the MVP,
- a leaf-column decomposition engine on day one,
- a scheduler that hides all data movement semantics from users,
- a framework that removes all direct access escape hatches.

## Execution Modes

The rewrite has two sanctioned mutation modes.

### Scheduled Mode

This is the primary fast path.

- shared world access is allowed,
- safety comes from declared access and scheduler ordering,
- hot iteration is chunk-oriented,
- structural edits are deferred through command buffers and resolve jobs.

### Direct Exclusive Mode

This is the escape hatch.

- immediate mutation is allowed only through exclusive mutable access to the storage,
- this mode is primarily for tools, tests, bootstrap code, import/export, debugging, and one-off integrations,
- it is not the performance path the architecture is optimizing for.

Hard rule:

- mutating access must either go through the scheduler,
- or require an exclusive mutable reference to the storage core.

No third mutation mode should exist in the rewrite.

## Selected Architecture In One Page

The next `that_bass` should be treated as a storage engine plus scheduler.

Core idea:

- users register scheduled functions,
- each function declares its query and command needs,
- the scheduler expands those functions into per-chunk jobs,
- those jobs are independently stealable,
- the scheduler honors declared happens-before relationships,
- structural edits are recorded into per-job command buffers,
- resolve work collects and batches those per-job command buffers at the function level before applying them and making them visible to later dependent jobs,
- tables are chunked, dense, and keyless by default,
- stable identity is opt-in through managed keys,
- direct immediate mutation remains available only behind exclusive mutable access.

This combines the strongest parts of the earlier proposals:

- chunked storage from `future/03-chunking-locks-and-query-plans.md`,
- scheduler-first execution from `future/04-scheduler-first-lockless-mode.md`,
- optional identity from `future/07-keyless-and-user-keyed-tables.md`,
- conservative soundness rules from `future/08-nested-query-soundness.md`.

## Why This Combination Was Chosen

Each selected idea fixes a different bottleneck:

- chunking improves locality and raises the parallelism ceiling,
- scheduler-owned access safety removes lock-order complexity from the hot path,
- keyless-by-default tables remove unnecessary identity maintenance,
- chunk-slice queries give a simple API that also enables SIMD-friendly kernels,
- per-job command buffers preserve parallel writes without shared hot-path synchronization,
- explicit resolve jobs make visibility boundaries honest.

These ideas also reinforce each other:

- chunking gives the scheduler small enough units of work to distribute well,
- scheduler control makes chunk-level writes safe without locking each query live,
- keyless tables make dense chunk moves cheap,
- batch command resolution fits chunk-local storage naturally.

## Rejected Or Deferred Alternatives

### Rejected As Primary Architecture

- Better runtime locking as the main answer.
  - Reason: it preserves the hardest part of the current design in the hottest path.
- Treating singletons/resources specially in the first rewrite.
  - Reason: the goal is one uniform resource model first, optimization later.
- Deterministic execution as a default property.
  - Reason: it would constrain scheduler freedom without matching the target workloads.
- Hard-capping chunk length at `<= 256`.
  - Reason: byte-budgeted chunk sizing is a better control knob; row/chunk bit partitioning can be table-specific.
- Bundling chunk work into non-stealable opaque worksets.
  - Reason: affinity is useful, but jobs must remain individually stealable.

### Deferred Beyond MVP

- Full relational/nested query algebra.
- `Families` resource.
- Specialized singleton fast paths.
- Field decomposition beyond a carefully limited first design.
- Scratch-chunk transactional writes as the default write model.
- User-keyed tables.
- Event-system parity with the current crate.

## Terminology

### Logical Schema

The set of logical datum types that define a table.

Example:

```rust
(Position, Velocity, Mass)
```

### Physical Column

The actual stored column unit. Usually one logical datum maps to one physical column. Future decomposition may allow one logical datum to map to several physical columns.

### Table

The schema-owned storage container for one logical schema. A table owns metadata and a set of chunks.

### Chunk

A densely packed, independently allocated subset of one table's rows. A chunk has:

- a capacity,
- a current count,
- one storage region per physical column,
- table-local metadata needed for row packing and scheduling.

### Resource

The smallest scheduler-visible access unit.

In the selected design, the resource granularity floor is:

- one physical column,
- of one chunk,
- of one table.

### Job

A runtime work unit scheduled by the executor. In the selected design, functions expand into per-chunk jobs. Jobs are independently stealable.

### Resolve Job

A scheduled batched application step that resolves deferred structural operations recorded by earlier jobs, typically for one function's collected command-buffer output rather than one originating chunk job at a time.

### Happens-Before

An ordering guarantee between jobs or resolve jobs. The system is intentionally non-deterministic outside declared happens-before edges.

### Row

An ephemeral row locator valid for the current job epoch only. It is not stable identity.

### Key

An opt-in managed stable identity for keyed tables.

## Storage Model

## Tables

The rewrite keeps table-based organization, but the storage core changes from "one giant column array per table" to "many independently allocated dense chunks per table".

The first rewrite should keep a strong distinction between:

- logical schema metadata,
- table runtime state,
- chunk memory,
- optional identity side data.

Tables are keyless by default.

## Chunk Capacity

Chunk capacity is derived from a configurable target byte budget and the table's inline physical row width.

Definitions:

- `inline_row_bytes = sum(size_of(all inline physical columns for one row))`
- `target_chunk_bytes = tunable configuration value, intentionally easy to benchmark`
- `raw_target_rows = max(1, floor(target_chunk_bytes / max(1, inline_row_bytes)))`
- `target_chunk_rows = previous_power_of_two(raw_target_rows)`, with a minimum of `1`

Consequences:

- chunk capacity is per-table,
- capacity remains a power of two,
- tiny row types may produce large capacities,
- huge row types may produce capacity `1`,
- the byte target can be changed centrally for benchmarking without rewriting storage code.

Important note:

- the default target byte count is intentionally not fixed in this specification,
- plausible starting points are in the `16 KiB` to `32 KiB` range,
- benchmarks must determine the default.

## Chunk Growth Policy

The rewrite should start small and avoid wasting memory.

Selected bootstrap policy:

1. the first chunk for a table starts with capacity `1`,
2. when growth is needed, capacity doubles,
3. once the table reaches its derived `target_chunk_rows`, all new chunks are allocated at that full target capacity.

This keeps tiny tables cheap while converging to a stable execution grain.

## Chunk Allocation Layout

Each chunk should use one allocation, not one allocation per column.

The allocation layout should be:

- chunk header and small metadata,
- then one aligned region per physical column,
- with precomputed column offsets,
- with all row storage densely packed from `0..count`.

Benefits:

- fewer allocator calls,
- better page locality across correlated columns,
- simpler ownership and recycling,
- easier whole-chunk movement or scratch strategies later.

## Dense Packing And Removal

Every chunk is always densely packed.

Removal semantics:

- row removal conceptually performs `swap_remove`,
- the removed row's slot is filled with the current tail row of the chunk,
- table row order is explicitly unstable by design.

This is not an optimization detail. It is a core invariant.

## Query Model

## Resource Granularity Floor

The planner and scheduler reason about resources at the following minimum scope:

- physical column x chunk x table.

This is deliberate.

It means:

- `Position` and `Velocity` in the same chunk are different resources,
- future subcolumn decomposition can become separate resources,
- one hot chunk does not force whole-table serialization.

## Chunk-Only Query Outputs

Queries do not yield individual rows directly.

They yield chunk projections such as:

- `&[Position]`
- `&mut [Position]`
- structured views made of multiple chunk slices

Row-level iteration is a layer on top of chunk results.

Example direction:

```rust
schedule.push(
    query::all((query::write::<Position>(), query::read::<Velocity>())),
    |positions, velocities| {
        for (position, velocity) in positions.zip(velocities) {
            position.x += velocity.x;
            position.y += velocity.y;
        }
    },
);
```

The important semantic choice is:

- the scheduler reasons about chunk resources,
- the user receives chunk slices,
- fine-grained per-row iteration is a local adapter, not the scheduler's native unit.

## Query Declaration

Queries are declared ahead of execution.

Required MVP combinators:

- `query::read::<T>()`
- `query::write::<T>()`
- `query::all(...)`
- `query::one(...)`
- likely `query::opt(...)`
- a way to request ephemeral `Row<'job>`

Why `query::all(...)` exists:

- it distinguishes one conjunctive stream from multiple independent query objects,
- it avoids overloading tuple syntax ambiguously,
- it leaves room for future dynamic query forms.

## Query Safety Rule

If two declared live queries can overlap in a conflicting way and the engine cannot prove them disjoint, they must be rejected.

Examples:

- `Read<T>` plus `Write<T>` on potentially overlapping rows: reject.
- `Write<T>` plus `Write<T>` on potentially overlapping rows: reject.
- `Write<T> + Has<U>` versus `Write<T> + Not<Has<U>>`: acceptable if the planner can prove disjointness.

MVP stance:

- be conservative,
- reject more rather than less,
- provide deferred write paths for patterns that cannot be made live safely,
- and produce diagnostics that steer users toward those deferred paths when possible.

## Nested Queries

The rewrite explicitly does not promise arbitrary live nested queries in the MVP.

Selected stance:

- direct overlapping nested queries are not a first-pass goal,
- the planner should reject unsound live overlaps,
- future work may add relational operators and special nest-aware modes,
- scratch-chunk transactional execution remains an advanced option, not the default model.

See `future/08-nested-query-soundness.md`.

## Scheduler Model

## Executor Shape

Assumed executor model:

- fixed-size worker pool,
- local deques per worker,
- work stealing,
- independently stealable job objects,
- optional correlation or affinity hints so related jobs tend to stay on one thread.

The ability to process thousands of jobs efficiently is a hard requirement.

## Static Schedule Versus Runtime Job Expansion

The selected model has two levels:

### Static Planning

At schedule build time, the system analyzes:

- declared query access,
- command capabilities,
- explicit ordering constraints,
- implicit declaration-order happens-before relationships.

This produces a reusable function-level schedule.

### Runtime Expansion

At frame execution time, functions expand into per-chunk jobs based on current table/chunk state.

This separation matters because:

- chunk existence is dynamic,
- tables may grow mid-frame,
- the expensive reasoning should happen ahead of time when possible.

## Job Granularity

Functions emit per-chunk jobs.

Important selected constraint:

- chunk jobs remain independently stealable,
- they are not fused into non-stealable bundles,
- affinity hints may bias placement without constraining correctness.

This preserves the scheduler's freedom to exploit parallelism on dominant tables.

## Dynamic Scheduling

New chunk jobs must be schedulable immediately when new chunks become visible.

Example:

1. function `A` resolves inserts into table `T`,
2. that resolution creates chunk `C12`,
3. later function `B` depends on chunk jobs over all chunks of `T`,
4. `B` must become able to schedule and run the job for `C12` in the same frame if ordering allows.

The intended model is:

- lightweight in-flight scheduling for new jobs,
- heavier reshaping or reanalysis at frame boundaries.

## Ordering Semantics

Default rule:

- declaration order induces happens-before for conflicting functions.

Important refinement:

- this happens-before relation is resource-scoped, not globally serializing.

Example:

- if function `F2` touches chunk `X`,
- it only waits for earlier jobs or resolve jobs from `F1` that also affect `X` or otherwise conflict with `F2`'s declared access.

This is what lets late functions start on unaffected chunks before all earlier work is globally finished.

## Determinism

Determinism is explicitly not a goal.

Required property:

- all declared happens-before edges must hold.

Allowed property:

- different worker counts or steal decisions may change exact execution order and row insertion order.

## Structural Commands

## Recording

Structural operations are deferred.

Each runtime job records its structural edits into a local command buffer.

Why local buffers:

- no shared hot-path contention while recording,
- predictable ownership,
- cheap append behavior.

## Resolution

Command buffers are applied by scheduled resolve jobs.

Important selected semantics:

- a function's own structural edits are not visible inside the function's live chunk jobs,
- they become visible only after the relevant resolve jobs complete,
- later dependent functions in the same frame can observe them if happens-before allows.

Selected resolve granularity:

- command recording remains per runtime job,
- but resolution is not modeled as "one resolve job per originating chunk job",
- instead, the resolve phase for one scheduled function collects all of that function's local command buffers and resolves them in batch,
- the resolver may internally partition the batch by target table or chunk when that is provably safe and profitable, but the baseline semantic unit is the function-level resolve family.

## Batch APIs

Command buffers should expose batch-friendly interfaces.

Examples:

- `one(row)`
- `array([row; N])`
- `slice(&[row]) where row: Copy`

The merge stage should group compatible commands to minimize structural coordination and memory movement.

This is especially important for remove-heavy workloads:

- batching all command buffers from one function into one resolve pass avoids spawning many mutually conflicting remove-resolve jobs,
- gives the resolver global visibility over all affected chunks,
- and opens the door to better sorting, deduplication, and capacity planning.

## Insert Semantics

Insert is relatively light:

- rows append into available space or new chunks,
- no existing rows move,
- conflicts are mostly about target table/chunk capacity and visibility, not row relocation.

## Remove Semantics

Remove is heavier:

- removal uses `swap_remove`,
- rows in the target chunk may move,
- any other command or query that depends on those same rows or chunk positions must respect that movement.

Current selected rule:

- potentially overlapping removes on the same chunk conflict,
- later dependent jobs must wait for remove resolution.

Open performance question:

- can some remove patterns be relaxed or batched more aggressively without weakening the row-movement guarantees needed by `Row<'job>` targeting?

## Deferred Value Writes

The rewrite should leave room for non-structural deferred writes as an additional conflict-management tool.

Example direction:

- `Set<T>`
- `Patch<T>`
- `WriteLater<T>`

Rationale:

- some overlapping patterns are not best solved by live exclusive access,
- some are better expressed as "read now, produce value updates, apply later",
- this may become the safe answer for classes of nested or overlapping workloads that do not justify full relational operators.

This is not required for the first MVP milestone, but it is an intentional design hook rather than an afterthought.

## Identity Model

## Keyless Default

The default table mode is keyless.

Keyless tables do not maintain:

- engine-managed keys,
- reverse slot maps,
- per-row stable identity.

This is deliberate. Hot scan-heavy data should not pay identity tax by default.

## `Row<'job>`

Keyless row targeting uses an ephemeral row handle.

Planned properties:

- packed into a `u64`,
- contains at least:
  - table index,
  - chunk index,
  - row index within the chunk,
- row bits versus chunk bits are table-parameterized,
- lifetime-carried API, for example `Row<'job>`, to signal that the handle is transient and should not be stored.

`Row<'job>` is for:

- equality within one job,
- deferred row-targeted commands tied to the current scheduled epoch,
- local joins or diagnostics.

`Row<'job>` is not for:

- persistent storage,
- cross-frame identity,
- post-resolution assumptions that the same row still names the same logical object.

## Managed Keys

Stable identity is opt-in.

Selected direction for keyed tables:

- keyed tables carry an inline `Key` column,
- queries can request `&[Key]` directly,
- a global `Keys` resource maintains `Key -> Row` mapping,
- the inline column provides the cheap `Row -> Key` direction.

This is the cleanest way to preserve:

- cheap query-time key access,
- reverse lookup,
- stable identity through row movement.

User-keyed tables are intentionally deferred.

## Deferred Identity Extensions

Two identity-related extensions remain intentionally outside the MVP:

- user-keyed tables,
- the `Families` resource layered on managed keys.

User-keyed tables matter for rows whose stable identity exists but should not be engine-generated.

`Families` matters for hierarchical relationships such as:

- parent,
- first child,
- next sibling.

The selected direction is:

- first make keyless and managed-key tables correct,
- then add user-keyed and relationship-oriented identity layers afterward.

## Globals And Singletons

The first rewrite treats globals, resources, and singleton-like data exactly like other tables.

Implications:

- no special resource map in the MVP,
- no singleton fast path,
- singletons can be modeled as one-row tables,
- scheduler reasoning stays uniform.

Convenience APIs such as:

```rust
query::one(query::read::<Physics>())
```

are encouraged, but they are syntax over the same table machinery.

## Column Decomposition

Optional decomposition remains part of the long-term direction, but not the MVP.

Current intended stance:

- by default, one logical datum becomes one physical column,
- future decomposition may split POD-like types into multiple physical columns,
- future resource IDs and query descriptors must leave room for subcolumn access,
- recursive or heap-rich decomposition is deferred.

This is a design constraint on the storage metadata, not an immediate implementation requirement.

## Event Model

The current crate has an event system, but event parity is not a first-blocker for the rewrite.

Selected stance:

- first make storage, queries, scheduling, and resolve semantics correct,
- then reintroduce structural events with explicit frame and resolve semantics,
- avoid rebuilding current event behavior blindly before the new visibility model is stable,
- and make event payloads identity-aware:
  - keyed tables can use stable keys,
  - keyless tables cannot promise persistent row identity and will need a different payload design.

## Scope By Phase

The plan intentionally has two layers:

### MVP Rewrite Scope

Tasks `00` through `10` define the first coherent rewrite milestone:

- core metadata,
- chunk storage,
- keyless rows,
- chunk-native queries,
- schedule building,
- executor runtime,
- command buffers and resolve jobs,
- managed keys,
- table-backed globals,
- validation and migration discipline.

### Post-MVP Scope

Tasks `11` and later capture important extensions that should not block the first coherent milestone:

- direct exclusive API and compatibility boundary,
- user-keyed tables,
- `Families`,
- relational query operators,
- deferred value writes,
- advanced nested-query support,
- event-system reintroduction,
- optional column decomposition and layout specialization.

## Rewrite Strategy

This rewrite should be developed side by side with the current implementation rather than by incrementally mutating the existing core in place.

Reasons:

- the current crate remains the best behavior oracle for many semantics,
- the new design changes almost every important internal invariant,
- side-by-side work makes benchmarking and regression comparison easier,
- documentation can be updated without destabilizing the current public surface prematurely.

## Remaining Open Design Questions

These questions are intentionally still open:

1. What should the default `target_chunk_bytes` be after benchmarking?
2. What exact API should express explicit barriers, selective barriers, or relaxed ordering?
3. How should dynamic chunk-job injection interact with existing worker-local queues and affinity hints?
4. How should batched remove resolution minimize ordering overhead without weakening soundness?
5. Should user-keyed tables enter the first post-MVP identity wave, or later?
6. How much shallow decomposition of heap-owning datums is worth supporting?
7. When events return, should they be tied to resolve jobs, frame phases, or both?
8. What exact surface should the direct exclusive API expose, and how close should it stay to the current crate's direct database ergonomics?
9. Should deferred non-structural writes such as `Set<T>` arrive immediately after MVP, or only after relational query operators exist?
10. What exact semantics should user-keyed tables provide:
   - unique only,
   - replace/upsert,
   - ordered versus hashed lookup,
   - or several modes?
11. How should `Families` maintenance be staged:
   - batched scheduled updates first,
   - selective atomics later,
   - or a different split?

## Success Criteria

The rewrite is on track when all of these become true:

- the hot path does not rely on ad hoc runtime lock orchestration,
- scan-heavy tables can run without identity maintenance,
- per-chunk jobs scale across worker threads,
- chunk sizing is easy to tune and benchmark,
- same-frame structural visibility is governed by explicit resolve jobs,
- scheduler semantics are simple enough to explain in a few pages,
- the codebase has enough instrumentation to choose defaults empirically.
