# Task 05: Schedule Builder, Dependency Graph, And Happens-Before Semantics

This task defines how scheduled functions become an executable plan.

Read this file together with `future/plan/specification.md` and `future/plan/standards.md`. The rewrite moves correctness and most coordination into the scheduler. That means the schedule builder is not a convenience layer. It is a core part of the engine's safety model.

## Purpose

Build the reusable schedule representation that:

- records scheduled functions,
- understands their access needs,
- computes default happens-before edges,
- exposes hooks for future ordering controls,
- can be reused across frames while chunk topology changes underneath it.

## Required Reading

- `future/plan/specification.md`
- `future/plan/standards.md`
- `future/04-scheduler-first-lockless-mode.md`
- `future/08-nested-query-soundness.md`
- `AGENTS.md`
- `src/v1/query.rs`
- `src/v1/modify.rs`
- `src/v1/destroy.rs`

## Selected Scheduler Principles

1. The schedule is built from functions, not from ad hoc job submission.
2. Functions expand into per-chunk jobs at runtime.
3. Declaration order induces default happens-before for conflicting work.
4. Happens-before is resource-scoped rather than globally serializing.
5. The schedule should be reusable across frames.
6. Dynamic in-flight job injection is required for newly created chunks.

## Two-Level Model

The rewrite needs two different levels of scheduling logic:

### Function-Level Planning

At schedule-build time, analyze:

- query access descriptors,
- command capabilities,
- hierarchical dependency shapes,
- explicit ordering annotations,
- declaration order.

Output:

- a graph of function families and resolve families,
- enough metadata to instantiate concrete chunk jobs later.

### Runtime Job Expansion

At frame execution time:

- inspect current tables and chunks,
- instantiate one job per eligible chunk for each scheduled function,
- instantiate resolve work after the relevant function jobs complete,
- inject additional downstream chunk jobs if resolve work creates new chunks.

This distinction is mandatory. Without it, dynamic chunks and reusable schedules fight each other.

## What Counts As A Conflict

A function conflicts with another function when their dependency identifiers can overlap and at least one side is exclusive.

Examples:

- `read(Position)` vs `read(Position)`: no conflict.
- `write(Position)` vs `read(Position)`: conflict.
- `write(Position)` vs `write(Position)`: conflict.
- `write(store)` vs any table/chunk/column access in that store: conflict.
- `write(chunk)` vs any column access in that chunk: conflict.
- `insert into Table<Position, Velocity>` vs later `read(Position)` on that table: conflict through visibility and chunk creation.
- `remove rows from table T` vs later query over those chunks: conflict through row movement and visibility.

The planner should represent these conflicts without yet naming concrete chunk indices.

Important model detail:

- leaf column accesses expand to dependency chains such as:
  - `Read(store)`
  - `Read(table)`
  - `Read(chunk)`
  - `Write(column)`
- broader structural work can request:
  - `Write(store)`
  - `Read(store) + Write(table)`
  - `Read(store) + Read(table) + Write(chunk)`

The scheduler only needs identifiers plus access modes, but it must compare them hierarchically.

## Declaration Order

Default behavior:

- later functions must not violate the effects of earlier conflicting functions.

Important refinement:

- this does not mean the entire later function waits for the entire earlier function,
- it means each concrete runtime job waits only on the earlier jobs or resolve jobs that affect the same resources.

Example:

If function `Integrate` and function `Clamp` both write `Position`, then:

- `Clamp(chunk_5)` waits for `Integrate(chunk_5)`,
- `Clamp(chunk_9)` waits for `Integrate(chunk_9)`,
- but `Clamp(chunk_5)` does not need to wait for `Integrate(chunk_9)`.

That resource-scoped refinement is a major performance property of the rewrite.

The new hierarchical dependency model refines this further:

- a later leaf-column job may depend on an earlier whole-chunk writer for that same chunk,
- a later whole-chunk writer may depend on an earlier leaf-column writer in that chunk,
- but unrelated chunks can still run independently.

## Resolve Nodes Are Part Of The Plan

Structural command application must appear as explicit node families in the schedule.

Why:

- inserts are not visible until resolve,
- removes are not visible until resolve,
- later functions need real dependency edges against visibility changes,
- benchmark and trace output should show where time is spent.

The plan should therefore treat:

- function chunk jobs,
- resolve families that batch one function's recorded command buffers,

as different but related node families.

## Explicit Ordering Controls

The MVP must at least leave space for future user controls such as:

- barriers,
- selective barriers,
- relaxed ordering overrides,
- explicit happens-before or happens-after relationships.

It is acceptable if the first implementation only supports declaration order internally, as long as the schedule representation clearly leaves room for future order annotations.

One practical requirement from the design discussion is that these controls should be able to strengthen or relax the default declaration-order edges without changing the underlying chunk-resource model.

## Reusable Schedule

The schedule should be compiled once and reused across frames.

This means the schedule representation should cache:

- function descriptors,
- query access analysis,
- conflict relationships between function families,
- enough metadata to expand jobs quickly each frame.

It should not assume:

- a fixed chunk count,
- a fixed set of chunk indices forever.

## Example Schedule Shape

For a simple sequence:

```rust
schedule.push(query::all((write::<Position>(), read::<Velocity>())), integrate);
schedule.push(Insert::<(Position, Velocity)>(), spawn);
schedule.push(query::all((write::<Position>(), read::<Bounds>())), clamp);
```

the plan should conceptually become:

```text
Integrate family
  -> Integrate resolve family (maybe empty if no structural commands)
  -> Spawn family
  -> Spawn resolve family
  -> Clamp family
  -> Clamp resolve family
```

with resource-scoped runtime dependencies inside those families.

Important clarification:

- the resolve family is not "one node per source chunk job" by default,
- it is the scheduled phase that gathers all command buffers produced by the function and resolves them as one batched unit or as a small number of batched internal partitions.

## Implementation Checklist

1. Define a function-family descriptor.
2. Define a resolve-family descriptor.
3. Define plan-time conflict edges between families.
4. Define runtime dependency templates for per-chunk execution jobs and function-level batched resolve families, using hierarchical dependency identifiers rather than leaf-only resource names.
5. Define a stable schedule object that can be reused across frames.
6. Add tests for:
   - no conflict between pure reads,
   - chunk-scoped ordering between same-writer families,
   - broad chunk/store writers conflicting correctly with descendant accesses,
   - visibility ordering through resolve families,
   - stable schedule reuse after chunk-count changes.

## Pitfalls

### Pitfall: Planning concrete chunk indices into the static schedule

Chunk existence is dynamic. The plan must describe how to reason about chunks, not freeze the current chunk list.

### Pitfall: Treating resolve as an implementation detail

Resolve jobs define visibility. They belong in the model.

### Pitfall: Globally serializing declaration order

Doing so would erase one of the main advantages of chunk-aware scheduling.

## Done Criteria

This task is done when:

- functions can be registered into a reusable schedule,
- the schedule records conflict and ordering relationships,
- resolve families are represented explicitly,
- the plan is ready for per-frame expansion into concrete chunk jobs.
