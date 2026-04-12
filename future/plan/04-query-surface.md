# Task 04: Query Surface, Chunk Views, And Access Analysis

This task defines how users ask for data and how the engine interprets those requests before any runtime work begins.

Read this file together with `future/plan/specification.md` and `future/plan/standards.md`. The rewrite does not keep the current row-yielding query model. It moves to chunk-native query outputs and requires up-front access declarations that the scheduler can reason about.

## Purpose

Implement the MVP query surface:

- chunk-only outputs,
- clear combinators for conjunctive query streams,
- optional row-handle requests,
- access descriptors that can be analyzed before runtime,
- conservative rejection of conflicting overlapping query declarations.

## Required Reading

- `future/plan/specification.md`
- `future/plan/standards.md`
- `future/08-nested-query-soundness.md`
- `AGENTS.md`
- `src/v1/query.rs`
- `src/v1/row.rs`
- `src/v1/filter.rs`

## Selected Query Principles

1. Queries yield chunk slices, not individual rows.
2. Queries are declared before execution.
3. Scheduler safety comes from declared access and scheduling, not from live lock orchestration.
4. If overlapping conflicting access cannot be proven safe, the query set is rejected.
5. Nested relational operators are postponed, but the API must leave room for them.

## Required User-Facing Building Blocks

The exact API syntax can evolve. The MVP needs the following concepts:

- `query::read::<T>()`
- `query::write::<T>()`
- `query::rows()`
- `query::option(...)`
- `query::all(...)`
- `query::one(...)`
- filter combinators such as `query::has::<T>()` and `query::not(...)`

The design must avoid this ambiguity:

```rust
(query::write::<Position>(), query::read::<Velocity>())
```

because it could mean:

- one conjunctive stream,
- or two independent streams.

That is why `query::all(...)` should exist.

`query::all(...)` should also be the one fallible query constructor:

- `Read<T>`, `Write<T>`, `Rows`, and the other leaf descriptors are just declarations and are not
  invalid by themselves,
- only the conjunctive combination can become unsound,
- so invalid overlapping live access should be rejected at `query::all(...)` construction time,
- and it should be impossible to hold an invalid `All` query value.

## Chunk View Semantics

If a query matches a chunk, the user callback should receive dense inhabited-prefix views.

Examples:

```rust
&[Velocity]
&mut [Position]
(&mut [Position], &[Velocity])
(Rows<'job>, &[Contact])
(Rows<'job>, query::Optional<&[Contact]>)
```

Those views should all share one positional indexing convention:

- index `i` in every slice-like view refers to the same row in that chunk.

## Row-Level Iteration Is Local Sugar

The scheduler reasons about chunk jobs.

If users want per-row iteration, they should use local adapters such as:

```rust
for (position, velocity) in positions.zip(velocities) {
    position.y += velocity.y;
}
```

or future helper adapters that preserve the same semantics.

This keeps the execution model simple:

- the scheduler never has to think in terms of one job per row,
- the query API still feels ergonomic for row-wise logic.

## Access Analysis Model

Every query item compiles into an access descriptor over columns.

Examples:

- `read::<Position>()` -> shared access to the `Position` column,
- `write::<Position>()` -> exclusive access to the `Position` column,
- `rows()` -> no conflicting data access, but requires generation of a `Rows<'job>` view,
- `one(read::<Physics>())` -> shared access with a cardinality constraint.

These descriptors should be generated before runtime expansion into concrete chunk jobs.

## Conflict Rejection

The MVP planner must reject overlapping live access unless it can prove disjointness.

Examples:

### Reject

```rust
query::all((query::write::<Position>(), query::read::<Position>()))
```

Reason:

- same column,
- one mutable side,
- no disjointness proof.

### Accept

```rust
(
    query::all(query::write::<Position>())
        .expect("query declaration should succeed")
        .filter(query::has::<Dynamic>()),
    query::all(query::write::<Position>())
        .expect("query declaration should succeed")
        .filter(query::not(query::has::<Dynamic>())),
)
```

Reason:

- same written datum,
- but the planner can prove table-level disjointness if table-shape membership is used as the partition.

This proof system should start small and conservative.

When the planner rejects a query set, the diagnostic should try to suggest the next-best safe shape, for example:

- split the work into two scheduled functions,
- use a deferred write path such as a future `Set<T>` style command,
- or wait for a future relational operator if the pattern is fundamentally a join.

## Filters

Filters remain important even though the execution engine changes.

MVP support should preserve the existing mental model:

- filters are table-level admission predicates,
- they decide which tables or chunks are eligible,
- they are not row predicates.

That matches the current crate and the chosen chunk-based model well.

## `query::one(...)`

Because singletons are modeled as normal tables, the query layer needs a convenience wrapper that means:

- expect exactly one row across the matched table or table set,
- produce a single view rather than a chunk stream.

This is a query-level convenience, not a special storage path.

## Optional And Missing Data

The current crate has `Option<R>` row semantics. The rewrite should likely keep an optional query concept, but it must adapt to chunk-native outputs.

Possible directions:

- `query::option(read::<T>())` returns a zip-friendly optional chunk view per matching stream,
- or optionality remains a structural query combinator rather than a plain type wrapper.

This exact shape is open, but the task must at least document and prototype one approach.

## Deliberate Non-Goal For This Task

Do not implement arbitrary nested relational operators here.

This task should leave room for future additions such as:

- `lookup`,
- `combine`,
- `permute`,
- `exclude_self`.

It should not require them for the MVP.

It should, however, leave obvious extension points for:

- deferred non-structural writes,
- future nested-query diagnostics,
- future relational operator lowering.

## Example API Sketch

```rust
schedule.push(
    query::all((
        query::rows(),
        query::write::<Position>(),
        query::read::<Velocity>(),
    ))
    .expect("query declaration should succeed"),
    |rows, positions, velocities, commands| {
        for ((row, position), velocity) in rows.zip(positions).zip(velocities) {
            position.x += velocity.x;
            if position.x > 1000.0 {
                commands.remove(row);
            }
        }
    },
);
```

This sketch captures the selected core semantics:

- chunk-native views,
- optional row handles,
- deferred structural edits.

Current implementation note:

- `query::all(...)` is fallible and validates the full conjunctive query when it is built,
- filters attach through `.filter(...)` on `query::all(...)`,
- `query::option(...)` yields a zip-friendly `query::Optional<_>` view rather than a raw `Option<&[T]>`.

## Implementation Checklist

1. Define the query item descriptors.
2. Define `query::all(...)`.
3. Define chunk-view projection for matched tables/chunks.
4. Define filter attachment to query descriptors.
5. Implement conservative conflict detection for overlapping accesses.
6. Add tests for:
   - simple chunk read,
   - simple chunk write,
   - `rows()` plus data slice alignment,
   - reject obvious aliasing conflicts,
   - accept a simple provably disjoint filter split.

## Pitfalls

### Pitfall: Smuggling row-level borrow semantics back into the core API

Keep row iteration as an adapter on top of chunk views.

### Pitfall: Overbuilding the proof engine

Start with obvious table-level disjointness. The MVP does not need a theorem prover.

### Pitfall: Confusing chunk absence with optional value semantics

Be precise about whether optionality means:

- missing table membership,
- missing query stream,
- or maybe one-row cardinality failures.

## Done Criteria

This task is done when:

- the rewrite has a chunk-native query surface,
- access descriptors can be built ahead of time,
- obvious conflicting overlaps are rejected,
- obvious disjoint cases can be accepted,
- the API shape leaves room for future query algebra without forcing it into the MVP.

## Implementation Review

Task 04 is implemented in the isolated `v2` lane.

The current implementation lives primarily in:

- `src/v2/query.rs`
- `tests/v2/query_surface.rs`
- `examples/v2/query_surface.rs`

## Actions Taken

The current implementation does the following:

1. defines typed query descriptors:
   - `query::Read<T>`
   - `query::Write<T>`
   - `query::RowsRequest`
   - `query::OptionQuery<Q>`
   - `query::One<Q>`
2. exposes the public constructors:
   - `query::read::<T>()`
   - `query::write::<T>()`
   - `query::rows()`
   - `query::option(...)`
   - `query::one(...)`
   - `query::all(...)`
3. implements `query::Optional<_>` as a zip-friendly optional view that yields `Option`s of the sub-query's iterated item type,
4. implements table-level filters through:
   - `query::has::<T>()`
   - `query::not(...)`
   - `.filter(...)` on `query::all(...)`,
5. projects chunk views directly from `Table` chunks for reads, writes, rows, optional sub-queries, and `one(...)`,
6. validates declared accesses at `query::all(...)` construction time so invalid `All` queries can
   not exist, and rejects obvious aliasing conflicts there,
7. proves one conservative disjointness case through complementary table-level filters,
8. treats `query::read::<T>()` and `query::write::<T>()` as inline-column views only, so sidecar-only columns do not falsely match the dense slice projection API.

## Current Status

The implemented surface is intentionally query-local:

- chunk views and access analysis exist,
- `query::all(...)` is the fallible validation boundary for conjunctive queries,
- `query::option(...)` is usable in `zip(...)`,
- filters exist for table admission and conservative disjointness proofs,
- but schedule registration and executor integration remain later tasks.

That split is deliberate. Task 04 establishes the query contract before Task 05 and Task 06 turn it into a schedule and runtime.
