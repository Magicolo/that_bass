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
- `query::all(...)`
- `query::one(...)`
- likely `query::opt(...)`
- filter combinators such as `Has<T>` and `Not<Has<T>>`

The design must avoid this ambiguity:

```rust
(query::write::<Position>(), query::read::<Velocity>())
```

because it could mean:

- one conjunctive stream,
- or two independent streams.

That is why `query::all(...)` should exist.

## Chunk View Semantics

If a query matches a chunk, the user callback should receive dense inhabited-prefix views.

Examples:

```rust
&[Velocity]
&mut [Position]
(&mut [Position], &[Velocity])
(&[Row<'job>], &[Contact])
```

Those views should all share one positional indexing convention:

- index `i` in every slice refers to the same row in that chunk.

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

Every query item compiles into an access descriptor over physical columns.

Examples:

- `read::<Position>()` -> shared access to physical `Position` columns,
- `write::<Position>()` -> exclusive access to physical `Position` columns,
- `rows()` -> no conflicting data access, but requires row-handle materialization,
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

- same logical datum,
- one mutable side,
- no disjointness proof.

### Accept

```rust
(
    query::all((query::write::<Position>(), Has<Dynamic>())),
    query::all((query::write::<Position>(), Not<Has<Dynamic>>())),
)
```

Reason:

- same written datum,
- but the planner can prove table-level disjointness if schema membership is used as the partition.

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

- `query::opt(read::<T>())` returns an optional chunk slice per matching stream,
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
    )),
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
