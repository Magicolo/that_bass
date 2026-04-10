# Task 11: Direct Exclusive API, Immediate Mutations, And Compatibility Boundary

This task defines the non-scheduled escape hatch for the rewrite.

Read this file together with `future/plan/specification.md` and `future/plan/standards.md`. The selected architecture is scheduler-first, but the user explicitly did not want the library to lose its direct low-level utility. The rewrite therefore needs a disciplined immediate mode instead of a vague "sometimes you can still touch storage directly" story.

## Purpose

Implement the direct exclusive mode where:

- reads and writes happen immediately,
- mutation requires exclusive mutable access to storage,
- the API remains useful for tests, tools, import/export, bootstrap, and debugging,
- the scheduler-first architecture is not undermined.

## Required Reading

- `future/plan/specification.md`
- `future/plan/standards.md`
- `AGENTS.md`
- current direct APIs in:
  - `src/create.rs`
  - `src/query.rs`
  - `src/modify.rs`
  - `src/destroy.rs`

## Selected Rule

All mutating access in the rewrite must use one of two modes:

1. scheduled mode,
2. direct exclusive mode via `&mut` access to storage.

There is no third mutation path.

This rule should be visible in both the implementation and the docs.

## Why This Task Exists

Without an explicit direct mode, one of two bad outcomes happens:

- the rewrite quietly becomes a framework and loses the standalone database story,
- or ad hoc immediate APIs creep back in without a coherent safety boundary.

This task prevents both.

## Use Cases

The direct exclusive path should exist for:

- tests,
- editor or tooling code,
- bootstrapping world state,
- importing or exporting data,
- debugging and inspection,
- command-line or server-side one-off flows that do not want a schedule runtime.

It is not the mode being optimized for hot frame execution.

## Expected Surface Area

The exact API can evolve, but it should support direct forms of:

- create/insert,
- query/read,
- modify/remove/destroy,
- maybe schedule construction itself.

The key condition is that mutating operations require exclusive access to the storage root.

## Relationship To The Current Crate

This task is the main place to decide how much of the current crate's ergonomic shape survives in the rewrite.

Questions to answer:

- should current-style deferred operation objects still exist in direct mode?
- should direct mode expose chunk-native queries or a compatibility layer over them?
- should current-style `Database`-oriented ergonomics survive as wrappers over the new storage core?

These are not all answered yet, but this task is where the answers belong.

## Isolation From Scheduled Mode

Scheduled mode and direct exclusive mode must not be interleaved casually.

At minimum, define:

- whether a schedule can be built while holding exclusive storage access,
- whether schedule execution requires quiescent storage state,
- whether direct mutation is forbidden while workers are active,
- what runtime state transitions are valid.

The simplest first answer is:

- no direct mutation while scheduled execution is live,
- no scheduled execution while exclusive direct mutation is active.

That may later be refined, but the first implementation should keep the boundary hard.

## Compatibility Layer Strategy

One plausible implementation path is:

- expose a thin direct wrapper API that internally lowers to the same storage primitives used by resolve jobs,
- but executes them immediately under exclusive access.

This gives:

- one storage engine,
- one semantics model,
- two execution modes.

That is preferable to building a second independent mutation subsystem.

## Example Direction

```rust
let mut world = World::new();

world.insert_now((Position { x: 0.0, y: 0.0 }, Velocity { x: 1.0, y: 0.0 }));

world.query_now(query::all((query::write::<Position>(), query::read::<Velocity>())))
    .for_each_chunk(|positions, velocities| {
        for (position, velocity) in positions.zip(velocities) {
            position.x += velocity.x;
        }
    });
```

The exact syntax is not the point. The point is:

- immediate semantics,
- exclusive ownership,
- no scheduler involvement required.

## Implementation Checklist

1. Define the exclusive-mode boundary type or API.
2. Define which immediate operations exist in v1.
3. Decide how direct queries relate to chunk-native views.
4. Decide how much current API compatibility is worth preserving.
5. Add tests for:
   - immediate mutation under exclusive access,
   - inability to overlap direct mutation with live scheduled execution,
   - basic direct query ergonomics.

## Pitfalls

### Pitfall: Recreating the old lock-heavy design under a new name

Direct mode should rely on exclusivity, not on recreating the current concurrent lock choreography.

### Pitfall: Letting direct mode bypass core invariants

It is an execution mode, not a second semantics model.

### Pitfall: Overcommitting to exact current API shape too early

Preserve the spirit of the direct API, but do not let compatibility freeze the new storage model.

## Done Criteria

This task is done when:

- the rewrite has an explicit immediate mutation mode,
- that mode requires exclusive mutable access,
- its relationship to scheduled mode is documented and enforced,
- agents can use the rewrite as a standalone storage engine without inventing their own escape hatch.
