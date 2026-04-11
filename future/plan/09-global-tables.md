# Task 09: Tables For Globals, `query::one`, And Uniform Resource Semantics

This task makes the “everything is a table” rule real.

Read this file together with `future/plan/specification.md` and `future/plan/standards.md`. The rewrite explicitly rejects a separate singleton fast path in the first version. This task makes sure that decision is carried through the API and the implementation rather than being quietly walked back.

## Purpose

Implement global settings, resources, and singleton-like data as normal tables, while still giving users convenient query forms.

## Required Reading

- `future/plan/specification.md`
- `future/plan/standards.md`
- `future/04-scheduler-first-lockless-mode.md`
- `AGENTS.md`
- `src/v1/resources.rs`

## Selected Model

There is no separate physical resource map in the first rewrite.

Instead:

- global data lives in ordinary tables,
- the scheduler uses the same dependency logic for that data as for other tables,
- convenience is expressed in the query surface rather than in a separate storage path.

This is a deliberate simplification.

## Why This Was Chosen

Benefits:

- one coherent mental model,
- one resource identity system,
- one scheduling rule set,
- fewer special cases during the rewrite.

Cost:

- singleton tables may carry some metadata that a dedicated hash map would avoid.

That cost is acceptable for the first version because it keeps the architecture uniform while the real performance bottlenecks are still being measured.

## `query::one(...)`

The key convenience API for globals is:

```rust
query::one(query::read::<Physics>())
query::one(query::write::<Physics>())
```

Intended semantics:

- exactly one logical row is expected,
- the query layer produces a single view rather than a chunk stream,
- scheduling and conflict analysis still treat the access through the same table model.

Possible failure modes that the API must define:

- missing row,
- more than one row,
- wrong table state.

Do not leave those implicit.

## Scheduler Implications

Because globals are just tables:

- writes to a singleton table conflict like any other exclusive write,
- reads from that singleton table participate in the same access graph,
- barriers and ordering annotations later will not need a separate resource path.

This consistency is one of the main reasons to keep globals inside the table model initially.

## Example

```rust
schedule.push(
    query::all((
        query::write::<Position>(),
        query::read::<Velocity>(),
        query::one(query::read::<DeltaTime>()),
    )),
    |positions, velocities, dt| {
        for (position, velocity) in positions.zip(velocities) {
            position.x += velocity.x * dt.seconds;
            position.y += velocity.y * dt.seconds;
        }
    },
);
```

This should be legal without any special global-resource subsystem.

## Table Shape For Singletons

The first rewrite should not add storage specialization here.

That means:

- a singleton table still has chunk logic,
- its derived chunk capacity may well be `1`,
- it still participates in table descriptors, chunk descriptors, and scheduler resource IDs.

This may look heavier than a resource map. That is acceptable in v1 of the rewrite.

## Future Optimization Boundary

Even though this task does not specialize globals, it should clearly mark where such specialization could happen later.

Good future boundary:

- physical storage specialization behind the same logical table/query/scheduler interface.

Bad future boundary:

- introducing a second unrelated resource API that bypasses the table model.

## Implementation Checklist

1. Ensure singleton-like tables can be declared through the same table machinery.
2. Implement `query::one(...)`.
3. Define and test cardinality failure behavior.
4. Confirm scheduler conflict logic treats singleton tables exactly like others.
5. Add example tests for:
   - one global read,
   - one global write,
   - mixed chunk stream plus `query::one(...)`,
   - cardinality error cases.

## Pitfalls

### Pitfall: Quietly introducing a hidden resource map

That would undermine the chosen uniform model.

### Pitfall: Making `query::one(...)` a scheduler special case

It is a query convenience, not a separate dependency class.

### Pitfall: Over-optimizing singleton tables before measurement exists

The rewrite already has enough moving parts. Keep this simple first.

## Done Criteria

This task is done when:

- globals and singleton-like data can be stored and queried through ordinary tables,
- `query::one(...)` exists with explicit behavior,
- the scheduler treats those accesses through the same resource model as all other table data.
