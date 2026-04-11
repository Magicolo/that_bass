# Task 07: Command Buffers, Batched Resolution, And Structural Visibility

This task defines how structural writes are recorded and when they become visible.

Read this file together with `future/plan/specification.md` and `future/plan/standards.md`. This task is where the scheduler and the storage model become one coherent execution model. If command recording, merge, and resolution are poorly defined, the rest of the rewrite will not behave predictably.

## Purpose

Implement deferred structural mutation through:

- per-job command buffers,
- batch-oriented command APIs,
- scheduled batched resolve phases,
- explicit visibility boundaries,
- chunk-aware conflict handling for inserts and removes.

## Required Reading

- `future/plan/specification.md`
- `future/plan/standards.md`
- `future/04-scheduler-first-lockless-mode.md`
- `future/07-keyless-and-user-keyed-tables.md`
- `future/08-nested-query-soundness.md`
- `AGENTS.md`
- `src/v1/create.rs`
- `src/v1/modify.rs`
- `src/v1/destroy.rs`

## Selected Principles

1. Jobs record commands locally, not into a shared global buffer.
2. Resolution is explicit scheduled work.
3. A function does not observe its own structural edits live.
4. Later dependent functions may observe those edits in the same frame after resolve.
5. Resolution should be batch-oriented.
6. Remove is semantically heavier than insert because remove moves rows.
7. The default resolve granularity is function-level batching over all command buffers produced by that function, not one resolve job per source chunk.

## Command Buffer Ownership

Preferred ownership model:

- one local command buffer per runtime job,
- then collected and batched before resolve execution.

Why:

- no shared synchronization while recording,
- append-friendly,
- good cache locality for the producer,
- natural alignment with per-chunk execution.

The resolver can merge buffers later by:

- command kind,
- target table,
- target chunk,
- target identity policy.

Important design change:

- local recording remains one buffer per originating job,
- but those buffers are not replayed as one resolve job per buffer,
- they are collected into the producing function's resolve phase and processed in batch.

## Command Shapes

At minimum, the rewrite should define command families for:

- `Insert<T>`
- `Remove<Row<'job>>` for keyless tables
- `Remove<Key>` for keyed tables later
- future deferred value writes such as `Set<T>` or `Patch<T>`
- future `Destroy`, `Add`, `RemoveComponent`, `Modify`, and event emission hooks

Batch-friendly APIs should exist from the start.

Examples:

```rust
insert.one(row);
insert.array(rows);
insert.slice(rows);
```

The exact API names are flexible. The batchability is not.

## Resolve As Scheduled Work

Resolve work should not happen implicitly at arbitrary points in a worker callback.

Instead:

- function jobs complete,
- their local command buffers become inputs to resolve families,
- batched resolve work runs under the scheduler,
- later dependent jobs wait on that batched resolve work when needed.

This is what makes same-frame visibility analyzable.

## Insert Resolution

Insert resolution should:

1. collect all insert commands produced by the function,
2. group rows by target table,
3. compute required capacity across the whole batch,
4. append into existing partially full chunks when possible,
5. allocate new chunks when needed,
6. publish new chunk/job visibility for downstream work.

Because insert does not move existing rows:

- its ordering requirements are lighter than remove,
- the main effects are visibility and chunk creation.

## Remove Resolution

Remove resolution should:

1. collect all remove commands produced by the function,
2. group targets by table and chunk,
3. sort and deduplicate row targets within each affected chunk,
4. resolve each affected chunk from highest row index downward,
5. apply `swap_remove`,
6. update any identity side data for keyed tables later,
7. publish the new chunk counts and moved-row consequences before downstream dependent jobs run.

The important semantic fact is that remove moves rows.

That is why remove conflicts more strongly than insert.

## Why Remove Needs Extra Care

Suppose two jobs both queue remove commands for the same chunk.

Even if they target different rows:

- the first remove can move the last row into the second row's slot,
- therefore the second target can become stale if the resolver does not batch or serialize carefully.

The MVP should therefore use one of these safe strategies:

- serialize conflicting chunk application inside one batched resolve pass according to declaration order when needed,
- or merge all same-chunk removes into one batched chunk-resolution step with a stable descending-order algorithm.

Either approach is valid. The key is to be explicit.

## Suggested Merge Policy

One practical direction is:

1. every job records locally,
2. when the function completes, all of its command buffers are collected,
3. the resolve phase groups buffered commands by command family, table, and chunk,
4. the resolver applies those groups in one scheduled batched resolve pass or in a very small number of internal partitions when provably disjoint.

This keeps local recording cheap while still allowing safe global optimization across everything that the function produced.

It also addresses the earlier serialization issue:

- many remove-producing chunk jobs no longer create many mutually conflicting remove-resolve jobs,
- instead they feed one batched remove resolution pass with full knowledge of all targeted chunks.

## Visibility Example

Consider:

```rust
schedule.push(query::all((query::read::<Emitter>(),)), emit_particles);
schedule.push(query::all((query::write::<Particle>(),)), integrate_particles);
```

If `emit_particles` records `Insert<Particle>` commands, the intended semantics are:

- particle inserts are not visible inside `emit_particles`,
- the batched insert resolve phase runs after the emitter jobs,
- newly created particle chunks become eligible for `integrate_particles` in the same frame,
- `integrate_particles` receives additional runtime jobs for those new chunks.

This is a core use case that should drive the implementation.

## Example API Sketch

```rust
schedule.push(
    query::all((query::read::<Spawner>(),)),
    |spawners, mut insert: Insert<(Position, Velocity)>| {
        for spawner in spawners {
            insert.one((spawner.position, spawner.velocity));
        }
    },
);
```

and:

```rust
schedule.push(
    query::all((query::rows(), query::read::<Lifetime>())),
    |rows, lifetimes, mut remove: Remove| {
        for (row, lifetime) in rows.zip(lifetimes) {
            if lifetime.done {
                remove.one(row);
            }
        }
    },
);
```

## Open Performance Questions This Task Must Preserve

These are still open, so do not bury the answers prematurely:

1. How much can remove resolution be parallelized without weakening row-targeting guarantees?
2. When several local buffers target the same table but different chunks, what internal grouping structure inside one batched resolve phase is cheapest?
3. When the batched resolve phase sees provably disjoint targets, how aggressively should it partition that work internally?
4. When deferred value writes arrive, should they reuse the same resolve machinery as structural commands or have a lighter-weight path?

Design the command pipeline so those choices can be benchmarked.

## Implementation Checklist

1. Implement local per-job command buffers.
2. Implement batch append APIs.
3. Implement collection of all buffers produced by one function.
4. Implement pre-resolve grouping and merge logic across that whole function batch.
5. Implement insert resolution.
6. Implement remove resolution with safe same-chunk handling.
7. Wire batched resolve phases into the scheduler runtime.
8. Add tests for:
   - same-frame insert visibility to later functions,
   - no self-visibility inside the producing function,
   - safe same-chunk batched remove,
   - new chunk injection after insert resolution,
   - one function with many originating command buffers still producing one batched resolve phase.

## Pitfalls

### Pitfall: Treating command buffers as just an optimization

They are a semantic boundary, not only a performance feature.

### Pitfall: Accidentally exposing partially resolved state

Visibility must change at resolve boundaries, not midway through local buffer replay.

### Pitfall: Making remove behave like append-only insert

Remove is the movement-heavy operation. Its ordering rules need to stay stronger.

## Done Criteria

This task is done when:

- jobs can record structural edits locally,
- one function's command buffers are collected and resolved in batch,
- later functions can observe resolved changes in the same frame,
- insert and remove semantics are both explicit and testable.
