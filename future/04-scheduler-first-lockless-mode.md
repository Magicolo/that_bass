# Scheduler-First Lockless Mode

This document describes the future I would optimize for if the target is truly high-end games and realtime simulation.

## Thesis

The highest-performance version of `that_bass` is not a better lock-based query engine.

It is:

- a storage engine plus scheduler,
- where systems declare what they read and write,
- the scheduler builds a dependency graph,
- hot execution runs in readonly mode without runtime locks,
- structural mutations are queued in command buffers,
- sync points make queued changes visible only when necessary.

This should become the primary fast path for games.

## Why This Is The Right Direction

The library goal is:

- maximum parallelism,
- maximum locality,
- simple API,
- "massive efficiency" for users.

Runtime locks work against that goal because:

- they add overhead to every hot iteration,
- they push coordination cost into the frame hot path,
- they make performance dependent on contention timing,
- they complicate worst-case frame-time predictability.

Schedulers are better for games because:

- the game loop already has phases,
- most work is repeated every frame,
- access patterns are known by system type,
- structural changes usually do not need instant visibility.

## The Proposed Model

## 1. Systems Become First-Class

Example shape:

```rust
#[system]
fn integrate(mut query: Query<(&mut Position, &Velocity)>, dt: Res<DeltaTime>) {
    query.each(|(p, v)| {
        p.x += v.x * dt.0;
        p.y += v.y * dt.0;
    });
}
```

The important part is not the macro. The important part is that system params define:

- read set,
- write set,
- structural command capabilities,
- resources,
- maybe optional annotations for deferred writes not visible in the query itself.

## 2. The Frame Runs In Readonly Mode

During scheduled execution:

- iteration over chunks is readonly with respect to structure,
- component value writes are allowed when exclusively scheduled,
- add/remove/create/destroy become commands,
- each worker thread owns a command buffer.

This is the Flecs staging idea adapted to Rust and to the current `that_bass` model.

## 3. Sync Points Are Automatic

The scheduler inserts merge points only when needed.

Example:

- System A writes component values only: no structural sync needed.
- System B queues `Add<TransformDirty>` commands.
- System C reads `TransformDirty`.
- The scheduler inserts a sync point between B and C, not sooner.

This matters because naive "flush everything at end of phase" leaves performance on the table.

## 4. Structural Operations Stop Fighting Queries

Instead of `Query` and `Create/Modify/Destroy` coordinating via locks in the hot path:

- the frame query code just reads chunk slices,
- command buffers record structural edits,
- sync points merge them,
- post-merge queries see the new world.

That gives:

- lockless hot reads,
- deterministic visibility boundaries,
- better batchability of edits.

## API Shape

## User-facing mode

Users should feel like they are writing simple Rust functions.

Core concepts:

- `Query<T>`
- `Res<T>` / `ResMut<T>`
- `Commands`
- `Events<T>` or observers
- maybe `Local<T>` and `ThreadLocal<T>`

This is a deliberately familiar model because it is proven to scale in game teams.

## Escape-hatch mode

The current direct database API should not disappear.

It should remain for:

- tools,
- tests,
- one-off logic,
- external integrations,
- low-level debugging,
- use cases that are not frame scheduled.

But it should be treated as:

- flexible mode,
- not the peak-performance mode.

## Architecture Sketch

## Layer 1: Storage Core

- archetypes or chunked archetypes,
- keys,
- query plans,
- structural command application,
- events / observers,
- resources.

## Layer 2: Access Metadata

- derive or infer read/write sets,
- resource access,
- structural side effects,
- optional annotations for out-of-band reads/writes.

## Layer 3: Schedule Builder

- build DAG from conflicts and dependencies,
- allow explicit ordering when needed,
- batch non-conflicting systems,
- split chunkable systems into jobs.

## Layer 4: Runtime Executor

- worker threads,
- per-thread command buffers,
- sync points,
- frame phases,
- tracing/profiling hooks.

## The Critical Feature: Chunk Jobization

The scheduler should not only run independent systems in parallel.
It should also split one system across many chunks.

Example:

- a movement system over 3 million entities should become many chunk jobs,
- not one giant loop on one worker.

This is where chunked archetypes and scheduler-first execution reinforce each other.

## Command Buffer Design

The current explicit `resolve()` calls are good low-level semantics.
For the scheduler path, they should evolve into:

- per-thread append-only command buffers,
- merged at sync points or end-of-phase,
- grouped by archetype transition when possible.

Useful command forms:

- `Spawn<T>`
- `Destroy(Key)`
- `Add<T>(Key)`
- `Remove<T>(Key)`
- `SetResource<T>`
- maybe `Emit<E>`

Potential optimization:

- represent structural edits as compact archetype-transition batches rather than per-entity commands once they reach the merge stage.

## Safety Model

In the scheduler mode, safety comes from:

- access-declared scheduling,
- readonly structural phase during iteration,
- chunk exclusivity for mutable system jobs,
- no direct structural mutation without command buffering.

This is a stronger, simpler mental model than "it is safe because we took the right locks in the right order".

## Effects On Existing Features

## Query

`Query` becomes lighter in the hot path because:

- it no longer acquires runtime locks,
- it reads from already assigned chunk views,
- keyed lookup can use current-frame chunk mappings.

## Events

Events should probably become frame/stage-aware:

- immediate value-change events for same-job local logic are often a bad idea,
- structural and observer events should probably fire at merge points,
- users should be able to opt into:
  - immediate local events,
  - post-sync events,
  - end-of-frame events.

## Resources

Resources should integrate into the same access graph as queries.

This includes:

- singleton reads/writes,
- thread-local resources,
- maybe stage-local scratch allocators.

## Main Risks

## Risk 1: The library becomes a framework

This is real.

Mitigation:

- keep storage core separate from scheduler,
- let users opt into scheduling rather than forcing it everywhere.

## Risk 2: Sync semantics become subtle

This is also real.

Mitigation:

- make frame visibility rules explicit,
- expose sync points and diagnostics,
- keep structural visibility coarse and deterministic.

## Risk 3: API sprawl

A scheduler path plus direct path can create two libraries in one.

Mitigation:

- define the scheduler API as a layer, not a second storage engine.

## What To Borrow From Existing Systems

- Bevy: system parameter ergonomics and access-derived parallelism.
- Shipyard: workload batching vocabulary.
- Flecs: staging, sync points, query annotations, thread-local command queues.
- Unity Jobs: dependency graph thinking and chunk-sized job discipline.
- hecs/yaks: storage layer and scheduler layer can be separate.

## Recommended End State

The end state I would target is:

- `that_bass_core`
  - chunked archetype storage
  - direct query/manipulation API
  - command primitives
- `that_bass_schedule`
  - systems, schedules, phases, executor
  - lockless frame iteration
  - auto sync points
  - profiling/tracing

Users that want absolute performance for games should mostly live in the scheduler layer.

## Strong Conclusion

If the project truly wants to push hardware to its limits, a scheduler-first lockless mode is not optional. It is the natural destination.

Chunked archetypes plus deferred commands solve the storage side.
Scheduler-owned access solves the execution side.
Together they can give `that_bass` a much stronger performance ceiling than any purely lock-tuned evolution of the current API.
