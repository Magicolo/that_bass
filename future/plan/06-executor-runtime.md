# Task 06: Executor Runtime, Stealable Per-Chunk Jobs, And Dynamic Injection

This task defines the runtime executor that consumes the schedule produced in Task 05.

Read this file together with `future/plan/specification.md` and `future/plan/standards.md`. The user requirement is explicit: per-chunk jobs must be independently stealable, the scheduler must handle thousands of jobs, and correlation should be an affinity hint rather than a hard bundle.

## Purpose

Implement the runtime executor:

- worker pool,
- local deques,
- work stealing,
- cheap runtime job objects,
- family-aware affinity,
- dynamic injection of new jobs created by resolve work.

## Required Reading

- `future/plan/specification.md`
- `future/plan/standards.md`
- `future/04-scheduler-first-lockless-mode.md`
- `AGENTS.md`
- any benchmark scaffolding created in Task 00

## Selected Executor Shape

Assume:

- fixed-size worker pool,
- one local deque per worker,
- work stealing when local work is exhausted,
- many small job objects,
- per-job readiness tracking,
- optional family or affinity tags that bias job placement.

This model matches the selected design constraints:

- thousands of jobs,
- chunk-sized work,
- dynamic chunk creation,
- no determinism requirement.

## Runtime Job Object

The executor needs a cheap, explicit job representation.

Suggested fields:

- job family ID,
- concrete table ID,
- concrete chunk ID,
- projected resource access summary,
- dependency counter or ready flag,
- affinity hint,
- pointer or index to callback/execution logic,
- pointer or index to local command-buffer destination.

The exact storage can be pooled or slab-backed. Avoid heavy heap allocation per job.

## Why Jobs Must Stay Individually Stealable

The user explicitly rejected hard bundling of chunk work into non-stealable super-tasks.

Reason:

- a dominant table may produce many tiny jobs,
- one worker should not monopolize those jobs by construction,
- the scheduler must preserve the option to spread them across cores when that wins.

Therefore:

- affinity is advisory,
- stealability is fundamental.

## Affinity Without Bundling

The executor should still try to keep related jobs near each other when practical.

Possible strategies:

- same-function jobs default to the same worker-local queue that emitted them,
- batched resolve work prefers the worker that owns or aggregates the producing function's buffers,
- newly injected chunk jobs prefer the worker that created the new chunk,
- steal decisions only happen when local work dries up or load imbalance is large.

This preserves locality without sacrificing stealability.

## Dynamic Injection

When resolve work creates new chunks, later function families that depend on those chunks must be able to receive new runtime jobs immediately.

That means the executor needs:

- a way to append newly ready jobs into the live runtime,
- a way to compute their dependency state from the schedule template,
- a way to place them into local queues with an affinity hint,
- a way to let other workers steal them if needed.

This is one of the hardest executor requirements in the rewrite.

## Suggested Runtime Flow

1. Build the frame-local runtime job pool from the reusable schedule and current chunks.
2. Seed ready jobs into worker-local queues.
3. Workers execute jobs and record local command buffers.
4. When a family's jobs complete, its batched resolve work becomes ready.
5. Resolve work may:
   - mutate tables,
   - create chunks,
   - publish new row counts,
   - inject downstream chunk jobs.
6. The executor continues until all job families and resolve families for the frame are drained.

## Scheduling Heuristics To Explore

This task should leave the executor able to benchmark different heuristics.

At minimum, make these swappable:

- local push policy,
- steal victim choice,
- affinity strength,
- whether newly injected jobs go first to the producer's deque or a shared injection queue,
- threshold for preferring local execution versus spreading cheap work.

These are performance questions, not semantic questions. Design the executor so they can be tuned.

## Instrumentation

The executor must expose enough metrics to tune it.

Examples:

- number of jobs created per frame,
- average ready-queue depth,
- steal count,
- average dependency fan-in,
- average wait time before execution,
- injection count from resolve jobs,
- job execution time distribution.

Without these numbers, scheduler tuning will be guesswork.

## Example Data Flow

Suppose a frame has:

- 200 `Integrate` chunk jobs,
- 4 `Spawn` jobs that emit inserts,
- 1 batched `SpawnResolve` pass that consumes those 4 local buffers,
- 204 `Clamp` jobs after new chunks appear.

The executor should be able to show that flow concretely in traces:

```text
Integrate(chunk 0..199)
Spawn(chunk producers)
SpawnResolve(table T creates chunks 200..203)
Clamp(chunk 0..203)
```

The important part is not the exact trace syntax. It is that the runtime job expansion and injection are visible and debuggable.

## Implementation Checklist

1. Implement the worker pool.
2. Implement local deques and stealing.
3. Implement pooled runtime job objects.
4. Implement job readiness tracking.
5. Implement affinity hints.
6. Implement dynamic job injection from resolve work.
7. Add benchmarks for:
   - many tiny jobs,
   - mixed-cost jobs,
   - heavy injection workloads,
   - dominant-table workloads.

## Pitfalls

### Pitfall: Letting affinity become hidden serialization

Affinity should improve locality, not prevent parallelism.

### Pitfall: Heap-allocating every job node independently

That will bury the benefits of chunk-level parallelism in allocator overhead.

### Pitfall: Ignoring injection cost

Dynamic scheduling is a core feature, not a slow path. Measure it directly.

## Done Criteria

This task is done when:

- the runtime can execute many per-chunk jobs across a worker pool,
- jobs are independently stealable,
- affinity exists without forcing bundling,
- resolve work can inject new downstream chunk jobs during the same frame.
