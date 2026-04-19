# Task 06: Executor Runtime, Stealable Per-Chunk Jobs, And Dynamic Injection

This task defines the runtime executor that consumes the schedule produced in Task 05.

Read this file together with `future/plan/specification.md` and `future/plan/standards.md`. The user requirement is explicit: per-chunk jobs must be independently stealable, the scheduler must handle thousands of jobs, and correlation should be an affinity hint rather than a hard bundle.

## Purpose

Implement the runtime executor:

- worker pool,
- worker-local ready queues,
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
- one local ready queue per worker,
- an optional shared ready queue for injected or broadly placed jobs,
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

Selected refinement:

- ordinary jobs do not perform structural mutation directly,
- only resolve phases create new visible chunks or otherwise change structural topology,
- and the main injection case is therefore newly visible chunks inside already-known tables.

This is intentionally narrower than a general "any job may structurally change the store and force
all tasks to update" model.

## Suggested Runtime Flow

1. Build the frame-local runtime job pool from the reusable schedule and current chunks.
2. Seed ready jobs into worker-local queues.
3. Workers execute jobs and record local command buffers.
4. When a family's jobs complete, its batched resolve work becomes ready.
5. Resolve work may:
   - mutate tables,
   - create chunks,
   - publish new row counts,
   - report the affected known tables or chunks,
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

## Update Strategy

The selected executor direction does not rely on a generic user-defined mid-frame `update(&T)`
callback.

Instead:

- schedule-time initialization computes stable per-function descriptors,
- known eligible tables can be cached up front when types make them knowable,
- resolve phases emit narrow structural change information,
- and the executor uses that information to inject new chunk jobs without stopping the world.

Heavier reshaping remains a frame-boundary concern.

## Implementation Checklist

1. Implement the worker pool.
2. Implement local ready queues and stealing.
3. Implement pooled runtime job objects.
4. Implement job readiness tracking.
5. Implement affinity hints.
6. Implement resolve-to-executor reporting of affected tables or chunks.
7. Implement dynamic job injection from resolve work.
8. Add benchmarks for:
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

### Pitfall: Requiring a generic stop-the-world task update path

The selected direction narrows mid-frame dynamism to resolve-driven topology changes in known
tables. Do not add a heavier generic update mechanism unless benchmarks and semantics justify it.

## Done Criteria

This task is done when:

- the runtime can execute many per-chunk jobs across a worker pool,
- jobs are independently stealable,
- affinity exists without forcing bundling,
- resolve work can inject new downstream chunk jobs during the same frame.

## Implementation Review

The current repository now implements this task in `src/v2/runtime.rs` with:

- a public `Seed` built from the current table snapshot rather than direct live-store coupling,
- a public `Executor` and `Options` surface,
- worker-local and shared `VecDeque` ready queues protected by `parking_lot::Mutex`,
- stealing by taking independent jobs from other workers when the local queue is empty,
- one runtime function job per seeded chunk and one runtime resolve job per function family,
- per-job dependency synthesis that merges chunk-specific dependencies with schedule-time static
  job dependencies such as singleton-table or `Keys` access,
- explicit readiness tracking with successor lists and per-job predecessor counters,
- runtime-owned command resolution that mutates `Store` directly and reports resulting
  `VisibleChunk` values through `runtime::Outcome`,
- same-frame injection of later function jobs when resolve phases expose new chunks,
- resolve-to-function visibility edges also delaying the paired later resolve family so a
  function with no seeded jobs cannot resolve before earlier resolves have had a chance to inject
  new work into it,
- runtime traces and summary metrics through `runtime::Report`,
- and executor-focused examples, tests, and benchmarks.

Important implementation note:

- Task 06 also tightened the Task 05 builder so `Insert<T>` table creation updates the matching
  pending query families' known tables and wildcard-chunk dependency templates before the
  reusable schedule is finalized. Without that, later resolve families could become ready too
  early when a newly created table was only discovered through typed insert planning.

## Actions Taken In The Repository

The following concrete actions were taken to satisfy this task:

- added `src/v2/runtime.rs` as the frame-local executor runtime,
- introduced the `Seed`, `Executor`, `Callbacks`, `FunctionContext`, `ResolveContext`,
  `Outcome`, `VisibleChunk`, `Trace`, and `Report` runtime vocabulary,
- implemented worker-local and shared ready queues plus stealing,
- implemented initial per-chunk job seeding from the frame snapshot,
- implemented one batched resolve job per function family,
- implemented runtime-owned batched command resolution and resolve-driven injection of new
  downstream chunk jobs during the same frame,
- added `tests/v2/executor_runtime.rs` for seeded execution, stealing, and injection behavior,
- folded the runnable executor walkthrough into `examples/v2/main.rs` so the single `v2` cheat
  sheet keeps the public runtime surface visible,
- added `benches/v2/runtime.rs` so many-job and injection-heavy executor costs are benchmarkable,
- and updated the schedule builder so typed insert planning refreshes known-table and dependency
  metadata for already-registered matching query families.

Current status:

- implemented in the current repository layout.
