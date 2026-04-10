# Task 00: Rewrite Lane, Scope, And Measurement

This is the first implementation task for the rewrite.

Read this file together with `future/plan/specification.md` and `future/plan/standards.md`. This task exists to prevent the rewrite from turning into an unbounded refactor with no benchmarks, no architectural boundary, and no stable source of truth.

## Purpose

Create the rewrite lane before implementing the rewrite.

That means:

- choose where the new architecture will live in the repository,
- preserve the current crate as a behavioral reference,
- set up measurement before tuning,
- define the minimal vocabulary and success criteria that every later task uses.

If this task is skipped, later tasks will drift and the rewrite will become harder to review.

## Required Reading

- `AGENTS.md`
- `future/plan/specification.md`
- `future/plan/standards.md`
- `future/03-chunking-locks-and-query-plans.md`
- `future/04-scheduler-first-lockless-mode.md`
- `src/lib.rs`
- `src/table.rs`
- `src/query.rs`

## Decisions Already Taken

- The rewrite is side-by-side, not an in-place mutation of the current engine.
- The scheduler owns hot-path safety.
- Tables are chunked and keyless by default.
- Singletons/resources are just tables in the first version.
- Jobs are independently stealable per-chunk runtime units.
- The target chunk byte budget must be easy to change for benchmarking.

## Deliverables

1. A dedicated rewrite module or crate boundary.
2. A short architecture README for that boundary, pointing back to `future/plan/specification.md`.
3. A benchmark harness that can compare current behavior against rewrite prototypes.
4. A minimal instrumentation plan for:
   - schedule build cost,
   - runtime queue overhead,
   - chunk allocation cost,
   - insert/remove resolve cost,
   - dense scan throughput.
5. A glossary module or top-level docs page that defines:
   - schema,
   - table,
   - chunk,
   - row,
   - key,
   - job,
   - resolve job,
   - happens-before.

## Recommended Repository Strategy

Do not start by deleting or rewriting `src/table.rs`, `src/query.rs`, or the existing deferred operations.

Prefer one of these:

- a new internal module tree such as `src/next/` or `src/v2/`,
- or a sibling crate in the workspace if the rewrite needs clean compilation boundaries.

The current implementation should remain available for:

- behavior reference,
- regression comparison,
- documentation cross-checking,
- staged migration.

## Measurement First

The rewrite will introduce many tunable parameters:

- target chunk byte count,
- chunk growth policy,
- scheduler queue structures,
- affinity heuristics,
- command buffer merge strategies.

Without a benchmark harness, those values will become lore instead of engineering decisions.

At minimum, create repeatable measurements for:

### Scan Kernels

- one table, one hot column,
- one table, two columns zipped,
- one dominant table with many chunks,
- several medium tables.

### Structural Workloads

- batch insert into empty table,
- batch insert into partially full table,
- batched remove with clustered targets,
- batched remove with random targets,
- create-heavy then query-heavy frame shapes.

### Scheduler Overhead

- many tiny chunk jobs,
- medium chunk jobs,
- a mix of cheap and expensive jobs,
- dynamic chunk creation mid-frame.

## Suggested Benchmark Matrix

Use a matrix like:

```text
workload x row width x target_chunk_bytes x worker_count
```

Example row widths:

- 8 bytes,
- 16 bytes,
- 32 bytes,
- 64 bytes,
- 128 bytes.

Example chunk byte targets:

- 8 KiB,
- 16 KiB,
- 32 KiB,
- 64 KiB.

This matters because the chunk-sizing formula is width-sensitive by design.

## Rewrite Guardrails

These rules should be written down before later tasks begin:

1. Do not optimize singletons specially in the first pass.
2. Do not add user-keyed tables until managed-keyed tables are stable.
3. Do not promise arbitrary live nested queries in the MVP.
4. Do not hide row movement; document `swap_remove` semantics clearly.
5. Do not hardcode one chunk byte size as if it were proven.
6. Do not let internal queue implementation leak into user-facing API terminology.

## Suggested Milestones

This task should also define the major gates that later tasks must pass:

### Gate A: Storage Skeleton

- schemas/tables/chunks compile,
- chunk sizes derive from row width,
- memory layout is inspectable.

### Gate B: Query Skeleton

- chunk-only query declarations exist,
- access analysis compiles,
- conflict rejection works for basic cases.

### Gate C: Scheduler Skeleton

- worker pool exists,
- per-chunk jobs are stealable,
- declaration-order happens-before is represented.

### Gate D: Structural Visibility

- command buffers record per job,
- resolve jobs run,
- later functions observe inserted chunks in the same frame.

### Gate E: Identity Layer

- keyless tables work,
- managed-key tables work,
- keyed queries can request chunk `Key` slices.

## Pseudo-Code Sketch

This task should leave the project with a clear shape like:

```rust
pub mod v2 {
    pub mod schema;
    pub mod storage;
    pub mod query;
    pub mod schedule;
    pub mod commands;
    pub mod key;
    pub mod bench;
}
```

That exact layout is negotiable. The point is the boundary, not the names.

## Risks

### Risk: The rewrite becomes a shadow crate with no consumers

Mitigation:

- keep small executable examples or tests that drive the new API early,
- keep benchmark targets running against it.

### Risk: The team debates defaults before tooling exists

Mitigation:

- benchmarks first,
- configuration as data,
- no buried constants.

### Risk: Current semantics get lost

Mitigation:

- cross-reference `AGENTS.md`,
- keep the current code and tests available as semantic reference,
- write down intentional semantic departures as they happen.

## Done Criteria

This task is done when:

- the rewrite lives behind a dedicated boundary,
- benchmark and instrumentation scaffolding exists,
- the glossary and success criteria are written down,
- later tasks have a stable place to land code without destabilizing the current crate.
