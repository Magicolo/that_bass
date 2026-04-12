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
- `src/v1/mod.rs`
- `src/v1/table.rs`
- `src/v1/query.rs`

## Decisions Already Taken

- The rewrite is side-by-side, not an in-place mutation of the current engine.
- The scheduler owns hot-path safety.
- Tables are chunked and keyless by default.
- Singletons/resources are just tables in the first version.
- Jobs are independently stealable per-chunk runtime units.
- The target chunk byte budget must be easy to change for benchmarking.

## Deliverables

1. A dedicated rewrite module or crate boundary.
2. A short architecture README or top-level module docs for that boundary, pointing back to `future/plan/specification.md`.
3. A benchmark harness that can compare current behavior against rewrite prototypes.
4. A minimal instrumentation plan for:
   - schedule build cost,
   - runtime queue overhead,
   - chunk allocation cost,
   - insert/remove resolve cost,
   - dense scan throughput.
5. A glossary module or top-level docs page that defines:
   - meta,
   - table,
   - chunk,
   - column,
   - row,
   - key,
   - job,
   - resolve job,
   - happens-before.

## Recommended Repository Strategy

Do not start by deleting or rewriting `src/v1/table.rs`, `src/v1/query.rs`, or the existing deferred operations.

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
2. Do not add user-keyed tables until the `Keys` extension over ordinary `Key` columns is stable.
3. Do not promise arbitrary live nested queries in the MVP.
4. Do not hide row movement; document `swap_remove` semantics clearly.
5. Do not hardcode one chunk byte size as if it were proven.
6. Do not let internal queue implementation leak into user-facing API terminology.

## Suggested Milestones

This task should also define the major gates that later tasks must pass:

### Gate A: Storage Skeleton

- metas/tables/chunks compile,
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
- tables with `Key` columns work with the `Keys` extension,
- keyed queries can request chunk `Key` slices.

## Pseudo-Code Sketch

This task should leave the project with a clear shape like:

```rust
pub mod v2 {
    pub mod schema;
    pub mod store;
    pub mod query;
    pub mod schedule;
    pub mod command;
    pub mod key;
    pub mod bench;
}
```

That exact layout is negotiable. The point is the boundary, not the names.

## Implementation Review

The current repository now satisfies the task in this concrete shape:

- the stable engine lives under `src/v1/` and remains available as the behavioral reference,
- the rewrite lane lives under `src/v2/` and is isolated from `v1`,
- the crate root only exposes `pub mod v1;` and `pub mod v2;`,
- the rewrite boundary is documented directly in `src/v2/mod.rs`,
- the current foundation API is intentionally small:
  - `Store`,
  - `Configuration`,
  - `ChunkPlan`,
  - `command`,
  - `instrumentation`,
  - `key`,
  - `query`,
  - `schedule`,
  - `schema`,
  - `store`,
- the benchmark harness is grouped by generation in:
  - `benches/v1/`,
  - `benches/v2/`,
- the integration suites are grouped by generation in:
  - `tests/v1/`,
  - `tests/v2/`,
- the measurement plan exists in `src/v2/instrumentation.rs`,
- the glossary now lives in `src/v2/mod.rs` rather than in a separate file.

This means Task 00 should now be treated as implemented, not merely proposed.

## Actions Taken In The Repository

The following concrete actions have already been taken to satisfy this task:

- the stable runtime was moved under `src/v1/` so the rewrite can evolve without sharing the same module root,
- the rewrite foundation was isolated under `src/v2/` and is exposed only through `pub mod v2;`,
- the `v2` boundary docs and glossary were consolidated into `src/v2/mod.rs`,
- the early `v2` public API was renamed to use short complete English words:
  - `Store`,
  - `Configuration`,
  - `ChunkPlan`,
  - `command`,
- integration tests and benchmarks were reorganized by generation so the repository layout now mirrors the source layout:
  - `tests/v1/`,
  - `tests/v2/`,
  - `benches/v1/`,
  - `benches/v2/`,
- the proc-macro crate now points stable derive output at `that_bass::v1::...`,
- Task 00 benchmark scaffolding exists as generation-specific bench targets instead of a single mixed benchmark entry point.

These actions are part of the accepted baseline for later tasks. Future tasks should build on this layout instead of reopening the boundary question unless the rewrite plan itself changes.

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

Current status:

- done in the current repository layout.
- the current baseline includes the `src/v1/` and `src/v2/` split, grouped `tests/` and `benches/`, and the renamed `Store`-oriented `v2` foundation surface.
