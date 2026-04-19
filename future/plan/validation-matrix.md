# Validation Matrix

This file is the concrete verification map for the `v2` rewrite lane.

Read this together with:

- [specification.md](/home/goulade/Projects/rust/that_bass/future/plan/specification.md)
- [standards.md](/home/goulade/Projects/rust/that_bass/future/plan/standards.md)
- [10-validation-and-migration.md](/home/goulade/Projects/rust/that_bass/future/plan/10-validation-and-migration.md)
- [AGENTS.md](/home/goulade/Projects/rust/that_bass/AGENTS.md)

The goal is simple: every meaningful rewrite change should have an obvious place to prove itself.

## Validation Layers

### Semantic Validation

Use the integration suites under `tests/v2/` as the main semantic oracle.

- [foundation.rs](/home/goulade/Projects/rust/that_bass/tests/v2/foundation.rs)
  - rewrite-lane boundary, chunk-capacity planning, measurement categories.
- [metadata.rs](/home/goulade/Projects/rust/that_bass/tests/v2/metadata.rs)
  - `Meta`, row packing, table interning, dependency-path generation.
- [chunk_layout.rs](/home/goulade/Projects/rust/that_bass/tests/v2/chunk_layout.rs)
  - one-allocation chunks, bootstrap growth, dense-prefix storage, `swap_remove`.
- [keyless_rows.rs](/home/goulade/Projects/rust/that_bass/tests/v2/keyless_rows.rs)
  - generated `Rows<'job>`, transient row handles, batched remove behavior.
- [query_surface.rs](/home/goulade/Projects/rust/that_bass/tests/v2/query_surface.rs)
  - chunk-native projection, optional views, query validation, disjointness proof.
- [global_tables.rs](/home/goulade/Projects/rust/that_bass/tests/v2/global_tables.rs)
  - `Store::initialize_global(...)`, `query::one::<T>()`, singleton scheduling rules.
- [managed_keys.rs](/home/goulade/Projects/rust/that_bass/tests/v2/managed_keys.rs)
  - reserved/live key states, keyed publish, keyed remove, keyed random lookup.

### Concurrency Validation

Use the executor and resolution suites to prove happens-before behavior and same-frame visibility.

- [schedule_builder.rs](/home/goulade/Projects/rust/that_bass/tests/v2/schedule_builder.rs)
  - dependency conflict rules, wildcard coverage, barriers, declaration-order edges.
- [executor_runtime.rs](/home/goulade/Projects/rust/that_bass/tests/v2/executor_runtime.rs)
  - seeded per-chunk jobs, work stealing, resolve-driven chunk injection.
- [command_resolution.rs](/home/goulade/Projects/rust/that_bass/tests/v2/command_resolution.rs)
  - per-job command recording, batched resolve, same-frame downstream visibility.
- [global_tables.rs](/home/goulade/Projects/rust/that_bass/tests/v2/global_tables.rs)
  - singleton accesses participate in the same dependency graph as other tables.

### Unsafe Validation

Unsafe-sensitive paths should receive focused nightly Miri coverage rather than only normal tests.

Current focused suites:

- `cargo +nightly miri test --test v2 suite::chunk_layout`
- `cargo +nightly miri test --test v2 suite::keyless_rows`
- `cargo +nightly miri test --test v2 suite::query_surface`
- `cargo +nightly miri test --test v2 suite::schedule_builder`
- `cargo +nightly miri test --test v2 suite::executor_runtime`
- `cargo +nightly miri test --test v2 suite::command_resolution`
- `cargo +nightly miri test --test v2 suite::managed_keys`
- `cargo +nightly miri test --test v2 suite::global_tables`

When a patch changes an unsafe-adjacent path, extend this list only as much as needed to cover that
path directly.

### Performance Validation

Use `benches/v2/` for performance work. Keep benchmark logic outside library modules.

- [foundation.rs](/home/goulade/Projects/rust/that_bass/benches/v2/foundation.rs)
  - store construction, chunk-capacity planning, chunk allocation and bootstrap growth, target
    chunk-byte sweeps.
- [runtime.rs](/home/goulade/Projects/rust/that_bass/benches/v2/runtime.rs)
  - many tiny chunk jobs, same-frame injection, worker-count sweeps, injection-policy sweeps.
- [workloads.rs](/home/goulade/Projects/rust/that_bass/benches/v2/workloads.rs)
  - scan-heavy row widths,
  - one dominant table versus many medium tables,
  - keyed-column versus keyless scan cost,
  - singleton-table access through `Store::initialize_global(...)` plus `query::one::<T>()`,
  - remove-heavy runtime cost.

## Migration State

The current migration strategy is intentionally side-by-side.

- `src/v1/` remains the stable reference implementation.
- `src/v2/` is the rewrite lane.
- [examples/v2/](/home/goulade/Projects/rust/that_bass/examples/v2/) is the runnable API-evolution trace.
- `benches/v1/` and `benches/v2/` stay split so regression comparison remains straightforward.

Current status:

- Tasks `00` through `10` are implemented in `src/v2/`.
- Tasks `11` and later are still post-MVP work.
- No compatibility layer is promised yet; migration is by comparison, validation, and progressive
  API stabilization.

## Per-Task Map

### Tasks `00` to `04`

Primary proof points:

- `tests/v2/foundation.rs`
- `tests/v2/metadata.rs`
- `tests/v2/chunk_layout.rs`
- `tests/v2/keyless_rows.rs`
- `tests/v2/query_surface.rs`
- `examples/v2/main.rs`

### Tasks `05` to `07`

Primary proof points:

- `tests/v2/schedule_builder.rs`
- `tests/v2/executor_runtime.rs`
- `tests/v2/command_resolution.rs`
- `examples/v2/main.rs`
- `benches/v2/runtime.rs`

### Tasks `08` to `09`

Primary proof points:

- `tests/v2/managed_keys.rs`
- `tests/v2/global_tables.rs`
- `examples/v2/main.rs`
- `benches/v2/workloads.rs`

### Task `10`

Primary proof points:

- this file,
- [10-validation-and-migration.md](/home/goulade/Projects/rust/that_bass/future/plan/10-validation-and-migration.md),
- `benches/v2/runtime.rs`,
- `benches/v2/workloads.rs`,
- and the repo-level maintenance rules in [AGENTS.md](/home/goulade/Projects/rust/that_bass/AGENTS.md).

## Completion Questions For Any Rewrite Patch

Before considering a rewrite patch finished, answer these:

1. Which task file does it satisfy?
2. Which `tests/v2/` file proves the semantics?
3. Which focused nightly Miri suite covers the unsafe-sensitive path, if any?
4. Which benchmark file measures the cost if performance is part of the change?
5. Which example in `examples/v2/` shows the public surface?
6. Were [specification.md](/home/goulade/Projects/rust/that_bass/future/plan/specification.md), [10-validation-and-migration.md](/home/goulade/Projects/rust/that_bass/future/plan/10-validation-and-migration.md), and [AGENTS.md](/home/goulade/Projects/rust/that_bass/AGENTS.md) updated if the semantics changed?
