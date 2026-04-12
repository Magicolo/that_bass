# Task 10: Validation, Migration, Deferred Features, And Documentation Discipline

This task closes the initial MVP rewrite milestone. It exists to keep the project measurable, migratable, and honest about what is not yet in scope.

Read this file together with `future/plan/specification.md` and `future/plan/standards.md`. A large rewrite fails when it either freezes at “almost done” or tries to absorb every interesting idea before the core is stable. This task is the countermeasure.

## Purpose

Define:

- the validation strategy,
- the migration strategy,
- the performance-tuning loop,
- the list of intentionally deferred features,
- the documentation maintenance rules for the rewrite.

## Required Reading

- `future/plan/specification.md`
- `future/plan/standards.md`
- `future/README.md`
- `future/06-recommended-roadmap.md`
- `AGENTS.md`
- current tests in `tests/v1/` and `tests/v2/`
- current benchmarks in `benches/v1/` and `benches/v2/`

## Validation Strategy

The rewrite needs validation on three fronts:

### 1. Semantic Validation

Check that the new engine's documented behavior matches the selected specification.

Examples:

- chunk queries return dense inhabited-prefix slices,
- inserts become visible only after resolve,
- later dependent functions can observe inserts in the same frame,
- row order is unstable,
- keyed tables preserve `Key <-> Row` mapping.

### 2. Concurrency Validation

Check that scheduler behavior respects declared happens-before edges.

Examples:

- same-writer functions serialize only where resources overlap,
- conflicting remove resolution does not race,
- newly created chunks become eligible for downstream jobs in the same frame,
- lack of determinism does not violate declared ordering.

### 3. Performance Validation

Check that the rewrite actually improves the targeted workloads.

Examples:

- dense scan throughput,
- scheduler overhead on cheap jobs,
- insert/remove batch cost,
- keyed versus keyless table cost,
- effect of different `target_chunk_bytes` settings.

## Benchmark Discipline

The rewrite should ship with benchmark scenarios that are treated as first-class engineering tools, not occasional experiments.

At minimum benchmark:

- scan-heavy tables with different row widths,
- many tiny chunk jobs,
- one dominant table versus many medium tables,
- same-frame insert then downstream query,
- remove-heavy workloads,
- keyed versus keyless tables,
- singleton-table access through `query::one(...)`.

Tests in this rewrite should use `checkito` heavily rather than ad hoc arbitrary values.
Unsafe-sensitive paths should also receive targeted Miri coverage.

The benchmark harness must make it easy to sweep:

- `target_chunk_bytes`,
- worker count,
- affinity heuristics,
- resolve merge strategies.

## Migration Strategy

The rewrite should not immediately replace the current crate behavior wholesale.

Recommended migration path:

1. keep the current implementation intact as reference,
2. land rewrite components behind a dedicated boundary,
3. prove basic workloads and semantics there,
4. add compatibility helpers only after the core model is stable,
5. decide later whether the rewrite replaces the current API, lives beside it, or becomes a sibling crate.

This is a large enough change that premature public-API unification would create confusion.

## Deferred Features

These are important ideas, but they are not part of the first successful rewrite milestone:

### Relational Query Operators

- `lookup`
- `combine`
- `permute`
- `exclude_self`

Reason:

- core storage and scheduling must stabilize first.

### `Families`

Reason:

- depends on keyed identity being stable first,
- relationship maintenance deserves its own design pass.

### User-Keyed Tables

Reason:

- unique-index semantics and update behavior need separate design work.

### Event-System Parity

Reason:

- the new visibility model must settle first,
- event timing should be defined in terms of resolve jobs and frame phases.

### Advanced Field Decomposition

Reason:

- the metadata should preserve room for it now,
- but implementing it too early would multiply complexity.

### Scratch-Chunk Transactional Writes

Reason:

- promising for some future nested/conflicting patterns,
- but not the default MVP write model.

Post-MVP tasks after this file make those deferred areas concrete enough to plan without letting them block the first milestone.

## Documentation Discipline

The rewrite will only stay coherent if the docs stay synchronized with the code.

Required rule:

- after any significant architectural, semantic, scheduling, storage, or roadmap change, update:
  - `AGENTS.md`,
  - `future/plan/specification.md`,
  - and the specific task file or future proposal that changed.
- when committing that change, use the commit-message standard from `future/plan/standards.md` so the rationale and non-obvious effects are preserved in history.

This rule is not optional. It is part of the process.

## End-Of-Task Validation

Whenever a meaningful implementation step lands in the repository, run at minimum:

```bash
cargo fmt
cargo build
cargo clippy --all-targets --all-features
```

Strongly recommended as well:

```bash
cargo test
```

Required when the changed code touches unsafe paths directly or indirectly:

```bash
cargo +nightly miri test --test v2
```

For rewrite-lane work, prefer a focused `v2` Miri run over the full repository test suite. Only run `v1` under Miri when the changed unsafe path reaches into `v1` or shared code that is only exercised there. If nightly Miri is unavailable in the environment, say so explicitly.

If the rewrite introduces separate benchmark targets or crates, add their equivalent checks here too.

## Example Completion Checklist For A Later Rewrite Patch

When a future implementation patch claims to complete part of the rewrite, it should be able to answer:

1. Which task file does this satisfy?
2. What benchmark or test proves it?
3. What semantic rules changed, if any?
4. Were `AGENTS.md` and `future/plan/specification.md` updated?
5. Were `cargo fmt`, `cargo build`, and `cargo clippy --all-targets --all-features` run?
6. Does the commit message follow the repository conventional-commit standard and explain the rationale?

If a patch cannot answer those, it is not finished.

## Done Criteria

This task is done when:

- the validation plan exists,
- the migration path is explicit,
- the deferred-feature list is visible,
- the documentation maintenance rule is written down,
- every later rewrite patch can point back to this task and say how it was verified.
