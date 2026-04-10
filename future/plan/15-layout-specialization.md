# Task 15: Optional Column Decomposition And Layout Specialization

This task implements the advanced layout path that the rewrite deliberately postpones until the core is stable.

Read this file together with `future/plan/specification.md` and `future/plan/standards.md`. The proposals explored increasingly granular storage layouts, but the selected direction was to postpone them until after chunking, scheduling, and identity policy are working well.

## Purpose

Add optional layout specialization such as:

- datum decomposition into multiple physical columns,
- subcolumn query access,
- more selective scheduler conflicts,
- better SIMD opportunities for suitable data.

## Required Reading

- `future/plan/specification.md`
- `future/plan/standards.md`
- `future/02-field-granularity-and-layout.md`
- `AGENTS.md`

## Selected Scope

This is an opt-in expert feature, not the default storage path.

Default rule remains:

- one logical datum maps to one physical column.

This task adds the optional ability to split suitable logical datums into several physical columns.

## Canonical Example

Given:

```rust
struct Position {
    x: f64,
    y: f64,
}
```

Optional decomposition could store it as:

- one `x` column,
- one `y` column.

Then queries might request:

```rust
PositionView<'a> { x: &'a [f64], y: &'a [f64] }
```

or even:

```rust
query::write::<Position::Y>()
```

depending on the final API design.

## Why This Was Deferred

It adds real power, but also multiplies complexity:

- logical-to-physical mapping becomes richer,
- scheduler resource analysis becomes finer,
- derive or declaration macros become more involved,
- chunk layout code gets more complicated,
- query surface area expands.

Those are worthwhile complexities only after the core engine is stable and benchmarked.

## First-Phase Limits

The first decomposition pass should stay conservative.

Recommended limits:

- POD-like types first,
- shallow decomposition only,
- no recursive decomposition of arbitrary heap-rich graphs,
- explicit opt-in through declaration or derive support.

The open question about shallow heap-owning decomposition should stay open until there are benchmarks and concrete use cases.

## Scheduler Implications

Once decomposition exists, the scheduler can reason about smaller resources.

Example:

- one job writing `Position::Y`,
- another reading `Position::X`,

may no longer conflict if the physical columns are distinct.

This is one of the main reasons Task 01 had to distinguish logical datums from physical columns from the start.

## Query API Requirements

This task must decide:

- how decomposed access is named,
- how mixed decomposed and non-decomposed access works,
- how chunk views are presented,
- how zipped row iteration adapts to decomposed views.

Do not add decomposition without a clear user-facing naming model.

## Benchmark Requirements

Benchmark decomposition against packed storage for:

- simple arithmetic kernels,
- selective single-field writes,
- mixed-field reads,
- wide-struct cache pressure scenarios.

This task only justifies itself if it wins on real workloads.

## Implementation Checklist

1. Add an opt-in decomposition declaration path.
2. Extend metadata and descriptors to enumerate decomposed physical columns.
3. Extend chunk layout code to store them.
4. Extend query descriptors for subcolumn access.
5. Extend scheduler resource conflict logic accordingly.
6. Add benchmarks comparing packed and decomposed layouts.

## Pitfalls

### Pitfall: Making decomposition the new default

That would front-load too much complexity onto ordinary users.

### Pitfall: Allowing decomposition to outpace the query API design

The physical layout and the surface API must stay coherent.

### Pitfall: Assuming finer storage is automatically faster

Extra indirection and more metadata can erase the theoretical win. Benchmark it.

## Done Criteria

This task is done when:

- suitable types can be opt-in decomposed,
- the scheduler can exploit the finer resource granularity,
- the API for decomposed access is coherent,
- benchmarks justify the feature on real workloads.
