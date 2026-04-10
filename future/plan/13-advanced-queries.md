# Task 13: Relational Query Operators, Deferred Value Writes, And Advanced Nested Access

This task extends the MVP query model into the more ambitious territory discussed in the proposals.

Read this file together with `future/plan/specification.md`, `future/plan/standards.md`, and `future/08-nested-query-soundness.md`. The MVP deliberately rejects or defers some overlapping live-query patterns. This task is where the engine grows beyond that conservative baseline.

## Purpose

Add safe answers for patterns that today would be rejected or awkward:

- relational query operators,
- deferred non-structural writes such as `Set<T>`,
- advanced nested-query handling,
- optional scratch-chunk transactional experiments.

## Required Reading

- `future/plan/specification.md`
- `future/plan/standards.md`
- `future/08-nested-query-soundness.md`
- `AGENTS.md`

## Recommended Order Inside This Task

Do not implement the wildest idea first.

Recommended progression:

1. relational operators,
2. deferred value writes,
3. limited nest-aware direct execution,
4. scratch-chunk transactional mode experiments.

That order preserves the bias toward simpler and more explainable semantics.

## Relational Operators First

Most nested loops are really query algebra problems.

Priority operators:

- `lookup`
- `combine`
- `permute`
- `exclude_self`

Why first:

- one planner sees both sides,
- same-row exclusion can be structural,
- access analysis stays centralized,
- users get a powerful answer without manual nested query hazards.

## Deferred Value Writes

Some patterns are not best expressed as live mutable overlap.

Examples:

- read one chunk,
- compute updates for another or the same logical datum,
- apply those updates later through an explicit command buffer.

Possible API directions:

- `Set<T>`
- `Patch<T>`
- `WriteLater<T>`

This is the extension that can make patterns like:

- `Read<T>` plus later writes to `T`,

sound and ergonomic without needing full live nested mutability.

## Limited Nest-Aware Direct Execution

If live nesting is still desirable after relational operators and deferred writes exist, it should be introduced narrowly.

Likely limits:

- only for planner-classified safe shapes,
- same-row exclusion by construction,
- specialized proxy or reborrow machinery where needed,
- clear diagnostics when the request cannot be made live and sound.

Do not ship arbitrary "query inside query" recursion as if it were ordinary iteration.

## Scratch-Chunk Transactional Mode

This remains the most radical option in the discussed design space.

The essential shape is:

1. read shared chunk data,
2. copy writable projection into scratch storage,
3. mutate scratch slices,
4. reconcile on commit.

Potential policies:

- first commit wins,
- last commit wins,
- version-checked optimistic commit,
- user merge,
- reduction-only modes.

This should be treated as an experiment until benchmarked and clearly explained.

It is promising, but it changes the semantics of "write" enough that it should not quietly replace ordinary chunk writes.

## Diagnostics Matter

When the planner rejects a live overlap, the engine should be able to say something useful such as:

- use `lookup`,
- use `exclude_self`,
- split into two scheduled functions,
- use deferred `Set<T>`,
- this pattern currently requires a snapshot.

That guidance is part of the feature, not a nice-to-have.

## Benchmark Requirements

This task should benchmark at least:

- classic nested self-joins,
- all-pairs kernels,
- deferred-value-write alternatives,
- scratch-chunk copy and commit overhead,
- conflict-heavy versus conflict-light workloads.

Without these benchmarks, the advanced query extensions will be impossible to compare fairly.

## Implementation Checklist

1. Add one relational operator at a time.
2. Add planner diagnostics that recommend them.
3. Prototype deferred value writes and their resolve path.
4. Define which live nested cases can be supported honestly.
5. If scratch mode is attempted, benchmark it before committing to public semantics.
6. Add tests for:
   - self-exclusion correctness,
   - overlap rejection diagnostics,
   - deferred-write visibility,
   - any transactional conflict policy that is prototyped.

## Pitfalls

### Pitfall: Solving everything with the fanciest mechanism

Prefer the simplest feature that honestly solves the user's problem.

### Pitfall: Presenting transactional scratch writes as ordinary `&mut` semantics

If writes reconcile later, the API must make that explicit.

### Pitfall: Adding live nesting before diagnostics exist

Complex capability without clear failure guidance will be hostile to users.

## Done Criteria

This task is done when:

- overlapping query patterns have at least one principled post-MVP answer,
- relational operators exist for the common cases,
- deferred value writes are properly integrated or explicitly rejected,
- any live nested mode is honest about what it can and cannot do.
