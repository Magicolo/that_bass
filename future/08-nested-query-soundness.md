# Nested Query Soundness

This document proposes how `that_bass` could support nested queries, including overlapping and conflicting queries, without:

- violating aliasing rules,
- deadlocking on recursive lock acquisition,
- or silently producing incoherent results.

The motivating problem is the one already hinted at in the current design notes:

```rust
let query1: Write<Position> = ...;
let query2: Read<Position> = ...;

for item1 in query1 {
    for item2 in query2 {
        // problem:
        // - same table / same chunk / same column can deadlock
        // - same row can alias `&mut Position` with `&Position`
        // - even different rows are not currently reborrowed in a way Rust can prove sound
    }
}
```

The current source already recognizes the issue and sketches an incomplete approach in `src/v1/mod.rs:177-196`.

## The Core Problem

There are really three separate problems:

## 1. Lock Ordering Deadlock

Outer query acquires locks for one set of columns.
Inner query tries to acquire overlapping locks recursively.

If the inner query needs:

- the same lock with a stronger mode,
- or a lower-index lock than the outer query still holds,

you can deadlock or be forced into lock drop/reacquire strategies that are extremely hard to make sound.

## 2. Aliasing Unsoundness

Even if locking worked, plain nested references can still be wrong.

Examples:

- outer has `&mut Position` for row `i`, inner produces `&Position` for the same row,
- outer has `&Position` for row `i`, inner produces `&mut Position` for row `i`,
- outer has `&Position` for row `i`, inner produces `&mut Position` for a different row in the same borrowed slice, but the implementation has not actually split the slice to prove disjointness.

This is not a "just use a different lock" issue.
It is a borrow-shape issue.

## 3. Execution Semantics

Even if nested access is made sound, the library still has to define:

- whether the inner query sees the current outer row,
- whether it sees rows moved during the outer query,
- whether same-table conflicting nests are live, snapshot-based, or disallowed.

Without a crisp semantic model, users will write subtle bugs.

## Strong Conclusion Up Front

Arbitrary nested queries should not be implemented as "just run another normal `Query` inside the callback".

That model is too weak.

Instead, `that_bass` should support nested queries through three layers:

## Layer 1: Query Algebra For Common Cases

Most useful nested queries are really one of:

- keyed lookup,
- self-join,
- cartesian product,
- pairwise product,
- "all others except self",
- lookup by projected key,
- grouped lookup.

These should become first-class query operators planned as one execution unit.

## Layer 2: Nest-Aware Direct Execution

For cases that are truly nested and dynamic, the outer query must enter a special nest-aware mode.

In that mode:

- access requirements are merged up front,
- conflicting columns on the current table are upgraded before iteration begins,
- outer items are proxy borrows, not plain raw references,
- inner queries on the current table are served by reborrowing disjoint regions from the outer guard.

## Layer 3: Snapshot Or Scheduler Fallback

If the requested nest cannot be made live and sound:

- run it against a stable snapshot,
- or require the scheduler/query planner to split it into separate phases.

This is better than pretending arbitrary live recursion can always work.

## Proposal A: Query Algebra First

This should be the default answer for nested queries.

## Why

The current TODO list already hints that operations like `Permute` and `Combine` belong in the query model.
That instinct is correct.

Most nested loops are really relational operations:

### Examples

```rust
// lookup
outer.each(|(a, copy, nest)| nest.find(copy.0, ...));
```

becomes:

```rust
query::<(&mut A, &CopyFrom)>()
    .lookup::<&A>(|(_, copy)| copy.0)
    .each(|((a1, _), a2)| a1.0 = a2.0);
```

```rust
// all pairs
query::<&A>().permute::<&A>().each(|(left, right)| ...)
```

```rust
// all pairs except self
query::<&A>().permute_excluding_self::<&A>().each(|(left, right)| ...)
```

```rust
// combinations
query::<&A>().combine::<&A>().each(|(left, right)| ...)
```

### Benefits

- One planner sees both sides.
- One lock plan can be built.
- Self-row exclusion becomes explicit and cheap.
- Same-table aliasing can be handled structurally rather than recursively.
- Deadlock disappears because there is no second live query object recursively locking the world.

### Drawback

- It does not cover every dynamic case.

### Verdict

This should become the preferred API and optimization target.

## Proposal B: Nest-Aware Query Mode

For truly dynamic nesting, introduce a distinct mode:

```rust
database
    .query::<Outer>()
    .nest::<Inner>()
    .each(|(outer, nest)| { ... });
```

But this must not be implemented by storing a normal `Query<Inner>` inside the callback.

It must change the execution model.

## The Rule

`nest::<Inner>()` rewrites the outer query's access plan.

Specifically:

- Any outer read that conflicts with an inner write on the same datum becomes exclusive on the current table.
- Any outer write that conflicts with an inner read or write remains exclusive.
- Missing inner accesses that could require lower-index columns on the current table are pre-acquired in a compatible order.

This is close in spirit to the existing TODO in `src/v1/mod.rs:177-188`, but the key missing piece is the borrow model.

## The Missing Piece: Proxy Borrows

Plain `&T` and `&mut T` are not enough for sound live nesting.

Nest-aware outer iteration must yield proxy values that own the table/chunk borrow from which they came.

Conceptually:

- `OuterRef<'a, T>`
- `OuterMut<'a, T>`

These proxies can still deref to `&T` or `&mut T`, but they also carry:

- the row index,
- the borrowed chunk/table guard,
- the ability to carve out "everything except this row" for the inner query.

Without that ownership, the library cannot soundly reborrow the same column for inner access.

## Same-Table Reborrow Rules

Assume row-granularity for clarity.

### Outer `&T`, Inner `&T`

Safe.

- Share read access.
- Same row may be included or excluded depending on API choice.
- I recommend `exclude_self` by default for self-nests and explicit opt-in to include self.

### Outer `&mut T`, Inner `&T`

Safe only if the inner query on the current table excludes the outer row.

Implementation:

- outer owns `&mut [T]` or equivalent exclusive region,
- split around `row_i`,
- yield `&mut T` for `row_i`,
- inner sees `&[T]` over `[0..i)` and `(i+1..]`.

### Outer `&T`, Inner `&mut T`

Also possible, but only if `nest::<Inner>()` upgraded the outer table access to exclusive before iteration.

Implementation:

- exclusive guard is owned by the outer proxy even though the visible outer item is just `&T`,
- split around `row_i`,
- yield `&T` for `row_i`,
- inner gets mutable access only to the complementary rows.

### Outer `&mut T`, Inner `&mut T`

Possible for different rows only.

Implementation:

- exclusive outer guard,
- split around `row_i`,
- inner gets mutable access to residual rows only.

### Same Row

Never legal for conflicting modes.

The API should exclude the current outer row automatically whenever:

- the inner query could produce the same table/row pair,
- and any overlapping datum has at least one mutable side.

That needs to be structural, not user-enforced.

## Chunk-Grain Warning

This proposal is easiest and cleanest at row granularity.

If the outer query yields whole conflicting chunks:

- outer already exposes a borrow for the whole chunk,
- inner conflicting access to that same chunk cannot be made live without more subdivision or snapshotting.

Recommendation:

- support live conflicting nesting only for row-mode nests at first,
- for chunk-mode nests, either:
  - restrict to non-conflicting nests,
  - or use snapshot semantics.

## How Inner Execution Should Work

The inner query should conceptually have two paths:

## Path 1: Other Tables / Other Chunks

Run like a normal planned query.

- No special same-table alias concerns.
- Still obey canonical table/chunk ordering.

## Path 2: Current Outer Table / Current Outer Chunk

Do not lock recursively.
Do not call normal `Query::each`.

Instead:

- reuse the already-owned guard,
- reborrow complementary ranges,
- run a specialized local iterator over the current table/chunk.

This is why `nest` needs a dedicated runtime object, not a second normal query handle.

## Proposal C: Make Nesting A Planner Feature

An even stronger direction is to treat `nest::<Inner>()` as a planner transform rather than an iterator trick.

The planner can classify the nest as one of:

### Case 1: Lookup Nest

Use keyed/indexed lookup.

### Case 2: Same-Table Self Nest

Use one guard plus split/reborrow iteration.

### Case 3: Cross-Table Nest

Use regular nested chunk/table plan.

### Case 4: Impossible Live Nest

Fall back to:

- snapshot inner,
- deferred second pass,
- or hard error with a diagnostic suggesting a relational operator.

This is the best version, because it makes the library honest about what kind of nest is being executed.

## Snapshot Fallback

There will be cases where live semantics are too expensive or too complex.

Example:

- outer chunk-level mutable query,
- inner conflicting same-chunk query,
- user insists on a direct nested API.

In those cases, the engine can offer:

```rust
query.nest_snapshot::<Inner>()
```

Semantics:

- inner sees a stable snapshot taken at the start of the outer scope, table, chunk, or frame,
- no deadlock,
- no aliasing violation,
- but not live visibility.

This is powerful and honest.

## Proposal D: Scratch-Chunk Writes And Delayed Commit

This is the more radical idea:

- treat writes as copy-on-write at chunk granularity,
- let writers read the shared chunk under a read lock,
- copy the chunk into a temporary writable scratch chunk,
- hand out `&mut [T]` into the scratch copy instead of into shared storage,
- reconcile scratch chunks back into shared storage later.

This idea becomes much more plausible if chunk size stays small and bounded, for example:

- 64,
- 128,
- or at most 256 rows.

At that size, "copy the whole chunk" stops sounding absurd and starts sounding like a serious execution strategy.

## The Core Mechanism

For a writable query over one chunk:

1. Take a read lock on the shared chunk.
2. Copy the relevant columns, or perhaps the whole chunk projection, into a thread-local scratch chunk.
3. Release the shared chunk read lock.
4. Execute user code against `&mut [T]` or row references backed by the scratch chunk.
5. At chunk release, attempt to commit the scratch chunk back to shared storage.

This immediately side-steps the two nastiest live problems:

- no aliasing unsoundness from multiple `&mut [T]` into the same shared memory,
- no deadlock from recursive lock acquisition on the same shared chunk.

The library regains full control over commit semantics.

## Why It Is Attractive

## 1. Soundness Becomes Much Easier

If every writer mutates a private copy:

- overlapping `&mut [T]` no longer alias in memory,
- nested writes to the same chunk are mechanically safe,
- recursive query execution stops being a borrowing problem and becomes a commit problem.

That is a big simplification.

## 2. Deadlocks Mostly Disappear

Shared storage is only read-locked during copy-in.
The expensive user work happens without holding chunk locks.

Commit still needs coordination, but:

- it happens at chunk boundaries,
- can be retried,
- and does not require holding overlapping live borrows while waiting.

## 3. Chunk-Local Parallelism Gets Easier

If many systems or nested queries want to mutate the same archetype:

- they can all fork scratch chunks in parallel,
- then commit with a clear policy.

This is conceptually closer to:

- software transactional memory,
- chunk-level MVCC,
- or "mutable snapshot plus merge".

## 4. API Can Stay Familiar

The visible API can remain pleasantly ordinary:

```rust
query::<&mut Position>().chunk().each(|positions| {
    for p in positions {
        p.x += 1.0;
    }
});
```

Internally, `positions` may be backed by a scratch chunk instead of shared storage.

That is much easier to explain than proxy borrow gymnastics.

## What Exactly Gets Copied?

There are several options.

## Option 1: Copy The Entire Chunk

Every column, every row in the chunk.

### Good

- Very simple mental model.
- Commit can replace the whole chunk image.

### Bad

- Potentially too much bandwidth when the query only touches one column.

## Option 2: Copy Only The Projected Columns

Only columns visible to the query are copied into scratch.

### Good

- Lower bandwidth.

### Bad

- Commit logic becomes column-wise.
- Harder if later logic wants extra columns.

## Option 3: Copy Whole Rows But Only For The Touched Region

This is attractive only for row-subrange nests or very dynamic projections.

### Verdict

For a first experiment:

- copy only the writable projected columns,
- and maybe share readonly columns directly.

That captures most of the gain without doing unnecessary work.

## Commit Policy Options

This is the real heart of the design.

Once two or more scratch writers exist for the same shared chunk, what wins?

## Policy A: First Commit Wins

The first writer that reaches commit succeeds.
Later writers fail with a conflict.

### Good

- Simple.
- Honest.
- No silent overwrite.

### Bad

- Can create retries or failed work in hot loops.
- Potentially surprising if conflicts are frequent.

## Policy B: Last Commit Wins

Commits are serialized, and the last one replaces prior writes on overlapping data.

### Good

- Simple runtime model.

### Bad

- Very surprising semantics.
- Easy to hide bugs.
- Dangerous for games unless extremely explicit.

### Verdict

Do not make this the default.

## Policy C: Per-Column Or Per-Row Version Check

Commit succeeds only if the base version seen during copy-in still matches the shared version for the modified region.

### Good

- Closer to optimistic concurrency control.
- Detects true conflicts.

### Bad

- More metadata.
- More expensive commit path.

## Policy D: User-Specified Merge

The query or system provides a merge rule:

```rust
query::<&mut Velocity>()
    .scratch_merge(|base, left, right| { ... });
```

### Good

- Extremely powerful for additive or commutative updates.

### Bad

- Very complex.
- Hard to make ergonomic.
- Easy to misuse.

## Policy E: Restrict To Commutative Write Classes

Instead of arbitrary writes, only support certain staged operations in scratch mode:

- assignment,
- add,
- max,
- min,
- bit-or,
- custom reduction.

Now conflicting commits become a reduction problem, not an overwrite problem.

### Good

- Much more understandable.
- Can be massively parallel.

### Bad

- Not a general writable query anymore.

### Verdict

This is extremely interesting for numeric kernels.

## Best Semantic Framing

If the library adopts scratch-chunk writes, it should not describe them as "ordinary mutable access".

It should describe them as one of:

- transactional mutable chunk access,
- staged mutable chunk access,
- speculative mutable chunk access,
- or reduction-style chunk access.

That framing matters because users need to understand that:

- they are not mutating canonical state directly,
- commit can fail,
- commit can be retried,
- or commit semantics may depend on a policy.

## Cost Model

The obvious question is:

> is copying chunks in and out too expensive?

The answer depends heavily on:

- chunk size,
- number of copied columns,
- mutation density,
- CPU bandwidth,
- whether copy-in and compute overlap in parallel,
- whether commit writes whole cache-friendly slices.

## Why It May Be More Viable Than It Sounds

- Chunks are small.
- Copies are contiguous and parallelizable.
- Many hot game systems are bandwidth-heavy anyway.
- Eliminating lock contention and recursive borrow complexity can pay back a lot.
- Scratch chunks can be thread-local and reused across iterations.

## Why It May Still Be Too Expensive

- If every frame copies massive hot chunks multiple times, memory bandwidth may dominate.
- If many writes end up conflicting and failing commit, the wasted work is severe.
- If only a few values in a chunk are actually mutated, whole-chunk copy may be wasteful.

## Best Fit Scenarios

This design looks strongest for:

- chunk-mode kernels,
- nested conflicting writes on small chunks,
- optimistic parallel execution of AI or physics micro-solvers,
- workloads where conflicts are rare or structured,
- reduction-style updates,
- scheduler-controlled phases where scratch outputs are merged deterministically.

It looks weaker for:

- frequent arbitrary conflicting writes to the same hot chunks,
- giant chunks,
- logic that assumes immediate and unique mutation of canonical state.

## Understandability Risk

This is the biggest product risk.

If users write:

```rust
for a in query1 {
    for b in query2 {
        b.x += a.y;
    }
}
```

they will naturally assume that `b` is the real current state.

If `b` is actually a scratch copy whose commit may:

- fail,
- overwrite another result,
- or be merged later,

the semantics become non-obvious.

That does not make the design bad.
It means the design needs a different API surface and vocabulary than plain live `&mut`.

## Better API Shapes For This Design

Instead of pretending it is ordinary `&mut`, use explicit staging APIs:

```rust
query::<&mut Position>()
    .scratch()
    .on_conflict(Error)
    .each(|positions| { ... });
```

or:

```rust
query::<&mut Velocity>()
    .reduce(Add)
    .each(|velocities| { ... });
```

or:

```rust
query::<&mut Position>()
    .transactional()
    .each(|positions| { ... });
```

This makes users opt into the semantic tradeoff.

## Relationship To The Other Proposals

This is not a replacement for query algebra.

It is best understood as an alternative implementation strategy for the hardest cases:

- conflicting same-chunk mutable nests,
- chunk-level parallel write kernels,
- perhaps scheduler-controlled optimistic phases.

Compared to reborrow-based live nesting:

- it is simpler for soundness,
- potentially simpler for implementation,
- but more expensive and semantically more surprising.

Compared to pure snapshot nesting:

- it preserves a path to committing writes,
- but now needs an explicit reconciliation model.

## Recommended Position

Treat scratch-chunk writes as a serious advanced mode, not the default foundation.

My recommendation:

1. Use relational operators first.
2. Use reborrow-based live nesting for row-granular same-table cases where semantics are intuitive.
3. Explore scratch-chunk writes as:
   - an opt-in chunk-mode execution strategy,
   - a scheduler/internal optimization,
   - or a specialized API for transactional/reduction workloads.

The most promising sub-variant is:

- small fixed-size chunks,
- projected writable columns copied to thread-local scratch,
- optimistic commit with version checks,
- explicit conflict policy,
- and reduction-oriented modes for commutative updates.

## Scheduler-Based Answer

Once the scheduler-first design exists, it should become the preferred solution for many nested cases.

Instead of:

```rust
for outer in query1 {
    for inner in query2 {
        ...
    }
}
```

users often really want one of:

- a join operator,
- a lookup operator,
- a pairwise operator,
- or two scheduled phases with a temporary table.

The scheduler should encourage those shapes because:

- they are easier to parallelize,
- they avoid recursive live borrowing,
- they produce more predictable performance.

## Recommended API Surface

I would add four concepts, in this order:

## 1. Relational Query Operators

- `lookup`
- `permute`
- `combine`
- `excluding_self`
- maybe `group_join`

These should handle most useful nests.

## 2. `nest::<Inner>()` As A Special Query Mode

But only with:

- pre-merged access sets,
- proxy outer items,
- same-table row exclusion,
- reborrowed complementary ranges.

Do not expose it as simple recursive `Query`.

## 3. Explicit Snapshot Nesting

- `nest_snapshot::<Inner>()`

This gives users a safe "I know I want nested logic, but live coherence is not required" tool.

## 4. Planner Diagnostics

When a requested live nest is impossible or too costly, return a structured error or compile-time refusal that says something like:

- conflicting same-chunk mutable nest requires snapshot or relational operator,
- use `.lookup`, `.combine`, `.permute`, or `.nest_snapshot`.

## What This Means For The Current API

The present `Row` model that yields plain references is good for simple queries, but insufficient for fully general live nested conflicting queries.

That means one of two things has to happen:

## Option A

Nest-aware queries yield proxy items, while plain queries keep today's references.

This is my recommendation.

## Option B

All queries move to proxy items.

This is more uniform, but probably too disruptive and verbose.

## Recommendation

Keep plain refs for normal queries.
Introduce proxies only in the nest-aware mode.

## Final Recommendation

The best answer is a hybrid:

1. Build query algebra first so most nested user intent becomes one planned query.
2. Add a dedicated `nest::<Inner>()` mode that:
   - merges/upsizes accesses up front,
   - yields proxy outer borrows,
   - reuses the same table/chunk guard,
   - excludes the outer row on conflicting same-table nests,
   - never recursively acquires overlapping locks for the current table.
3. Add `nest_snapshot::<Inner>()` for the cases that are not worth making live.
4. Explore scratch-chunk transactional writes for chunk-oriented conflicting nests and reduction workloads.
5. Let the future scheduler steer users toward joins, lookups, multi-phase plans, or staged scratch-commit execution instead of arbitrary recursive loops.

## Strong Conclusion

Supporting arbitrary nested queries elegantly is possible, but not by layering one ordinary query on top of another.

The right design is:

- planner-first for common nested patterns,
- reborrow-based for live same-table nesting,
- scratch-commit as an advanced chunk-oriented alternative,
- snapshot or scheduler fallback for the rest.

That gives `that_bass` a path to powerful nested queries without sacrificing soundness or turning lock ordering into an unsalvageable mess.
