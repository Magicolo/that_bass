# Task 02: Chunk Sizing, Growth, And Single-Allocation Memory Layout

This task turns the metadata model from Task 01 into concrete storage layout rules.

Read this file together with `future/plan/specification.md` and `future/plan/standards.md`. This task is about how big chunks are, how they grow, how their bytes are laid out, and what invariants chunk memory must satisfy.

## Purpose

Design and implement chunk allocation as a first-class storage primitive.

The rewrite depends on chunks for:

- cache locality,
- scheduler job granularity,
- dense batched access,
- bounded structural work,
- future scratch-chunk experiments.

## Required Reading

- `future/plan/specification.md`
- `future/plan/standards.md`
- `future/03-chunking-locks-and-query-plans.md`
- `AGENTS.md`
- `src/v1/table.rs`
- `src/v1/modify.rs`
- `src/v1/destroy.rs`

## Selected Chunk Capacity Rule

Chunk capacity is derived from row width and a tunable byte budget.

Definitions:

```text
inline_row_bytes = sum(inline physical column sizes for one row)
raw_target_rows = floor(target_chunk_bytes / max(1, inline_row_bytes))
target_chunk_rows = max(1, previous_power_of_two(raw_target_rows))
```

Design implications:

- capacities are per-table,
- capacities are powers of two,
- `target_chunk_bytes` must be centralized and configurable,
- capacity should not be hardcoded deep in storage code,
- benchmarking different chunk byte targets must be trivial.

## Selected Growth Policy

The first chunks grow geometrically:

- 1,
- 2,
- 4,
- 8,
- ...

until the table's derived full chunk capacity is reached.

After that:

- new chunks allocate directly at the full target capacity.

This policy avoids waste for tiny tables without giving up a stable full-size chunk model for hot tables.

## Single Allocation Per Chunk

Each chunk uses one allocation.

Suggested memory shape:

```text
[ChunkHeader][padding][Column0 region][padding][Column1 region]...[ColumnN region]
```

The allocator should precompute:

- each region's offset,
- each region's alignment padding,
- the total allocation layout.

The chunk object then stores:

- pointer to base allocation,
- count,
- capacity,
- descriptor reference,
- maybe chunk-local generation counters,
- optional scratch bookkeeping later.

## Why One Allocation Was Chosen

Compared to one allocation per column, one allocation per chunk gives:

- fewer allocator calls,
- better TLB behavior,
- tighter ownership,
- easier recycling,
- easier future scratch-copy or full-chunk clone paths.

The tradeoff is:

- chunk allocation code becomes more careful about offset and alignment math.

That is a good trade for this project.

## Alignment Rules

Each physical column region must satisfy its own alignment requirements.

The implementation must:

- compute the offset for every column using `Layout` rules,
- keep region starts aligned,
- avoid hand-rolled pointer math without tests.

Add tests that validate:

- offsets are monotonic,
- offsets satisfy alignments,
- the total layout is valid for every supported row-width pattern.

## Dense Count Invariant

Every chunk stores rows densely from `0..count`.

There are no holes.

Consequences:

- queries can always expose `&[T]` or `&mut [T]` over the inhabited prefix,
- inserts append at `count`,
- remove uses `swap_remove`,
- partial occupancy is represented only by `count`, not by free lists.

## Pseudo-Code For Insert And Remove

Append:

```rust
let row = chunk.count;
write_all_columns(row, value);
chunk.count += 1;
```

Remove:

```rust
let last = chunk.count - 1;
if row != last {
    move_row(last, row);
}
drop_row(last);
chunk.count -= 1;
```

The exact copy/drop rules depend on datum traits and sidecar storage, but the dense invariant does not.

## Chunk Header Fields

The first implementation should keep the chunk header minimal and explicit.

Recommended fields:

- `count`,
- `capacity`,
- `chunk_index`,
- descriptor reference or pointer,
- optional generation/version counters for debugging and future planning.

Do not overload the chunk header with scheduler state in this task. Scheduler runtime metadata can live beside storage.

## Tuning Requirements

`target_chunk_bytes` must be easy to change.

That means:

- one obvious configuration path,
- benchmark code can sweep it without patching core storage,
- debug output can print the derived capacity for each table.

Examples of useful diagnostic output:

```text
table=Transform inline_row_bytes=32 target_chunk_bytes=32768 target_rows=1024
table=PhysicsSettings inline_row_bytes=128 target_chunk_bytes=32768 target_rows=256
```

## Heavy Or Heap-Rich Datums

This task should assume:

- inline physical column width is whatever the column actually stores inline.

That means:

- a `Vec<T>` field stored inline contributes pointer-sized inline width, not pointee size,
- cold/blob side storage is a later optimization,
- the capacity formula uses actual inline bytes, not logical payload size.

This is one reason the formula is based on physical row width.

## Implementation Checklist

1. Implement chunk-capacity derivation from descriptor plus config.
2. Implement growth from bootstrap chunk sizes to full chunk size.
3. Implement layout computation for single-allocation chunks.
4. Implement aligned column pointer lookup.
5. Implement dense inhabited-prefix accessors.
6. Add tests for layout correctness.
7. Add benchmarks that sweep chunk byte budgets and row widths.

## Pitfalls

### Pitfall: Encoding a hidden cap like `256` in a convenience type

Do not let old assumptions leak back in. Table-specific row/chunk bit partitioning exists specifically so chunk capacity can be byte-budget-driven.

### Pitfall: Using average column size

Capacity must be derived from full inline row width, not average type size.

### Pitfall: Mixing scheduler ownership into chunk memory too early

Keep storage layout separate from runtime job state.

## Done Criteria

This task is done when:

- chunk capacity is derived from row width and configuration,
- chunks allocate as one aligned block,
- dense prefix slice access is possible,
- growth behavior is tested,
- chunk byte targets can be benchmarked by changing configuration instead of patching storage internals.
