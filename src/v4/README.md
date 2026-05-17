# that_bass v4: Columnar Store Experiment

This directory contains a minimal, unsafe, type-erased columnar storage engine
extracted from the inline `store` module in `src/lib.rs`. It explores the
design space of:

- **Schema-defined tables** identified by exact `TypeId` sets.
- **Deferred operations** (insert/resolve, remove/resolve).
- **Manual memory layout** through `alloc`/`dealloc` with packed column
  allocations.
- **Type-erased column access** via `Meta` + function pointer vtables.
- **Template/Query traits** as the user-facing abstraction.

## Architecture

### Core Types

| Type | File | Role |
|------|------|------|
| `Store` | `mod.rs` | Collection of tables identified by their column schema. |
| `Table` | `table.rs` | Column collection with deferred row lifecycle (reserve/ensure/commit/release). |
| `Column` | `column.rs` | Typed or type-erased contiguous buffer for one data type. |
| `Meta` | `meta.rs` | Runtime type metadata: `TypeId`, `size`, `needs_drop`, and a `Functions` vtable for layout/drop/get/set. |
| `Vector` | `vector.rs` | Growable type-erased buffer used internally for deferred insert buffering. |
| `Row` / `Rows` | `row.rs` | Row identifiers and iterators for query results. |
| `At<'a, T>` | `row.rs` | Index + reference pair used for table/column lookups. |
| `Error` | `error.rs` | All possible error conditions. |

### Traits

| Trait | File | Role |
|-------|------|------|
| `Template` | `template.rs` | Describes a row schema for insertion. Supports tuples up to 8 columns. |
| `Query` | `query.rs` | Describes a read access pattern. Implemented for `Read<T>`, `Row`, `Table`, `Column`. |

### Operations

| Operation | Description |
|-----------|-------------|
| `Store::insert(template)` | Create or find a table matching the template's schema, return an `Insert` handle. |
| `Insert::one(item)` | Reserve a row slot and buffer the item data. |
| `Insert::resolve()` | Flush buffered data into the table's columns and commit rows. |
| `Store::query(query)` | Create a `Query` handle that iterates matched tables. |
| `Store::remove()` | Create a `Remove` handle for collecting rows to delete. |
| `Remove::one(row)` | Queue a row for removal. |
| `Remove::resolve()` | Compact all queued rows out of their tables. |

### Memory Model

Columns within a table share one allocation. The `resize` function (in
`utility.rs`) lays out columns sequentially in one buffer, computing offsets
via `Layout::extend`. Reallocating one column's capacity may relocate all
columns.

### Utility Functions (utility.rs)

- `sort` — Sorts and deduplicates `Meta` by `TypeId`.
- `allocate` / `deallocate` — Wrappers over `std::alloc::{alloc, dealloc}`.
- `ranges` — Groups sorted `(table, row)` pairs into contiguous `Range<u32>`.
- `resize` — Reallocates a packed multi-column buffer.
- `find` — Linear or binary search by key on a slice.

## Why v4?

v1 is the current stable engine with generational keys and concurrent queries.
v2 is a rewrite with chunk-based storage and a scheduler. v3 explores
fragment-based storage with gaps.

v4 takes a different angle: **absolute minimalism**. No chunks, no fragments,
no generational keys — just tables of typed columns with manual memory control.
It serves as a design laboratory for the core storage primitives that the
higher-level engines build on.

## Status

**Experimental.** The public API is unstable and likely to change. The code
compiles but the test suite (`tests/store.rs`) contains pre-existing
non-compiling tests that are not yet addressed.
