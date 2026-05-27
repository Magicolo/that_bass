# that_bass v4: Columnar Store Experiment

This directory contains a minimal, unsafe, type-erased columnar storage engine
extracted from the inline `store` module in `src/lib.rs`. It explores the
design space of:

- **Schema-defined tables** identified by exact `TypeId` sets.
- **Deferred operations** (insert/resolve, remove/resolve).
- **Manual memory layout** through `alloc`/`dealloc` with packed column
  allocations.
- **Type-erased column access** via `Meta` + function pointer vtables.
- **Composable `Module` abstraction** with lazy state discovery and
  builder-style construction.

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
| `Error` | `error.rs` | All possible error conditions. |

### The Module System

The `Module` trait is the central abstraction — it generalizes queries, inserts,
removes, and future operations into a single composable interface. Modules are
built up layer by layer via the `Push` trait and executed through a `State`
handle that lazily calls `update()` on demand.

| Trait | File | Role |
|-------|------|------|
| `module::Module` | `module.rs` | Core trait: `Item<'_>`, `State`, `initialize`, `update`, `get`. Blanket impls for `&M`, `&mut M`, `()`, and nested tuples. |
| `state::Module<M>` | `state.rs` | Builder wrapper around a `Module`. Constructed via `State::build()` and composed with `.push(module)`. Uses `ref_cast` for safe tuple splitting of composed modules. |
| `state::State<'a, M>` | `state.rs` | Runtime handle holding store, module, and initialized state. Provides `get()` (auto-updates) and `next()` for tuple-walking. |
| `state::Rest<'a, M>` | `state.rs` | Chained accessor produced by `State::next()` and `Rest::next()`. Walks a tuple of modules one element at a time. |

#### Building and Using Modules

```rust
// Compose multiple query modules
let mut state = store.state(
    State::build()
        .push(Query::build().read::<Position>().write::<Velocity>())
        .push(Query::build().read::<char>())
        .push(Query::build().read::<i32>()),
)?;

// Walk the chain: each next() yields one module's Item and the Rest
let (physics, rest) = state.next();
let (chars, rest) = rest.next();
let (ints, _)     = rest.next();
```

or use the simpler closure-based API:

```rust
store.with(query_module, |item| {
    // item is the composed Module::Item<'_>
})?;
```

### Query System

The query system is built on two layers.

| Trait | File | Role |
|-------|------|------|
| `query::Access` | `query.rs` | Per-table access primitive. `initialize` binds to a table, `get` returns typed slices. Implemented for `Read<T>`, `Write<T>`, `Row`, `Table`, `ReadWith(Meta)`. |
| `query::Module<A>` | `query.rs` | Wraps an `Access` impl as a `module::Module`. On `update`, lazily discovers matching tables. On `get`, returns a `Query<A>` handle. |
| `query::Query<'a, Q>` | `query.rs` | Snapshot of matched `(table_index, state)` pairs plus a reference to the store's tables. Provides `.iter()`. |

```rust
// Build a query that reads position and writes velocity
let module = Query::build().read::<Position>().write::<Velocity>();
// .read::<T>() and .write::<T>() push Read<T> / Write<T> onto the Access chain
```

### Operations

| Operation | File | Description |
|-----------|------|-------------|
| `Store::insert(template)` | `insert.rs` | Create or find a table matching the template's schema, return an `Insert` handle. |
| `Insert::one(item)` | `insert.rs` | Reserve a row slot and buffer the item data. |
| `Insert::resolve()` | `insert.rs` | Flush buffered data into the table's columns and commit rows. |
| `Store::remove()` | `remove.rs` | Create a `Remove` handle for collecting rows to delete. |
| `Remove::one(row)` | `remove.rs` | Queue a row for removal. |
| `Remove::resolve()` | `remove.rs` | Compact all queued rows out of their tables. |
| `Store::with(module, fn)` | `module.rs` | Initialize a module, call `get`, and pass the item to a closure. |
| `Store::state(module)` | `state.rs` | Create a `State` handle for a composed `Module`. |

### Utility Types and Traits (utility.rs)

| Name | Role |
|------|------|
| `At<'a, T>` / `AtMut<'a, T>` | Index + reference pair for table/column lookups. |
| `IntoNest` | Convert from flat tuples `(A, B, C)` to nested form `(A, (B, (C, ())))`. Macro-generated for up to 16 elements. |
| `IntoFlat` | Convert from nested form `(A, (B, (C, ())))` back to flat tuples `(A, B, C)`. Macro-generated for up to 16 elements. |
| `Push<T>` | Cons-style prepend trait used to build nested chains. `().push(a).push(b)` produces `(a, (b, ()))`. |
| `PushTail<N>` | Append one nested chain to the tail of another. |
| `Next` | Sequential access to tuple elements; each `next()` yields the head and the tail (`Rest`). |
| `allocate` / `deallocate` | Wrappers over `std::alloc::{alloc, dealloc}`. |
| `ranges` | Groups sorted `(table, row)` pairs into contiguous `Range<u32>`. |
| `resize` | Reallocates the packed multi-column buffer for a table. |
| `find` | Linear or binary search by key on a slice. |
| `vec_as_slice` / `box_as_slice` | Stub functions (`todo!()`) for raw pointer-to-slice views of `Vec` and `Box<[T]>`. |

### Memory Model

Columns within a table share one allocation. The `resize` function (in
`utility.rs`) lays out columns sequentially in one buffer, computing offsets
via `Layout::extend`. Reallocating one column's capacity may relocate all
columns. The `Template::resolve` method now takes `&Table` (immutable),
reflecting that column data pointers remain stable during deferred insert
flushing.

## Key Design Decisions

- **`Module` over ad-hoc traits.** Queries, inserts, removes, and future
  operations all implement the same `Module` trait. This enables uniform
  composition, state management, and lazy update behavior.
- **Lazy table discovery.** `query::Module::update` scans the store's tables
  incrementally, caching only matching ones. `State::get()` calls `update()`
  until no new tables are found, so consumers never see stale data.
- **`Push` for builder ergonomics.** `State::build().push(module_a).push(module_b)`
  compiles to a nested tuple `Module<(A, (B, ()))>` that the `State`/`Rest`
  machinery walks via `next()`.
- **`ref_cast` for tuple splitting.** `state::Module<(H, T)>` uses
  `RefCast::ref_cast` to safely borrow `H` and `T` independently from a shared
  reference, avoiding the need for unsafe transmutes on the tuple itself.
- **Immutable column access during resolve.** `Template::resolve` takes `&Table`
  because columns share one allocation and their `data` pointer is stable;
  writes go through raw pointers on the `Column`, not through `&mut Table`.

## Why v4?

v1 is the current stable engine with generational keys and concurrent queries.
v2 is a rewrite with chunk-based storage and a scheduler. v3 explores
fragment-based storage with gaps.

v4 takes a different angle: **absolute minimalism**. No chunks, no fragments,
no generational keys — just tables of typed columns with manual memory control.
It serves as a design laboratory for the core storage primitives and the
`Module` composition model that higher-level engines may build on.

## Status

**Experimental.** The public API is unstable and likely to change. The code
compiles but there are known gaps:

- `vec_as_slice`, `box_as_slice`, `vec_as_slice_mut`, `box_as_slice_mut` are
  `todo!()` stubs.
- `Store::with` is unsafe for composed modules because validation happens
  per-module rather than holistically (e.g., two `Write<T>` modules can alias).
- `write_with` is commented out.
- `Defer` path in `Insert::one` is `todo!()`.
- No integration test suite.
