# v4 TODO — Ordered Implementation Plan

This document lists every known gap, bug, and missing feature in `src/v4/`,
ordered by dependency and risk. Each item contains enough detail to be
picked up without prior context of the codebase.

---

### 20. Implement `Modify` (cross-table moves)

- **File:** New `src/v4/modify.rs`
- **What:** v1's `Modify<Add, Remove>` moves rows between tables by applying
  column additions/removals. v4 needs equivalent capability.
- **Design:** A `Modify` operation that:
  1. Takes a source row and a target template (schema to transform into).
  2. Reserves space in the target table.
  3. Copies shared columns from source to target.
  4. Applies new data to target-only columns.
  5. Removes the source row from the old table.
  All within appropriate column locks on both tables.
- **Verify:** Add columns test — insert `(u32,)`, modify to `(u32, f64)`, verify
  both old and new table counts.

---

## Phase 7 — Key System (Generational Keys)

### 21. Implement the `Keys` system

- **Files:** `src/v4/template.rs:90-104`, `src/v4/item.rs:203-229`, new `src/v4/key.rs`
- **Current state:** `Key` exists as a stub. `Key::initialize` returns `None`.
  `Key` in `item.rs` returns `None` from `initialize`. There is no key
  indirection — rows are identified by raw `(table_index, row_index)`.

- **What's needed:**
  1. A `Keys` storage structure (similar to v1's `src/v1/key.rs`):
     - An array of `Slot`s mapping key index → `(table_index, row_index, generation)`.
     - Generational indices to detect use-after-free.
     - A free list for recycling destroyed key slots.
  2. `Key` type: `(index: u32, generation: u32)`.
  3. `Key::initialize` — bind to a table and set up key storage.
  4. `Key::apply` — during insert resolve, allocate a key slot and set its
     (table, row) mapping.
  5. `Item for Key` — return key handles during queries.
  6. Key recycling in `remove` — when a row is removed, invalidate its key and
     return the slot to the free list.

- **Design decision:** Where does `Keys` live? Options:
  - Per-`Store` keys: one key space for all tables (like v1).
  - Per-table keys: each table has its own key space.
  - The v1 model (per-Store) is more flexible for cross-table moves.

- **Verify:**
  - Insert rows, get keys, query by key, verify correct data.
  - Remove a keyed row, verify the key becomes invalid.
  - Reuse a recycled key slot, verify generation prevents stale access.

---

## Phase 8 — Event System

### 22. Implement event emission and listeners

- **Files:** New `src/v4/event.rs`
- **What:** v1 has a full event system (`src/v1/event.rs`) with `Create`,
  `Destroy`, `Modify` event families, buffering, listener lifecycle, and
  retention policies. v4 has none.

- **Minimum viable design:**
  1. An `Events` registry on `Store` (or a separate `Events` handle).
  2. `Event` enum: `Created(Table, Range<u32>)`, `Destroyed(Table, Vec<(u32, u32)>)`.
  3. `Listener` trait with `on_create`/`on_destroy` callbacks.
  4. Buffered emission: events are queued during `resolve()` and dispatched
     after locks are released.
  5. Retention policy: `Keep::All` (keep until all listeners observe),
     `Keep::None` (drop after dispatch).

- **Integration points:**
  - `Insert::resolve()` emits `Created` events.
  - `Remove::resolve()` emits `Destroyed` events.
  - Future `Modify::resolve()` emits modify events.

- **Verify:** Register a listener, insert rows, verify listener received create
  events with correct table and row range.

---

## Phase 9 — Testing

### 23. Unit tests for `Table`

- **File:** New `#[cfg(test)] mod tests` in `src/v4/table.rs`
- **Coverage required:**
  - `Table::new` — creates table with correct column count and all columns dangling.
  - `Table::insert` — single insert, multi insert, zero-count insert.
  - `Table::insert` — overflow behavior (insert beyond capacity triggers resize).
  - `Table::remove` — remove first row, last row, middle row, all rows.
  - `Table::remove` — out-of-bounds removal panics (debug).
  - `Table::column` — find by TypeId, fail for unknown TypeId.
  - `Table::count` / `Table::capacity` — reflect actual state.
  - `Table::resize` — growth doubles capacity, data preserved.
  - `Table::drop` — all column data deallocated.
  - Column `get`/`get_mut` — returns correct slices.
  - Column `set`/`set_at` — writes correct values.
  - Column `copy_at` / `drop_at` — correct element movement.
  - Column `lock`/`unlock` — ZST columns skip locking.

### 24. Unit tests for `Meta`

- **File:** New `#[cfg(test)] mod tests` in `src/v4/meta.rs`
- **Coverage:**
  - `Meta::of::<T>()` is idempotent (same instance returned).
  - Different types get different metas.
  - `size()` returns `size_of::<T>()`.
  - `drops()` returns true for Drop types, false for Copy types.
  - `layout(count)` returns correct layout.
  - `extend` combines layouts correctly.
  - `initialize` copies data between allocations.
  - `resize` grows allocation, preserves data, shrinks correctly.
  - `copy` / `copy_at` / `drop` / `drop_at` correctness.
  - `get` / `get_mut` return correct type-erased references.
  - `set` / `set_at` correctly place values.

### 25. Unit tests for `Query`

- **File:** New `#[cfg(test)] mod tests` in `src/v4/query.rs`
- **Coverage:**
  - Build query with `.read::<T>()` — discovers matching tables.
  - Build query with `.write::<T>()` — discovers matching tables.
  - Build query with `.try_read::<T>()` — optional column access.
  - Build query with `.has::<T>()` / `.not::<T>()` filters — correct table matching.
  - `Query::update` — discovers tables lazily.
  - `Query::tables()` iteration — `Guard` yields correct count, correct slices.
  - `Guard::columns()` — destructured access to typed slices.
  - Guard drop — unlocks columns.
  - Clone a Query — cloned query starts with no states.
  - Conflict detection: read+write on same column in same query is rejected.

### 26. Unit tests for `Insert` and `Remove`

- **File:** New `#[cfg(test)] mod tests` in `src/v4/insert.rs` and `src/v4/remove.rs`
- **Coverage:**
  - Insert single item, verify table count increases.
  - Insert multiple items, verify all present.
  - Insert into table that doesn't exist yet (auto-creates).
  - Remove specific rows, verify table count decreases, remaining rows shift.
  - Remove from empty table (no-op).
  - Insert after remove reuses freed capacity.
  - `Insert::builder().column::<T1>().column::<T2>()` — multi-column insert.
  - Remove rows in descending order (required by swap_remove compaction).

### 27. Unit tests for `Filter`

- **File:** New `#[cfg(test)] mod tests` in `src/v4/filter.rs`
- **Coverage:**
  - `Has<T>` matches table containing T.
  - `Has<T>` rejects table without T.
  - `Not<Has<T>>` inverts.
  - Filter composition: `(Has<A>, Has<B>)` matches only if both present.
  - `()` filter matches everything.
  - `HasWith(meta)` matches by runtime Meta.

### 28. Unit tests for `Depend` / `Analysis`

- **File:** New `#[cfg(test)] mod tests` in `src/v4/depend.rs`
- **Coverage:**
  - Single read dependency passes analysis.
  - Read+Write on same resource is rejected.
  - Write+Write on different resources passes.
  - Ancestor resources (e.g., Column → Table → Tables → Store) propagate
    read access correctly.
  - Multiple composed dependencies are checked holistically.
  - `Error::all` combines multiple errors correctly.

### 29. Unit tests for `Slice`, `Vector`, and utility types

- **Files:** New test modules in `src/v4/slice.rs`, `src/v4/vector.rs`, `src/v4/utility.rs`
- **Coverage:**
  - `Slice::empty` / `Slice::get` / `Slice::get_mut` / `Slice::downcast_ref`.
  - `Vector::push` / `Vector::len` / `Vector::capacity` / `Vector::move_at`.
  - `IntoNest` / `IntoFlat` round-trip for tuples of various sizes.
  - `Push` trait builds correct nested chains.
  - `Defer` runs closure on drop, even on panic.
  - `allocate` handles zero-size layout correctly.
  - `ranges` (if implemented) groups sorted pairs.
  - `find` linear vs binary search threshold.

### 30. Integration tests

- **File:** New `tests/v4/` directory
- **Coverage:**
  - Full workflow: create store → insert → query → modify → remove → verify.
  - Multi-table scenarios: insert into tables with different schemas, query
    across them, filter by schema.
  - Cross-table modify: move rows between schemas.
  - Concurrent insert + query (readers see consistent snapshots).
  - Concurrent insert + remove (no double-free, correct counts).
  - Stress test: N threads each doing M operations, verify final state.

### 31. Miri validation

- **What:** Run `cargo +nightly miri test` on v4 tests. Since v4 uses pervasive
  raw pointer manipulation, Miri is essential for catching UB.
- **Setup:** Add a `miri` configuration in CI. Run on all tests in `src/v4/`
  and `tests/v4/`.
- **Verify:** Zero Miri errors.

### 32. Benchmarks

- **File:** New `benches/v4/` directory
- **Minimum benchmarks:**
  - Insert throughput (rows/sec) for various column counts.
  - Query iteration throughput.
  - Remove throughput (rows/sec).
  - Single-row random access via get/set.
  - Multi-column resize cost.

### 33. Fuzzing / property-based testing

- **What:** Use `proptest` or `arbitrary` to generate random sequences of
  (insert, query, remove) operations and verify invariants:
  - Table count never exceeds capacity.
  - Row count equals insertions minus removals.
  - All rows in query results are valid.
  - No stale data visible after remove.
- **Verify:** At least 10,000 random sequences pass without error.

---

## Phase 10 — API Completeness & Ergonomics

### 34. Implement `Store::with` (simple closure-based API)

- **File:** `src/v4/mod.rs` (on `impl Store`)
- **What:**
  ```rust
  pub fn with<I: Item, F: Filter>(
      &self,
      query: &mut Query<I, F>,
      f: impl FnMut(Guard<'_, I>),
  ) {
      for guard in query.tables() {
          f(guard);
      }
  }
  ```
- **Note:** This is simple iteration sugar. The README describes a more
  sophisticated version that composes multiple modules, which depends on the
  `Module` trait (see item #37).

### 35. Add `len()` / `is_empty()` helpers to `Store`

- **File:** `src/v4/mod.rs`
- **What:**
  ```rust
  pub fn len(&self) -> usize {
      self.tables.0.load().slice.len()
  }
  pub fn is_empty(&self) -> bool {
      self.len() == 0
  }
  ```

### 36. Add typed `.get::<T>()` / `.get_mut::<T>()` to `Guard`

- **File:** `src/v4/query.rs:129-144`
- **What:** Currently `Guard` only provides `.columns()` which returns tuples
  via `IntoFlat`. For single-column queries, add:
  ```rust
  pub fn get<T: 'static>(&self) -> &[T] { ... }
  pub fn get_mut<T: 'static>(&mut self) -> &mut [T] { ... }
  ```
  These would be convenience methods that bypass the `IntoFlat` tuple dance.

### 37. Implement or remove the `Module` / `State` / `Rest` composition system

- **Files:** Described in README but doesn't exist in source.
- **What:** The README describes:
  - A `Module` trait with `Item<'_>`, `State`, `initialize`, `update`, `get`.
  - A `State::build().push(module_a).push(module_b)` builder.
  - A `Rest` type for chained iteration via `next()`.
  - `ref_cast` for safe tuple borrowing of composed modules.

  **Decision needed:** Is this composition model the intended direction?
  - If **yes**: Implement it. This is a significant design and implementation
    effort. Start with a `module.rs` and `state.rs` that implement the described
    semantics.
  - If **no**: Remove references to it from README and AGENTS.md.

  The current tuple-based composition in `Query::Build` and `Insert::Build`
  using `Push` is a simpler alternative that already works. The `Module` trait
  would unify queries, inserts, and removes under one abstraction, enabling
  cross-operation dependency analysis.

### 38. Implement proc-macro derives for v4

- **File:** `that_base_derive/src/lib.rs` (add v4 support)
- **What:** v1 has `#[derive(Template)]` and `#[derive(Filter)]`. v4 needs
  equivalents for its `Template`, `Filter`, and `Item` traits.
- **Design:** The derive macros should emit `that_bass::v4::...` paths (similar
  to how v1 derives emit `that_bass::v1::...` paths). Behind the `v4` feature flag.
- **Minimum:**
  - `#[derive(Template)]` for structs — each field becomes a `Column<T>` in the
    template.
  - `#[derive(Filter)]` for structs and enums — similar to v1's filter derive.
- **Verify:** Compile tests for derives, verify generated code matches
  hand-written impls.

### 39. Add `Datum`-equivalent marker trait

- **File:** `src/v4/meta.rs` or new `src/v4/datum.rs`
- **What:** v1 has `pub unsafe trait Datum: Sized + 'static {}`. v4 has no
  equivalent. A marker trait would:
  - Provide a clear bound for what types can be columns.
  - Allow blanket impls or auto-derives.
  - Serve as documentation of the type contract.

---

## Phase 11 — Documentation

### 40. Add module-level doc comments to every file

- **Files:** All 14 `.rs` files in `src/v4/`
- **What:** Each module should have a `//!` doc comment explaining:
  - What the module provides.
  - Key types and their roles.
  - Important invariants or safety considerations.
- **Example:** `src/v4/meta.rs` should explain the `Meta` vtable pattern, the
  global `METAS` cache, and the intentional memory leak.

### 41. Add doc comments to all public API items

- **What:** Every `pub` item in v4 should have a `///` doc comment with:
  - What it does.
  - Usage example (where non-trivial).
  - Panics (if any).
  - Errors (if fallible).
  - Safety (if unsafe).
- **Priority types:** `Store`, `Table`, `Row`, `Rows`, `Column`, `Query`,
  `Guard`, `Build`, `Insert`, `Remove`, `Filter`, `Has`, `Not`, `Read<T>`,
  `Write<T>`, `Try`, `Template`, `Meta`, `Slice`, `Vector`, `Error`, `Depend`,
  `Dependency`, `Access`, `Resource`.

### 42. Write `examples/v4/main.rs`

- **File:** New `examples/v4/main.rs`
- **What:** A walkthrough example showing the complete v4 workflow:
  1. Create a store.
  2. Build insert templates.
  3. Insert rows.
  4. Build queries.
  5. Iterate and modify data.
  6. Remove rows.
  7. Clean up.
  Follow the style of `examples/v2/main.rs`.

### 43. Update `src/v4/README.md` to match reality

- **File:** `src/v4/README.md`
- **What:** The README describes several things that don't exist:
  - `module.rs`, `state.rs`, `row.rs`, `column.rs` files.
  - `Module`, `State`, `Rest` types.
  - `Store::with`, `Store::state` methods.
  - `vec_as_slice`/`box_as_slice` stubs (not in source).
  - `write_with` commented out (it's active).
  - `Defer` path in `Insert::one` (not in source).

  Either update to accurately describe the current state, or implement the
  described features and then the README becomes correct.

### 44. Update `AGENTS.md` v4 section

- **File:** `AGENTS.md:52`
- **What:** The AGENTS.md description of v4 references `src/v4/tasks/` (doesn't
  exist) and describes the `Module` trait system as if it's implemented. Update
  to reflect the actual source state and link to `src/v4/TODO.md`.

---

## Phase 12 — Polish & Debt

### 45. Fix `Row` in `item.rs` to return real rows

- **File:** `src/v4/item.rs:240-267`
- **What:** `Item for Row` currently returns `Rows::new(0..0, table)` — an
  empty range. Marked `// TODO: Implement`.
- **Fix:** Return `Rows::new(0..table.count(), table)`. Must ensure the row
  count is read under appropriate locks — use the count passed to `get()`.

### 46. Implement `Vector` usage or remove it

- **File:** `src/v4/vector.rs`
- **What:** `Vector` is exported (`pub use vector::Vector`) but not used
  anywhere in v4. `Insert` uses `Vec<T::Item>` directly for buffering.
  - If `Vector` is intended for deferred insert buffering, wire it up.
  - If it's unused vestigial code, remove it.

### 47. Distinguish `Insert::Build` from `Query::Build` by name

- **Files:** `src/v4/insert.rs:9`, `src/v4/query.rs:23`
- **What:** Both modules define a `Build` struct. If a user imports both
  (`use crate::v4::insert::*; use crate::v4::query::*;`), they collide.
- **Fix:** Rename to `InsertBuilder` and `QueryBuilder`, or keep `Build` but
  ensure they're not both re-exported at the same level.

### 48. `Guard::columns()` should work for single-column queries

- **File:** `src/v4/query.rs:138-143`
- **What:** `columns()` has bound `where I::Item<'a>: IntoFlat`. For a single
  `Read<T>`, `Item<'a>` = `&'a [T]`, which doesn't implement `IntoFlat` (only
  tuples do). This makes it impossible to call `.columns()` on a single-column
  query.
- **Fix options:**
  1. Add `IntoFlat` impls for non-tuple types (identity transform).
  2. Add a separate `.column()` (singular) method.
  3. Change the return type to not require `IntoFlat`.

### 49. `Vector::drop` calls `resize` with capacity 0 to deallocate

- **File:** `src/v4/vector.rs:66-69`
- **What:**
  ```rust
  impl Drop for Vector {
      fn drop(&mut self) {
          let _ = self.meta.resize(self.data, self.len, (self.cap, 0));
      }
  }
  ```
  This calls `Meta::resize` which allocates a new buffer of capacity 0 (which
  is `NonNull::dangling()`), copies data into it (nothing, since capacity is 0),
  deallocates the old buffer. This is correct but wasteful. Since the only
  purpose is deallocation, it should just call `deallocate` directly:
  ```rust
  fn drop(&mut self) {
      if let Ok(layout) = self.meta.layout(self.cap) {
          unsafe { utility::deallocate(self.data, layout) };
      }
  }
  ```

### 50. Add error context / backtrace support

- **File:** `src/v4/error.rs`
- **What:** `Error` variants don't carry enough context for debugging. Add:
  - The table index for `MissingTable`, `TableOverflow`, `TableUnderflow`.
  - The type name or `Meta` for `InvalidItem`, `ReadWriteConflict`,
    `WriteWriteConflict`.
  - Source file location for `FailedToAllocate`, `FailedToPush` (via
    `#[track_caller]` or `std::backtrace::Backtrace`).

### 51. `Store` should validate that `tables` is always valid after any operation

- **File:** `src/v4/mod.rs:21-35`
- **What:** Currently `Store` is just a wrapper around `Tables`. There's no
  validation that operations leave the store in a consistent state.
- **Fix:** Add a `debug_assert!`-based invariant checker:
  ```rust
  fn check_invariants(&self) {
      for table in self.tables.0.load().slice.iter() {
          debug_assert!(table.count() <= table.capacity());
          // etc.
      }
  }
  ```
  Call it after every `resolve()` — only in debug builds.

---

## Summary Table

| # | Phase | Item | Effort | Risk | Depends on |
|---|-------|------|--------|------|------------|
| 1 | 1 | Remove unused import | Tiny | None | — |
| 2 | 1 | Add Store derives | Tiny | None | — |
| 3 | 1 | Enable unsafe_op_in_unsafe_fn | Medium | Low | — |
| 4 | 2 | Fix Column Send/Sync | Large | **Critical** | — |
| 5 | 2 | Fix Tables find_or_add race | Medium | **Critical** | — |
| 6 | 2 | Document Template::apply safety | Tiny | None | — |
| 7 | 2 | Document Item::get safety | Tiny | None | — |
| 8 | 2 | Document Depend safety contract | Tiny | None | — |
| 9 | 2 | Document Meta::set invariant | Tiny | None | — |
| 10 | 3 | Add v4 feature flag | Small | Low | — |
| 11 | 3 | Fix store alias collision | Tiny | Low | 10 |
| 12 | 3 | Extract row.rs | Small | Low | — |
| 13 | 4 | Fix failing test | Small | Low | 21 (or workaround) |
| 14 | 5 | Document Column unsafe fns | Medium | Low | — |
| 15 | 5 | Document Table::resize invariants | Medium | Low | — |
| 16 | 5 | Audit Table::insert race | Medium | **High** | — |
| 17 | 5 | Audit lock ordering | Medium | **High** | — |
| 18 | 6 | Implement Remove::all() | Small | Low | — |
| 19 | 6 | Implement filtered destroy | Medium | Low | 18 |
| 20 | 6 | Implement Modify | Large | Medium | — |
| 21 | 7 | Implement Keys system | Large | Medium | 4, 5 |
| 22 | 8 | Implement event system | Large | Medium | 18, 20, 21 |
| 23 | 9 | Unit tests: Table | Medium | Low | 4, 16 |
| 24 | 9 | Unit tests: Meta | Small | Low | — |
| 25 | 9 | Unit tests: Query | Medium | Low | 17 |
| 26 | 9 | Unit tests: Insert/Remove | Medium | Low | 18 |
| 27 | 9 | Unit tests: Filter | Small | Low | — |
| 28 | 9 | Unit tests: Depend/Analysis | Small | Low | — |
| 29 | 9 | Unit tests: Slice/Vector/Utility | Small | Low | — |
| 30 | 9 | Integration tests | Large | Low | 21, 22 |
| 31 | 9 | Miri validation | Medium | Low | 23-30 |
| 32 | 9 | Benchmarks | Medium | Low | 21 |
| 33 | 9 | Fuzzing setup | Large | Low | 30 |
| 34 | 10 | Implement Store::with | Small | Low | — |
| 35 | 10 | Add Store::len/is_empty | Tiny | Low | — |
| 36 | 10 | Add Guard::get/get_mut | Small | Low | — |
| 37 | 10 | Implement or remove Module system | XL | Medium | Decision |
| 38 | 10 | Proc-macro derives | Large | Low | 10 |
| 39 | 10 | Add Datum marker trait | Small | Low | — |
| 40 | 11 | Module-level docs | Medium | Low | — |
| 41 | 11 | Public API docs | Large | Low | 40 |
| 42 | 11 | examples/v4/main.rs | Medium | Low | 21, 22 |
| 43 | 11 | Update README | Medium | Low | 37 decision |
| 44 | 11 | Update AGENTS.md | Tiny | Low | 43 |
| 45 | 12 | Fix Row in item.rs | Small | Low | 21 |
| 46 | 12 | Implement or remove Vector | Small | Low | — |
| 47 | 12 | Rename Build types | Small | Low | — |
| 48 | 12 | Fix Guard::columns single-col | Medium | Low | — |
| 49 | 12 | Fix Vector::drop efficiency | Tiny | Low | — |
| 50 | 12 | Add error context | Medium | Low | — |
| 51 | 12 | Add invariants checker | Small | Low | — |
