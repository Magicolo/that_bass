# Task 01: Unify Insert and Remove with the Module System

## Goal

Make `Insert` and `Remove` first-class `Module` implementations so they can be
composed alongside queries via `State::build().push(...).push(...)`.  Currently
they are standalone structs with ad-hoc `one()`/`resolve()` APIs on `Store`.

## Current State

### Insert (`src/v4/insert.rs`)

```rust
pub struct Insert<'a, T: template::Template> {
    table: u32,
    tables: &'a mut [Table],   // <--- raw slice, bypasses the Module system
    state: T::State,
    template: T,
}
```

- `Store::insert(template)` finds-or-creates the table, initializes the
  template's state, returns `Insert`.
- `Insert::one(item)` reserves a row and buffers data (deferred).
- `Insert::resolve()` flushes buffered data into columns and commits rows.
- Resolve needs `&mut Table` indirectly through the raw `tables` slice.

### Remove (`src/v4/remove.rs`)

```rust
pub struct Remove<'a> {
    rows: Vec<(u32, u32)>,        // (table_index, row_index)
    tables: &'a mut [Table],      // <--- raw slice
}
```

- `Store::remove()` returns a fresh `Remove`.
- `Remove::one(row)` queues a row for removal.
- `Remove::resolve()` sorts queued rows by table, compacts each table
  via `table.release(rows)`, returns count.
- Resolve needs `&mut Table` through the raw `tables` slice.

### The Module Trait (`src/v4/module.rs`)

```rust
pub trait Module {
    type Item<'a> where Self: 'a;
    type State;

    fn initialize(&self, store: &mut Store) -> Result<Self::State, Error>;
    fn update(&self, state: &mut Self::State, store: &Store) -> Result<bool, Error>;
    fn get<'a>(&'a self, state: &'a Self::State, store: &'a Store) -> Self::Item<'a>
        where Self: 'a;
}
```

Key constraint: `update` and `get` receive `&Store` (immutable).  Only
`initialize` gets `&mut Store`.  This means resolve-time mutation must happen
through some other mechanism — either by holding a `RefCell`-like interior
mutability in the state, or by having `get` return a handle with its own
mutable borrow of tables.

## Design Decisions

### Where table creation happens

Table creation (finding or inserting a table by schema) is only legal during
`Module::initialize` because that's where `&mut Store` is available.  This
matches the existing `Store::insert` flow where `find_or_insert_table` runs
before the `Insert` handle is returned.

`Module::update` will be a no-op for insert modules — once the table is
found/created and the template initialized, all work happens through the
handle returned by `get`.

### Resolve-time mutable access

The core tension: `Template::resolve` needs to write column data, but
`Module::get` only receives `&Store`.  Options:

1. **Interior mutability**.  Give each `Table` a lock (e.g. `UnsafeCell` or
   `RwLock`).  The resolve handle holds a guard acquired during `get`.  This
   complicates `Table` but keeps the `Module` trait signatures clean.

2. **Two-phase handle**.  `get` returns a handle with a lifetime tied to the
   `Module` itself (not the store).  The handle owns a mutable reference to
   the target table, obtained through some deferred allocation during
   `update`.  This needs careful lifetime design.

3. **Split `Module` into `Module` + `Mutator`**.  A second trait with a
   `resolve(&mut self, store: &mut Store)` method that only mutation-capable
   modules implement.  The `State`/`Rest` machinery only calls resolve when
   all modules have been `get`-ed and the user is done iterating.  This
   matches the existing deferred-operation philosophy (accumulate work, then
   resolve).

The third option (separate resolve step) is the most natural fit for v4's
design.  It mirrors how `template::Template` already separates `defer` from
`resolve`.

### Lazy vs eager insertion

Insert is intentionally lazy: `one()` buffers item data, `resolve()` commits.
This deferred style is a core v4 invariant and must be preserved.

Remove is also deferred: `one()` queues rows, `resolve()` compacts tables.

## Proposed Approach

### 1. Add a `resolve` method to `Module` (or a companion trait)

```rust
pub trait Module {
    // ... existing methods ...

    /// Apply deferred mutations to the store.  Called after `get` has been
    /// consumed.  Only mutation-capable modules do work here; read-only
    /// modules return Ok(()).
    fn resolve(&mut self, state: &mut Self::State, store: &mut Store) -> Result<(), Error> {
        let _ = (state, store);
        Ok(())
    }
}
```

Default implementation does nothing, so existing query modules compile
unchanged.

### 2. Create a `template::InsertModule<T>` wrapper

Like `query::Module<A>` wraps an `Access` impl, this wraps a `Template` impl
as a `Module`:

```rust
pub struct Insert<T: Template>(pub T);
```

- `initialize`: calls `declare()` on the template, finds or creates the
  table via `store.find_or_insert_table(...)`, calls `template.initialize()`.
  Stores the table index, template state, and a buffer in `Self::State`.
- `update`: no-op (returns `false`).  Table was already found/created during
  initialize.
- `get`: returns an `InsertHandle<'_, T>` holding a mutable reference to
  the buffered item queue + the table index.  The handle exposes `.one(item)`
  and possibly the table slice for row-based operations.
- `resolve`: calls `table.ensure()`, `template.resolve()`, `table.commit()`.

### 3. Create a `template::RemoveModule` wrapper

```rust
pub struct Remove;
```

- `initialize`: initializes an empty `Vec<(u32, u32)>` queue.
- `update`: no-op.
- `get`: returns a `RemoveHandle<'_>` that exposes `.one(row)` to queue rows.
- `resolve`: sorts queued rows, compacts tables via `table.release()`,
  returns count.

### 4. Wire `resolve` through `State` and `Rest`

The `State` and `Rest` types need a `.resolve(&mut self, store: &mut Store)`
method that walks the module chain and calls `resolve` on each element.

For tuples, this cascades:
```rust
impl<H: Module, T: Module> State<'_, (H, T)> {
    pub fn resolve(&mut self, store: &mut Store) -> Result<(), Error> {
        self.module.0.resolve(&mut self.state.0, store)?;
        self.module.1.resolve(&mut self.state.1, store)?;
        Ok(())
    }
}
```

## Challenges

1. **Lifetime of the insert handle**.  `Insert::one` currently calls
   `table.reserve(1)?.next()` which needs `&mut Table`.  The resolve handle
   returned by `get` must carry a mutable reference to the target table.
   Since `get` receives `&Store`, the handle must use interior mutability or
   the mutable reference must be stored in `State` (acquired during
   `initialize`).

2. **Table index stability**.  If a new table is created during
   `initialize`, it may reallocate the `Store::tables` Vec, invalidating any
   previously stored table indices or references.  The insert module must
   store the table index (a `u32`) rather than a reference.  On `resolve`,
   it must re-index into `store.tables`.

3. **Safety of concurrent resolve**.  If two insert modules target the same
   table (e.g. two `Insert<Column<T>>` with different T-s), their resolves
   must not interfere.  Column-level locking or per-table sequencing may be
   needed.  This is related to Task 02 (aliasing safety).

4. **Interaction with query modules in the same state**.  A user might
   compose `(Query::build().read::<T>(), Insert::build().column::<T>())`.
   The query sees the table before the insert resolves.  This is the expected
   lazy behavior (resolve hasn't been called yet), but should be clearly
   documented.

## Success Criteria

- [ ] `Insert<T: Template>` implements `Module`.
- [ ] `Remove` implements `Module`.
- [ ] Both can be composed alongside `Query` modules via `State::build().push(...)`.
- [ ] `State::resolve(&mut self, store: &mut Store)` propagates to all
      composed modules.
- [ ] The `access` test in `state.rs` can be extended to include an insert
      module in the chain.
- [ ] Existing `Store::insert` and `Store::remove` methods still work (or are
      clearly deprecated).
- [ ] Build passes with no new warnings.
