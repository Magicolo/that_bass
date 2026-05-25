# Task 03: Enable `WriteWith` — Mutable Column Access in `query::Access`

## Goal

Unblock the `WriteWith(Meta)` accessor (currently commented out) and resolve
the tension between `Access::get` taking `&table::Table` (immutable) and the
need to produce `&mut [T]` or `&mut Column` from that same table.

## The Conundrum

`query::Access::get` was deliberately changed from `&mut table::Table` to
`&table::Table`:

```rust
pub trait Access {
    fn get<'a>(&'a self, state: &'a Self::State, table: &'a table::Table) -> Self::Item<'a>
        where Self: 'a;
}
```

This was done to massively reduce unsafe code — with `&Table`, the borrow
checker ensures no two concurrent mutable borrows of the table itself.  But
it creates a problem: returning `&'a mut [T]` from a shared `&'a Table`
requires interior mutability through the `Column`'s internal data pointer.

`Write<T>` already works around this by casting the shared `&table::Table`
to a raw pointer and calling `column.as_mut(table.count)`:

```rust
impl<T: 'static> Access for Write<T> {
    fn get<'a>(&'a self, state: &Self::State, table: &'a table::Table) -> Self::Item<'a> {
        let column = unsafe { table.columns.get_unchecked(*state as usize) };
        unsafe { column.as_mut(table.count) }
    }
}
```

`WriteWith` is blocked because:

1. The function signatures for producing `&mut Column` from `&Column` aren't
   settled.
2. The column access pattern for runtime `Meta` differs from static `TypeId`.
3. There's uncertainty about how to handle the "shared reference → mutable
   output" pattern consistently.

## Current Access Implementations

| Accessor | State | Item | initialize | get |
|----------|-------|------|------------|-----|
| `Read<T>` | `u32` (column index) | `&'a [T]` | `table.column(TypeId)` | `column.as_ref(count)` |
| `Write<T>` | `u32` (column index) | `&'a mut [T]` | `table.column(TypeId)` | `column.as_mut(count)` |
| `ReadWith` | `u32` (column index) | `&'a Column` | `table.column(meta.identifier)` | `table.columns.get_unchecked(index)` |
| `WriteWith` | `u32` (column index) | `&'a mut Column` | (commented out) | (commented out) |
| `Row` | `()` | `Rows<'a>` | always succeeds | `Rows::new(0..count, table.index())` |
| `Table` | `()` | `&'a Table` | always succeeds | returns table directly |

## The Underlying Problem

`Column` stores data as a raw `NonNull<u8>` pointer:

```rust
pub struct Column {
    pub(crate) meta: Meta,
    pub(crate) data: NonNull<u8>,
}
```

The `as_ref` and `as_mut` methods both cast the same `NonNull<u8>` — they're
equally unsafe from the compiler's perspective.  The difference is that
`Read<T>` can produce a shared reference from the shared `&Table` safely
(since shared → shared is sound), while `Write<T>` already relies on the
invariant that no other accessor in the group writes to the same column.

In other words, `Write<T>` already works through the same pointer cast
pattern that `WriteWith` would use.  The safety guard is not in the method
signature — it's in the conflict detection (Task 02).

## Proposed Approach

### 1. Settle `Access::get` signatures

Keep them as-is.  The `&table::Table` → `&mut [T]` pattern is a controlled
unsafety that Task 02 will validate.  The function signatures don't need to
change to enable `WriteWith` — it just needs to mirror `Write<T>` for the
runtime-Meta case.

### 2. Implement `WriteWith` analogously to `ReadWith`

```rust
impl Access for WriteWith {
    type Item<'a>
        = &'a mut column::Column
    where
        Self: 'a;
    type State = u32;

    fn initialize(&self, table: &table::Table) -> Option<Self::State> {
        Some(table.column(self.0.identifier)?.index())
    }

    fn get<'a>(&'a self, state: &Self::State, table: &'a table::Table) -> Self::Item<'a> {
        // Same unsafe pattern as Write<T>: access the column by index,
        // cast through raw pointer to produce mutable reference.
        // Safety: Task 02 ensures no other module in this group
        // has Write access to the same column.
        let column = unsafe { table.columns.get_unchecked(*state as usize) };
        unsafe { &mut *(column as *const Column as *mut Column) }
    }
}
```

Alternatively, add a `get_mut` helper on `Table` (mirroring `columns_mut`
from an earlier revision) that casts internally:

```rust
impl Table {
    pub(crate) unsafe fn column_mut(&self, index: u32) -> &mut Column {
        unsafe {
            &mut *self.columns.as_ptr().cast::<Column>().add(index as usize)
        }
    }
}
```

### 3. Re-enable the builder method on `query::Module`

```rust
impl<A: Access> Module<A> {
    pub fn write_with(self, meta: Meta) -> Module<A::Out>
    where
        A: Push<WriteWith>,
    {
        self.push(WriteWith(meta))
    }
}
```

### 4. Add `write_with` to the `access` test

```rust
let mut s = Store::new();
let mut b = s.state(
    State::build()
        .push(Query::build().write_with(Meta::of::<u32>()))
        .push(Query::build().read::<char>()),
)?;
```

## Challenges

1. **`Column` has no `as_mut` equivalent for `Column` itself**.  `as_mut`
   and `as_ref` are typed (`<T>`), giving `&mut [T]` / `&[T]`.  `WriteWith`
   needs `&mut Column` — the column itself, not the data within it.  The
   direct cast `&*(column as *const Column as *mut Column)` is the simplest
   approach, but should be wrapped in a named helper for grep-auditability.

2. **`WriteWith` vs `Write<T>` consistency**.  Both should use the same
   underlying mechanism.  If we add a `table.column_mut(index)` helper, both
   accessors should call it.  This reduces the unsafe surface area.

3. **The `WriteWith` column is type-erased**.  The caller gets `&mut Column`
   and uses `column.meta()` + `column.set_with()` / `column.drop_with()` to
   operate on it.  This is less ergonomic than `Write<T>`'s typed `&mut [T]`,
   but necessary for dynamic schema cases.

4. **Would `WriteWith` need its own `as_mut_slice`?**  No — the caller can
   inspect `column.meta()` and use the Meta's function pointers.  The point
   of `WriteWith` is to grant mutable access to a column identified by
   runtime metadata, not to produce typed slices.

## Relation to Other Tasks

- **Task 02 (aliasing safety)** makes this task sound.  Without conflict
  detection, `WriteWith` would be another source of UB.
- **Task 01 (unify insert/remove)** may also need `WriteWith`-like access for
  the resolve phase if inserts write to columns by Meta rather than by
  static type.

## Success Criteria

- [ ] `WriteWith(Meta)` implements `Access`.
- [ ] `query::Module<A>::write_with(meta)` builder method is re-enabled.
- [ ] `Table` has a (possibly unsafe, crate-internal) method to get `&mut
      Column` by index from a shared reference.
- [ ] The `access` test in `state.rs` includes a `write_with` in the chain.
- [ ] Build passes with no new warnings.
- [ ] No new unsafe blocks without a clear safety comment.
