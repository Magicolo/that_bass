# Task 03: Keyless Row Handles, Dense Movement, And Structural Targeting

This task defines what row identity means in the rewrite when a table is not keyed.

Read this file together with `future/plan/specification.md` and `future/plan/standards.md`. This task is critical because the rewrite intentionally makes keyless tables the default. That means row movement, row targeting, and row lifetimes must be explicit and honest.

## Purpose

Implement the keyless identity model:

- ephemeral `Row<'job>` handles,
- dense `swap_remove` movement,
- row-targeted deferred commands,
- rules for when a `Row` is valid and when it is not.

## Required Reading

- `future/plan/specification.md`
- `future/plan/standards.md`
- `future/07-keyless-and-user-keyed-tables.md`
- `future/08-nested-query-soundness.md`
- `AGENTS.md`
- `src/v1/key.rs`
- `src/v1/modify.rs`
- `src/v1/destroy.rs`

## Selected Model

Keyless tables do not expose stable row identity.

Instead they expose an ephemeral row locator:

```rust
struct Row<'job> { ... }
```

Expected properties:

- cheap to copy,
- packed into a `u64`,
- impossible or inconvenient to store beyond the current job through lifetime-carrying API,
- suitable for equality checks and row-targeted commands inside the current scheduled epoch.

## Packed Representation

The exact layout is table-parameterized.

The user discussion already established the intended fields:

- table index,
- chunk index,
- row index inside the chunk.

The important design choice is:

- row bits and chunk bits are not globally fixed forever,
- each table can interpret the packed payload according to its chunk-capacity configuration.

This is one reason Task 01 must record table-specific row packing metadata.

## Why `Row<'job>` Needs A Lifetime

Without a lifetime or equivalent borrow coupling, users will treat `Row` as persistent identity.

That would be wrong because:

- inserts can create new chunks and change later targeting assumptions,
- removes perform `swap_remove`,
- the same physical row slot can be reused,
- a later row may occupy the same `(table, chunk, row)` coordinates.

The API should communicate this directly.

## Chunk Queries And Row Requests

The query surface should allow users to request row handles alongside chunk slices.

Example direction:

```rust
query::all((
    query::rows(),
    query::write::<Position>(),
))
.expect("query declaration should succeed")
```

Then within the chunk callback:

- the row handle slice names the current inhabited prefix,
- row handles correspond positionally with the other chunk slices.

This lets users:

- compare rows,
- queue targeted removes,
- build local maps,
- debug movement-sensitive behavior.

## Structural Targeting Semantics

For keyless tables, deferred remove or update operations must target rows through `Row<'job>`.

That creates a hard scheduling requirement:

- between the creation of a row-targeted command and the resolution that consumes it,
- no conflicting work may move rows in a way that invalidates the targeted chunk positions.

This requirement is especially important for remove, because remove itself moves rows through `swap_remove`.

## Selected Safety Rule For Remove

Potentially overlapping removes on the same chunk conflict and must not resolve concurrently.

That means:

- if two jobs can target the same chunk for remove,
- their resolve jobs must honor happens-before ordering,
- later dependent queries on that chunk must wait until the relevant remove resolution finishes.

This is intentionally conservative.

It preserves the honesty of `Row<'job>` targeting.

## Example

Suppose one job records:

```rust
commands.remove(row_a);
```

and another records:

```rust
commands.remove(row_b);
```

If `row_a` and `row_b` may be in the same chunk:

- they cannot be treated as freely parallel removals,
- because the first remove can move the second row by swapping in the tail row.

The planner or resolver must serialize or otherwise safely batch them.

## Batched Remove Strategy

Even though overlapping remove jobs conflict, resolution should still be batch-oriented.

Suggested internal shape:

1. collect all row targets for one chunk,
2. sort and deduplicate them,
3. resolve them from highest row index to lowest row index,
4. perform `swap_remove` logic while updating any needed remap state.

Descending order matters because it minimizes accidental invalidation during one batched remove set.

## Row Equality And Reuse

Within one job:

- `Row<'job>` equality is meaningful.

Across jobs or after resolution:

- equality is not a stable identity guarantee.

The documentation for `Row<'job>` must say this clearly.

## Interaction With Keyed Tables

Keyed tables are handled later in Task 08.

This task should not mix `Keys`-extension behavior into the keyless path.

The point of this task is precisely to make the zero-identity path honest and efficient.

## Suggested API Sketch

```rust
schedule.push(
    query::all((query::rows(), query::read::<Contact>()))
        .expect("query declaration should succeed"),
    |rows, contacts, commands| {
        for (row, contact) in rows.zip(contacts) {
            if contact.invalid {
                commands.remove(row);
            }
        }
    },
);
```

Important semantics:

- `row` is only for the current callback epoch,
- the remove is deferred,
- later dependent jobs see the change only after remove resolution,
- row order after resolution is unstable.

## Implementation Checklist

1. Define `Row<'job>`.
2. Define table-specific packing/unpacking rules.
3. Expose row-handle chunk queries.
4. Implement row-targeted command recording for keyless tables.
5. Implement batched remove ordering for one chunk.
6. Add tests covering:
   - row equality within one chunk callback,
   - row invalidation after remove resolution,
   - descending batched remove on one chunk,
   - reuse of physical row slots without identity guarantees.

## Pitfalls

### Pitfall: Accidentally giving `Row` stable semantics in helper APIs

Avoid APIs that accept owned `Row` values without any scheduling or lifetime context.

### Pitfall: Forgetting insert is lighter than remove

Insert appends and does not move existing rows. Remove is the operation that drives the strictest row-targeting constraints.

### Pitfall: Hiding unstable row order

The instability of row order is a core contract, not an internal detail.

## Done Criteria

This task is done when:

- keyless tables can expose transient row handles,
- row-targeted deferred commands exist,
- remove semantics are documented and tested,
- the code makes it difficult to mistake `Row<'job>` for stable identity.

## Implementation Review

Task 03 is implemented in the isolated `v2` lane.

The current implementation lives primarily in:

- `src/v2/schema.rs`
- `src/v2/query.rs`
- `src/v2/command.rs`
- `tests/v2/keyless_rows.rs`
- `examples/v2/keyless_rows.rs`

## Actions Taken

The current implementation does the following:

1. keeps `Row<'job>` as an ephemeral lifetime-carried handle packed into a `u64`,
2. uses table-specific `RowLayout` metadata to encode and decode chunk and row indices,
3. exposes `Table::rows(...) -> Rows<'job>` for chunk-aligned row-handle views,
4. implements `Rows<'job>` as a generated slice-shaped view rather than a stored column,
5. gives `Rows<'job>` safe slice-like helpers:
   - `len`,
   - `is_empty`,
   - `get`,
   - `first`,
   - `last`,
   - `split_at`,
   - `iter`,
   - `IntoIterator`,
   - `zip(...)`,
6. adds a keyless `command::Remove<'job>` buffer with `one`, `array`, `slice`, and `extend`,
7. resolves row-targeted removes in batch through table-local descending per-chunk application with sort-and-deduplicate behavior.

## Current Status

The implemented Task 03 surface is intentionally storage-local:

- generated row views exist,
- keyless remove recording exists,
- batched keyless remove resolution exists,
- but scheduler integration and full query-plan lowering remain later tasks.

That split is deliberate. It keeps the row-handle contract honest before Task 04 and Task 07 wire it into the larger execution model.
