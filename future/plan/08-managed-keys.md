# Task 08: Managed Keys, Inline `Key` Columns, And Reverse Mapping

This task adds stable identity back on top of the keyless-first storage core.

Read this file together with `future/plan/specification.md` and `future/plan/standards.md`. The rewrite deliberately makes identity optional. This task is the opt-in managed-identity layer for the cases where stable references matter more than the extra overhead.

## Purpose

Implement managed keyed tables through:

- an inline physical `Key` column,
- a global `Keys` resource for `Key -> Row` mapping,
- query support for `&[Key]`,
- structural maintenance of the bidirectional mapping.

## Required Reading

- `future/plan/specification.md`
- `future/plan/standards.md`
- `future/07-keyless-and-user-keyed-tables.md`
- `AGENTS.md`
- `src/key.rs`
- `src/create.rs`
- `src/destroy.rs`
- `src/modify.rs`

## Selected Model

Keyed tables opt into managed identity.

Selected storage shape:

- the table physically stores a `Key` column beside its normal data columns,
- queries can request that column directly,
- a global `Keys` resource stores the reverse `Key -> Row` mapping.

This is intentionally different from trying to reconstruct `Key` from generation fragments at query time.

## Why An Inline `Key` Column Was Chosen

The user requirement is clear:

- rows yielded by a query should be able to retrieve their `Key` at close to zero cost.

An inline `Key` column satisfies that directly:

- `&[Key]` is just another chunk slice,
- zipped queries with `Key` are trivial,
- `Row -> Key` is immediate.

The `Keys` resource then handles:

- `Key -> Row`,
- generation validity,
- slot recycling later if needed.

## Separation Of Responsibilities

### Table `Key` Column

Responsible for:

- cheap query-time access to keys,
- preserving key association when rows move inside a table,
- carrying identity through `swap_remove`.

### Global `Keys` Resource

Responsible for:

- reverse lookup,
- validity checks,
- generation management,
- free-list or recycling policy.

Do not blur these roles.

## Structural Maintenance Rules

Every structural operation on a keyed table must maintain both directions.

### Insert

- allocate or reserve keys,
- write the `Key` column in the destination chunk,
- publish reverse mapping in the `Keys` resource when the row becomes visible.

### Remove

- if `swap_remove` moves a tail row down, the moved row's reverse map must update,
- removed keys must become invalid or recycled according to the `Keys` policy.

### Modify / Move Across Tables

- if a keyed row moves between keyed tables, preserve the key,
- update reverse mapping to the new `Row`,
- if moving between keyed and keyless tables is allowed at all, define the conversion explicitly rather than letting it happen accidentally.

That last point is important enough to call out:

- keyed versus keyless table transitions must be explicit in the API and the planner.

## Query Support

Queries should be able to request keys like any other chunk slice.

Example:

```rust
query::all((
    query::read::<Key>(),
    query::write::<Transform>(),
))
```

This gives users:

- efficient key-aware scans,
- straightforward logging and diagnostics,
- cheap joining with side resources keyed by `Key`.

## Key Lookup

The scheduler-first mode still needs targeted keyed lookup paths for:

- command resolution,
- direct database escape hatches later,
- keyed table utilities.

The `Keys` resource should provide:

- `Key -> Row`,
- enough metadata to reject stale keys cleanly.

The exact surface API can be refined later. The data model must exist in this task.

## Deferred User-Keyed Tables

This task explicitly does not implement user-keyed tables.

Reason:

- the chosen rewrite already introduces major complexity through chunking, scheduling, and managed keys,
- user-key uniqueness and indexing policy deserve a separate design pass.

However, this task should preserve room for them by keeping `IdentityPolicy` extensible.

## Example

```rust
schedule.push(
    query::all((
        query::read::<Key>(),
        query::write::<Health>(),
    )),
    |keys, healths| {
        for (key, health) in keys.zip(healths) {
            if health.current == 0 {
                // user code can stash the key for later stable reference
                log_death(*key);
            }
        }
    },
);
```

This is the reason the inline `Key` column matters.

## Implementation Checklist

1. Extend the table descriptor for `ManagedKeys`.
2. Add physical `Key` column support.
3. Implement the `Keys` resource.
4. Implement keyed insert maintenance.
5. Implement keyed remove maintenance.
6. Implement keyed move maintenance.
7. Add tests for:
   - `&[Key]` query slices,
   - reverse lookup after insert,
   - reverse lookup after `swap_remove`,
   - stale key rejection after remove.

## Pitfalls

### Pitfall: Hiding keyed overhead inside the default path

The entire point of keyless-by-default is to let hot data avoid identity cost. Keep keyed behavior opt-in.

### Pitfall: Treating reverse mapping updates as secondary

If the `Key -> Row` map lags behind row movement, keyed identity becomes subtly incorrect.

### Pitfall: Accidentally turning `Key` into "just another component"

It is stored like a column, but it carries stronger invariants than ordinary data.

## Done Criteria

This task is done when:

- keyed tables can expose chunk `Key` slices,
- reverse lookup exists,
- row movement keeps the bidirectional mapping correct,
- keyless tables still pay none of this cost.
