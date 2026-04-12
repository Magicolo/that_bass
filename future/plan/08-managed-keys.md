# Task 08: `Keys` Resource, `Key` Columns, And Reverse Mapping

This task adds stable identity back on top of the keyless-first storage core.

Read this file together with `future/plan/specification.md` and `future/plan/standards.md`. The rewrite deliberately makes identity optional. This task is the opt-in stable-identity extension for the cases where stable references matter more than the extra overhead.

## Purpose

Implement stable identity through:

- ordinary tables that include a `Key` column,
- a global `Keys` resource for `Key -> Row` mapping,
- query support for `&[Key]`,
- structural maintenance of the bidirectional mapping.

## Required Reading

- `future/plan/specification.md`
- `future/plan/standards.md`
- `future/07-keyless-and-user-keyed-tables.md`
- `AGENTS.md`
- `src/v1/key.rs`
- `src/v1/create.rs`
- `src/v1/destroy.rs`
- `src/v1/modify.rs`

## Selected Model

Storage primitives stay agnostic to stable-identity extensions.

Selected storage shape:

- a table may physically store a `Key` column beside its normal data columns,
- queries can request that column directly because it is just another chunk slice,
- a global `Keys` resource discovers such tables and stores the reverse `Key -> Row` mapping.

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

Important boundary:

- storage primitives do not have a special keyed-table mode,
- the existence of a `Key` column implies that the `Keys` resource must participate,
- synchronization between the `Key` column and the reverse mapping belongs to the extension layer,
  not to primitive table metadata.

## Structural Maintenance Rules

Every structural operation on a table that carries `Key` must maintain both directions.

### Insert

- allocate or reserve keys,
- write the `Key` column in the destination chunk,
- publish reverse mapping in the `Keys` resource when the row becomes visible.

### Remove

- if `swap_remove` moves a tail row down, the moved row's reverse map must update,
- removed keys must become invalid or recycled according to the `Keys` policy.

### Modify / Move Across Tables

- if a keyed row moves between tables that both carry `Key`, preserve the key,
- update reverse mapping to the new `Row`,
- if moving between `Key`-carrying and non-`Key` tables is allowed at all, define the conversion explicitly rather than letting it happen accidentally.

That last point is important enough to call out:

- `Key`-carrying versus non-`Key` tables must transition explicitly in the API and the planner.

## Query Support

Queries should be able to request keys like any other chunk slice.

Example:

```rust
query::all((
    query::read::<Key>(),
    query::write::<Transform>(),
))
.expect("query declaration should succeed")
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

- the chosen rewrite already introduces major complexity through chunking, scheduling, and the
  `Keys` extension,
- user-key uniqueness and indexing policy deserve a separate design pass.

However, this task should preserve room for them by keeping extension hooks generic rather than by
 hardcoding identity modes into primitive table metadata.

## Example

```rust
schedule.push(
    query::all((
        query::read::<Key>(),
        query::write::<Health>(),
    ))
    .expect("query declaration should succeed"),
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

1. Keep `Key`-column discovery generic through table metadata.
2. Implement the `Keys` resource.
3. Implement keyed insert maintenance.
4. Implement keyed remove maintenance.
5. Implement keyed move maintenance.
6. Add tests for:
   - `&[Key]` query slices,
   - reverse lookup after insert,
   - reverse lookup after `swap_remove`,
   - stale key rejection after remove.

## Pitfalls

### Pitfall: Hiding keyed overhead inside the default path

The entire point of keyless-by-default is to let hot data avoid identity cost. Keep keyed behavior opt-in.

### Pitfall: Treating reverse mapping updates as secondary

If the `Key -> Row` map lags behind row movement, keyed identity becomes subtly incorrect.

### Pitfall: Reintroducing key management into primitive table metadata

`Key` is stored like a normal column, but its synchronization invariants belong to the `Keys`
 resource. Keep that boundary explicit.

## Done Criteria

This task is done when:

- tables with `Key` columns can expose chunk `Key` slices,
- reverse lookup exists,
- row movement keeps the bidirectional mapping correct,
- keyless tables still pay none of this cost.
