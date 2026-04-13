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

## Selected Synchronization Model

The selected direction is not a heap-allocated resolve-delta side channel.

Rejected direction:

- building `Vec<Key>` or `Vec<(Key, Row)>` side buffers during resolve and replaying them later.

Reason:

- it adds avoidable allocation and memory traffic to a hot structural path,
- and it makes the storage-to-`Keys` synchronization boundary more indirect than it needs to be.

Selected direction:

- keyed structural synchronization happens inline during batched resolve,
- through a private low-cost synchronization interface or direct `Keys` calls,
- with no per-resolve heap-built key-update vectors required by the model.

Practical consequence:

- the command resolver remains strongly coupled to `Keys` when a resolved table carries a `Key`
  column,
- because that coupling is the lower-cost option and performance is the primary driver here.

Important simplification:

- the reverse map only needs two semantic operations:
  - `publish(key, row)` for both insert and move,
  - `release(key)` for remove.
- a separate public `move` event is unnecessary,
- because row movement is just "the key now maps to a new row".

## `Keys` Slot States

The `Keys` resource should distinguish at least these states:

- `Free`
- `Reserved`
- `Live`

Selected semantics:

- `Keys::reserve()` transitions one slot from `Free` to `Reserved` and returns a `Key`,
- insert resolution transitions that slot from `Reserved` to `Live(Row)`,
- remove resolution transitions that slot from `Live(Row)` to `Free`,
- and row movement updates `Live(old_row)` to `Live(new_row)`.

Reserved keys are intentionally observable.

Reason:

- user code often needs to wire up references before the corresponding row becomes visible,
- so "reserved but not yet live" is a real and useful state.

That means keyed lookup must be able to distinguish at least:

- invalid or stale keys,
- reserved keys with no live row yet,
- live keys that map to a row.

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

More precise initialization rule:

- the presence of a `Key` column alone does not instantiate `Keys`,
- `Keys` is created when it is explicitly injected into a scheduled function or other supported
  initialization path,
- keyed insert maintenance is only supported through that injected `Keys` capability,
- and inserting arbitrary `Key` values without going through `Keys::reserve()` is intentionally not
  the supported path for live managed identity.

## Structural Maintenance Rules

Every structural operation on a table that carries `Key` must maintain both directions.

### Insert

- allocate or reserve keys,
- write the `Key` column in the destination chunk,
- publish reverse mapping in the `Keys` resource when the row becomes visible.

Selected refinement:

- user code reserves the key before recording the insert, typically through an injected `Keys`
  handle,
- the row tuple given to `Insert<T>` therefore already contains `Key`,
- and insert resolution only needs to publish the final `Key -> Row` mapping once the row has been
  written and made visible.

### Remove

- if `swap_remove` moves a tail row down, the moved row's reverse map must update,
- removed keys must become invalid or recycled according to the `Keys` policy.

Selected refinement:

- remove resolution reads the removed row's `Key` before the row is erased and releases it,
- and if `swap_remove` moves a tail row, remove resolution reads the moved row's `Key` and
  publishes its new row immediately.
- no extra `move` mechanism is required beyond that publish step.

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

## Keyed Random Access

The `Keys` resource exists primarily to support stable keyed lookup.

Selected query flow:

1. the keyed query path holds a reference to `Keys`,
2. it asks `Keys` for the current slot state of the requested `Key`,
3. if the key is `Live`, it obtains the corresponding `Row`,
4. it verifies that the row's table is compatible with the query,
5. it uses the row's table/chunk/row indices to reach the physical row,
6. it materializes the query item from that physical row.

The `Key` column compare is not meant to be the ordinary validity check.

Selected stance:

- it is acceptable to use a `debug_assert!` that the `Key` stored in the row's `Key` column
  matches the queried key,
- because the design goal is that `Keys` and structural resolution must never make an
  unsynchronized `Key -> Row` mapping observable in the first place.
- bounds and table-shape checks still remain ordinary runtime checks where needed.

## Key Lookup

The scheduler-first mode still needs targeted keyed lookup paths for:

- command resolution,
- direct database escape hatches later,
- keyed table utilities.

The `Keys` resource should provide:

- `Key -> Row`,
- enough metadata to reject stale keys cleanly.

The exact surface API can be refined later. The data model must exist in this task.

## Scheduler And Dependency Model

The `Keys` resource is an extension resource from the scheduler's perspective.

Selected dependency direction:

- `Keys::reserve()` should be allowed under a `Read(Keys)` dependency if the implementation
  remains internally concurrent-safe and linearizable through atomics or equivalent internal
  synchronization,
- keyed random-access lookup also uses `Read(Keys)`,
- keyed structural maintenance during resolve uses `Write(Keys)`,
- because publish and release operations define visibility for later keyed lookups.

This means `reserve()` should not serialize all key-producing jobs if the implementation can avoid
it.

## `Keys` Implementation Direction

The target direction is a slab-like concurrent slot structure, inspired by `slab` but specialized
for this crate's constraints.

Selected direction:

- append-only slot storage with stable indices,
- atomics on the hot path for reserve, publish, release, and random-access lookup,
- free-slot reuse through an atomic free structure where practical,
- and a very short-lived lock only for rare storage growth if a fully lock-free growth strategy is
  not justified in the MVP.

The important performance rule is:

- reserve should scale across threads,
- and random-access keyed lookup should remain close to one indexed load plus row decoding.

This task should preserve room for a segmented slab or similarly append-only growth structure so
slot storage itself does not have to move.

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
2. Implement the `Keys` resource with explicit `Free` / `Reserved` / `Live` slot states.
3. Implement `Keys::reserve()` through an atomic-first slab-like structure.
4. Implement keyed insert maintenance.
5. Implement keyed remove maintenance.
6. Implement keyed move maintenance through reverse-map publish, not a separate event model.
7. Implement keyed random-access lookup.
8. Add tests for:
   - `&[Key]` query slices,
   - reserved-but-not-live key observability,
   - reverse lookup after insert,
   - reverse lookup after `swap_remove`,
   - stale key rejection after remove,
   - a later keyed lookup observing the resolved row after same-frame keyed insert,
   - no unsynchronized `Key -> Row` visibility during remove-induced row movement.

## Pitfalls

### Pitfall: Hiding keyed overhead inside the default path

The entire point of keyless-by-default is to let hot data avoid identity cost. Keep keyed behavior opt-in.

### Pitfall: Treating reverse mapping updates as secondary

If the `Key -> Row` map lags behind row movement, keyed identity becomes subtly incorrect.

### Pitfall: Serializing `reserve()` through the scheduler without necessity

If `reserve()` can be implemented as an internally concurrent-safe operation, do not model it as a
whole-resource scheduler write by default.

### Pitfall: Using generic event-style extension hooks when a tighter path is cheaper

Task 08 should prefer direct resolve-time synchronization over an abstract event system if that is
the lower-cost design.

### Pitfall: Reintroducing key management into primitive table metadata

`Key` is stored like a normal column, but its synchronization invariants belong to the `Keys`
 resource. Keep that boundary explicit.

## Implemented Shape

Task 08 is implemented in `src/v2/` with this concrete surface:

- `Store::initialize_keys()` explicitly creates or reuses the managed `Keys` resource,
- `Store::keys()` exposes that resource when it exists,
- `schedule::Builder::add_keys(...)` marks one function as a `Read(Keys)` user,
- `runtime::FunctionContext::keys()` exposes the initialized `Keys` handle to scheduled work,
- `key::Keys` provides `reserve()`, `state(...)`, and `get(...)`,
- keyed insert resolve publishes reserved keys inline as rows become visible,
- keyed remove resolve releases removed keys and republishes moved keys inline during
  `swap_remove`,
- and `query::All::get(&Store, &Keys, Key)` provides the first keyed random-access lookup path.

Current implementation note:

- managed-key synchronization currently requires an inline `Key` column,
- and resolve uses a stricter internal split between first publication and live-row republication
  so duplicate live keys cannot masquerade as valid inserts.

## Done Criteria

This task is done when:

- tables with `Key` columns can expose chunk `Key` slices,
- reverse lookup exists,
- row movement keeps the bidirectional mapping correct,
- keyless tables still pay none of this cost.
