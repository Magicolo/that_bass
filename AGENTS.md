# AGENTS.md

## Purpose

`that_bass` is a high-performance in-memory database for realtime workloads, especially ECS-like game/server workloads where cache locality, predictable schemas, and concurrent reads matter more than full transactional semantics. The codebase is small in public surface area but dense in invariants and unsafe code. Treat it as a systems crate, not a CRUD library.

Primary source references:

- Crate root wiring: `src/lib.rs:1-2`
- Current stable engine (`v1`): `src/v1/mod.rs:1-334`
- Proc-macro crate: `that_base_derive/src/lib.rs:23-356`

## Current State

- Implemented and tested:
  - Schema-driven tables and generational keys.
  - Create, query, destroy, add, remove, modify.
  - Table splitting for parallel query processing.
  - Event listeners with buffered replay and retention policies.
  - Derive macros for `Datum`, `Template`, and `Filter`.
- Present but unfinished:
  - `Defer` exists but is `todo!()` and should be treated as unusable (`src/v1/defer.rs:55-116`).
  - A `Row` derive exists in the proc-macro crate but is not re-exported by `that_bass::v1`, is explicitly postponed in the roadmap, and its tests are commented out (`src/v1/mod.rs:133`, `that_base_derive/src/lib.rs:198`, `tests/v1/derive.rs:131-181`).
- Roadmap direction:
  - Stay table-based and locality-first for now.
  - Add better deferred orchestration, query combinators, richer transforms, generative testing, and possibly event filters.
  - Chunk-based storage, interpreter/serialization work, and scheduler integration are exploratory design notes, not active implementation (`src/v1/mod.rs:53-220`).
  - The selected long-term rewrite direction is now documented in `future/plan/specification.md`, `future/plan/standards.md`, `future/plan/README.md`, and the ordered `future/plan/00-foundation.md` through `future/plan/15-layout-specialization.md` task set. That plan currently assumes per-job command recording but function-level batched resolve phases. Treat those files as design documents for future work, not descriptions of current runtime behavior.
  - Task `future/plan/00-foundation.md` is implemented through the isolated `src/v2/` rewrite lane. Treat `src/v2/` as the future-engine boundary and keep it isolated from the current runtime modules.
  - For `v2`, prefer short public type names that are still full English words. Let module names carry context. Example shape: `Store`, `Configuration`, `command::Kind`, `instrumentation::Category`.

## Workspace Layout

| Path | Role | Key references |
| --- | --- | --- |
| `src/lib.rs` | Crate root; exposes the isolated `v1` and `v2` module trees only | `src/lib.rs:1-2` |
| `src/v1/mod.rs` | Current stable engine root, `Database`, `Meta`, `Datum`, roadmap notes, dead experimental sketches | `src/v1/mod.rs:26`, `src/v1/mod.rs:232`, `src/v1/mod.rs:253` |
| `src/v1/create.rs` | Deferred create queue and resolution | `src/v1/create.rs:14`, `src/v1/create.rs:26`, `src/v1/create.rs:135` |
| `src/v1/query.rs` | Query engine, split iteration, keyed joins via `By`, lock orchestration | `src/v1/query.rs:19`, `src/v1/query.rs:150`, `src/v1/query.rs:188`, `src/v1/query.rs:461`, `src/v1/query.rs:681`, `src/v1/query.rs:1015` |
| `src/v1/modify.rs` | Add/remove/modify resolution, table-to-table moves | `src/v1/modify.rs:21`, `src/v1/modify.rs:82`, `src/v1/modify.rs:225`, `src/v1/modify.rs:401`, `src/v1/modify.rs:532`, `src/v1/modify.rs:738`, `src/v1/modify.rs:848` |
| `src/v1/destroy.rs` | Targeted and filtered destruction | `src/v1/destroy.rs:16`, `src/v1/destroy.rs:32`, `src/v1/destroy.rs:144`, `src/v1/destroy.rs:241`, `src/v1/destroy.rs:398`, `src/v1/destroy.rs:450` |
| `src/v1/event.rs` | Event declaration, buffering, listener lifecycle, typed event families | `src/v1/event.rs:18`, `src/v1/event.rs:27`, `src/v1/event.rs:37`, `src/v1/event.rs:79`, `src/v1/event.rs:160`, `src/v1/event.rs:305`, `src/v1/event.rs:782-966` |
| `src/v1/table.rs` | Table/column storage, schema identity, allocation growth | `src/v1/table.rs:27`, `src/v1/table.rs:55`, `src/v1/table.rs:214`, `src/v1/table.rs:260`, `src/v1/table.rs:308`, `src/v1/table.rs:379` |
| `src/v1/key.rs` | Generational key slots, slot views, key recycling | `src/v1/key.rs:18`, `src/v1/key.rs:34`, `src/v1/key.rs:38`, `src/v1/key.rs:168`, `src/v1/key.rs:201`, `src/v1/key.rs:227`, `src/v1/key.rs:294` |
| `src/v1/template.rs` | Write-side schema declaration and row application | `src/v1/template.rs:9-19`, `src/v1/template.rs:39`, `src/v1/template.rs:119`, `src/v1/template.rs:174` |
| `src/v1/row.rs` | Read/write access declaration and row materialization | `src/v1/row.rs:11-37`, `src/v1/row.rs:81`, `src/v1/row.rs:260-448` |
| `src/v1/filter.rs` | Table-level filtering and dynamic filter trees | `src/v1/filter.rs:9`, `src/v1/filter.rs:34`, `src/v1/filter.rs:68`, `src/v1/filter.rs:93`, `src/v1/filter.rs:165`, `src/v1/filter.rs:211` |
| `src/v1/resources.rs` | Typed local/global caches for metadata and shared state | `src/v1/resources.rs:13-23`, `src/v1/resources.rs:107-170` |
| `src/v2/` | Isolated rewrite lane for the next storage-and-scheduler architecture; foundation currently includes `Store`, configuration, instrumentation vocabulary, and benchmarkable chunk-capacity planning | `src/v2/mod.rs`, `src/v2/store.rs`, `src/v2/instrumentation.rs`, `src/v2/command.rs` |
| `src/v1/core/view_vec.rs` | Snapshot vector primitive used by tables and keys | `src/v1/core/view_vec.rs:19`, `src/v1/core/view_vec.rs:34`, `src/v1/core/view_vec.rs:37`, `src/v1/core/view_vec.rs:59`, `src/v1/core/view_vec.rs:76`, `src/v1/core/view_vec.rs:182` |
| `src/v1/core/utility.rs` | Unsafe helpers, sorted-set helpers, requeue/fold helpers | `src/v1/core/utility.rs:16-74`, `src/v1/core/utility.rs:74-163`, `src/v1/core/utility.rs:171-230` |
| `that_base_derive/src/lib.rs` | Procedural macros; emits absolute `that_bass::v1::...` paths for the stable engine | `that_base_derive/src/lib.rs:23`, `that_base_derive/src/lib.rs:37`, `that_base_derive/src/lib.rs:83`, `that_base_derive/src/lib.rs:198` |
| `tests/` | Integration suites grouped by engine generation; read these before changing semantics | `tests/v1/check.rs`, `tests/v1/query.rs`, `tests/v1/event.rs`, `tests/v1/derive.rs`, `tests/v2/foundation.rs` |
| `examples/v2/` | Runnable sample usage for the rewrite lane; keep this aligned with the current `v2` public API | `examples/v2/main.rs`, `examples/v2/store_planning.rs`, `examples/v2/instrumentation.rs`, `examples/v2/vocabulary.rs` |
| `future/` | Architecture proposals and rewrite planning docs | `future/README.md`, `future/06-recommended-roadmap.md`, `future/plan/specification.md` |

Inactive or misleading files:

- `src/v1/core/borrow.rs` and `src/v1/core/or.rs` are in the repository but not exported by `src/v1/core/mod.rs`; they are currently not compiled.
- `src/v1/mod.rs` contains large dead experimental modules (`next_table_based`, `next_chunk_based`) and old sketches. Treat them as design notes, not runtime code (`src/v1/mod.rs:341-...`).
- `src/v2/` is real compiled code, but it is still only the rewrite foundation lane from `future/plan/00-foundation.md`. It is not yet feature-parity runtime code.
- Tests and benches are grouped under `v1` and `v2` directories to mirror the source layout. Keep new coverage in the matching generation directory.

## Core Mental Model

### 1. Data is stored by schema, not by entity archetype wrapper type

- A `Template` declares a sorted, deduplicated list of `Meta` descriptors (`src/v1/template.rs:39-62`).
- Tables are keyed by the exact ordered list of type IDs for their columns (`src/v1/table.rs:214-238`, `src/v1/table.rs:379-385`).
- `Create<T>` resolves into the one table whose schema exactly matches `T`.
- `Modify<A, R>` computes a target table by applying additions and removals to the source schema (`src/v1/modify.rs:685-735`).

### 2. Keys are generational, rows are movable

- `Key` is `(index, generation)`, with `u32::MAX` sentinels as null/invalid (`src/v1/key.rs:18-67`).
- `Slot` is the mutable indirection from key to current table and row (`src/v1/key.rs:23`, `src/v1/key.rs:253-307`).
- Rows move during destroy/remove/modify compaction. Code must never assume a key stays in one row, or even in one table, across an unlocked boundary.

### 3. Queries are snapshot-ish, not fully transactional

- `Keys` and `Tables` are backed by `ViewVec`, a concurrent snapshot vector that readers must manually update (`src/v1/core/view_vec.rs:19-120`).
- A `Query` lazily discovers newly visible tables via `Query::update` (`src/v1/query.rs:150-186`).
- Exact keyed lookup (`find`) re-validates under table locks using both slot metadata and `keys[row] == key` checks to avoid stale/moved row reads (`src/v1/query.rs:461-521`, `src/v1/query.rs:860-899`).

### 4. Structural operations are deferred queues with explicit `resolve`

- `Create`, `Destroy`, `Modify`, `Add`, and `Remove` accumulate work first, then mutate storage only in `resolve`.
- This separation is fundamental to the crate's concurrency model. Do not casually turn APIs eager.
- `fold_swap` / `try_fold_swap` are the standard "try-lock, requeue, then retry" building blocks for resolve-time fairness and deadlock avoidance (`src/v1/core/utility.rs:74-163`).

## Important Types and What They Mean

### `Meta` / `Datum`

- `Meta` is runtime type metadata plus copy/drop/layout hooks (`src/v1/mod.rs:260-321`).
- `Datum` is intentionally tiny: `Sized + 'static` only (`src/v1/mod.rs:271`).
- `Meta::get<T>()` is cached globally behind a `Mutex<BTreeMap<...>>` and leaks the metadata intentionally for process lifetime (`src/v1/mod.rs:273-321`).

### `Template`

- `Template` is the write-side schema contract (`src/v1/template.rs:19-32`).
- `declare` must register every written column, in sorted/deduplicated order.
- `initialize` binds the template to a specific target table.
- `apply` may only write the row identified by the supplied `ApplyContext`.
- ZST columns are legal and heavily optimized around; code often skips locking/copying when `meta.size() == 0`.

### `Row`

- `Row` is the read/write query contract (`src/v1/row.rs:37-56`).
- `declare` describes column access so conflicts can be rejected up front.
- `initialize` binds the row type to one table.
- `item` yields one logical row view.
- `chunk` yields the entire table slice for split/chunk queries.
- `Option<R>` means "optional presence in a table", not "optional key existence" (`src/v1/row.rs:395-425`).

### `Filter`

- Filters are table-level, not per-row (`src/v1/filter.rs:9-32`).
- `Has<T>` means table schema is a superset of `T`.
- `Is<T>` means exact schema equality with `T`.
- Composite filters are encoded through tuples, `Any`, `Same`, and `Not`.
- `dynamic()` exists because events and cached logic sometimes need a type-erased filter form (`src/v1/filter.rs:34-43`, `src/v1/filter.rs:211-216`).

## Operation Semantics

### Create

- `database.create::<T>()` precomputes `T`'s metadata/state and the exact target table (`src/v1/create.rs:26-32`, `src/v1/create.rs:186-198`).
- Reserving keys is separate from making rows visible. Queued keys are valid handles only after `resolve`.
- `Create::resolve`:
  - Reserves table capacity under an upgradable key lock.
  - Writes row data into not-yet-visible rows without column locks.
  - Copies reserved keys into the table key array.
  - Initializes key slots.
  - Increments table count.
  - Emits create events.
- The code explicitly accepts a short transient state where a slot can be initialized before `table.count` is committed; query code is written to report `InvalidKey` instead of reading out-of-bounds (`src/v1/create.rs:165-175`, `src/v1/query.rs:495-509`).

### Query

- `Query::update` discovers tables lazily and precomputes per-table row state and sorted column lock lists (`src/v1/query.rs:150-186`).
- Lock order matters:
  - Query column locks are sorted by column index inside a table.
  - The query engine only takes column locks while holding at most one table-key read lock at a time.
- `read()` downgrades a row type's writes to reads without redoing table discovery (`src/v1/query.rs:103-127`).
- `split()` yields one handle per table state. This is the intended path for external parallelism (`tests/v1/query.rs:128-181`).
- `By<V>` is the built-in keyed join helper. It sorts requested keys by current table, retries moved keys, and surfaces errors after locks are dropped (`src/v1/query.rs:40-48`, `src/v1/query.rs:333-401`, `src/v1/query.rs:620-697`).

### Modify / Add / Remove

- `Add<T>` and `Remove<T>` are aliases of `Modify<T, ()>` and `Modify<(), T>` (`src/v1/modify.rs:50-53`).
- Same-table modify is a pure column write path (`resolve_set`) (`src/v1/modify.rs:401-425`).
- Cross-table modify is a move:
  - Determine target schema/table and cached apply state.
  - Reserve target space.
  - Copy shared columns into target.
  - Compact source by moving tail rows down.
  - Apply new data only to target-only columns.
  - Update counts and slots.
  - Emit modify events (`src/v1/modify.rs:738-845`, `src/v1/modify.rs:848-901`).
- Table-key locks across source/target are always taken in ascending table-index order to avoid deadlocks (`src/v1/modify.rs:362-381`).
- `ModifyAll::resolve_with` can bulk-transform every row in matching tables and uses a special fast path when source and target tables are identical (`src/v1/modify.rs:532-676`).

### Destroy

- Destroy follows the same table-grouping and retry pattern as modify.
- The important behaviors are:
  - Rows targeted for destruction are collected per table.
  - Surviving tail rows are moved down to fill gaps.
  - Table count is decremented while still under the key write lock.
  - Slots for destroyed keys are released only after the table can no longer expose those rows.
  - Keys are recycled after locks are released (`src/v1/destroy.rs:241-359`).
- `DestroyAll` is a simpler table-wide wipe with direct count reset and bulk recycling (`src/v1/destroy.rs:450-501`).

### Events

- Events are table-structural, not row-mutation logs. The three raw families are `Create`, `Destroy`, and `Modify` (`src/v1/event.rs:79-83`).
- The listener system tracks:
  - how many listeners exist for each family,
  - whether any of them need per-key payloads,
  - buffered chunks that remain alive until all listeners have observed them (`src/v1/event.rs:85-107`, `src/v1/event.rs:181-227`, `src/v1/event.rs:305-364`).
- `Keep` controls listener buffer retention (`src/v1/event.rs:49-54`, `src/v1/event.rs:429-460`).
- Typed event families are macro-generated. `OnAdd` and `OnRemove` are projections of `Modify` events using schema diffs (`src/v1/event.rs:846-939`).

## Concurrency and Unsafe Invariants

These rules matter more than style. Breaking any of them is likely UB or subtle corruption.

### Sortedness is a design tool, not an optimization detail

- Template metadata must stay sorted and deduplicated (`src/v1/template.rs:39-62`, `src/v1/table.rs:214`).
- Query states must stay sorted by table index (`src/v1/query.rs:25`, `src/v1/query.rs:681-697`).
- Destroy/modify resolve states must stay sorted by source table index (`src/v1/destroy.rs:23`, `src/v1/modify.rs:28`).
- Column lock lists are sorted by column index before use (`src/v1/query.rs:171-176`).

### Key validity must be rechecked under the table lock

- `database.keys().get(key)` alone is not enough for safe row access.
- The authoritative check pattern is:
  - read the slot,
  - acquire the table key lock,
  - verify the slot still points to the same table,
  - verify `row < table.count()`,
  - verify `keys[row] == key`.
- This pattern appears in both `Query::try_find` and split lookup paths (`src/v1/query.rs:461-521`, `src/v1/query.rs:860-899`).

### Structural resolvers rely on upgradable key locks

- `Create`, cross-table `Modify`, and `Destroy` all coordinate around `table.header.keys` locks.
- The key lock is what serializes visibility changes to `table.count` with row/key movement.
- Comments around these transitions are important and should be preserved when editing (`src/v1/create.rs:141-175`, `src/v1/destroy.rs:319-359`, `src/v1/modify.rs:648-655`, `src/v1/modify.rs:823-833`).

### Try-lock + requeue is the normal strategy

- The code prefers making progress on uncontended tables over blocking on one hot table.
- `fold_swap` and `try_fold_swap` reorder worklists in place so failed lock attempts get retried later (`src/v1/core/utility.rs:74-163`).
- If you add a multi-table resolve path, copy this pattern instead of inventing a blocking one.

### Unsafe code assumes the trait contracts are true

- `Template::apply` assumes the declared write locks are held or the row is still unobservable.
- `Row::item` / `Row::chunk` assume the declared locks are held and row bounds are valid.
- `Column::get`, `get_all`, `set`, `copy`, `drop`, and `squash` are deliberately thin wrappers over raw pointers (`src/v1/table.rs:79-170`).
- `ViewVec` manually manages old snapshots through `STASH`; if you alter it, re-audit all pointer lifetime logic (`src/v1/core/view_vec.rs:34-120`, `src/v1/core/view_vec.rs:273-291`).

## Patterns to Follow When Making Changes

- Reuse `Resources` caches for expensive type-derived state such as metadata lists, access sets, and per-table modify state (`src/v1/resources.rs:107-170`, `src/v1/template.rs:110-117`, `src/v1/row.rs:250-257`, `src/v1/modify.rs:685-735`).
- Prefer schema-level reasoning over per-type ad hoc branching.
- Skip column locks and copy/drop work for zero-sized types when possible; the code already does this in many places.
- Keep comments that explain weird visibility windows or ordering constraints. They are part of the spec here.
- Preserve tuple macro coverage when extending `Row`, `Template`, or `Filter`.
- If you add a new operation that moves rows, update events and keyed-query retry logic together.

## Quirks, Caveats, and Under-Verified Areas

- Queries may visit a key twice if another thread resolves a move from a visited table to an unvisited table during iteration. This is a known limitation, not an accident (`src/v1/mod.rs:91-93`).
- `query.has` can briefly disagree with `query.find` during destroy/remove windows because `find` does stronger validation (`src/v1/destroy.rs:329-336`).
- `Defer` is present in API surface but not implemented. Any use will panic (`src/v1/defer.rs:55-116`).
- Event tests cover add events only (`tests/v1/event.rs:5-58`).
- Potential event quirk to verify before relying on it:
  - `emit_create` currently gates emission on `self.0.destroy` instead of `self.0.create` (`src/v1/event.rs:125-127`).
  - Treat create-event buffering/listener behavior as under-verified until fixed and tested.
- README-based doctests exist via `skeptic`, but the README is currently minimal (`build.rs`, `README.md`).

## Proc-Macro Notes

- `that_bass::v1` re-exports `Datum`, `Filter`, and `Template`, but not `Row` (`src/v1/mod.rs:26`).
- `Template` derive supports structs only.
- `Filter` derive supports structs and enums.
- `Row` derive is implemented in the proc-macro crate but not part of the supported public workflow yet (`that_base_derive/src/lib.rs:198`, `tests/v1/derive.rs:131-181`).
- The proc-macros emit absolute `that_bass::v1::...` paths. Do not rename the main crate casually without updating the macro crate.

## Testing and Validation Map

- `tests/v1/check.rs` is the highest-value regression suite. It fuzzes sequences of create/add/remove/destroy actions across schema combinations.
- `tests/v1/query.rs` covers keyed lookup, optional rows, split execution, join-like `By` flows, and concurrent split usage.
- `tests/v1/create.rs`, `tests/v1/destroy.rs`, `tests/v1/add.rs`, and `tests/v1/remove.rs` lock down the basic deferred semantics.
- `tests/v1/derive.rs` verifies `Filter` and `Template` derive compile paths.
- `tests/v2/foundation.rs` verifies the rewrite-lane foundation boundary, chunk-capacity planning formula, and required measurement categories.
- `benches/v1/create.rs` measures alternative create APIs, not end-to-end workload throughput.
- `benches/v2/foundation.rs` is the initial comparative harness for the rewrite lane. It currently measures boundary construction and chunk-capacity planning, and should grow with later `v2` tasks instead of adding benchmark hooks inside library code.
- `examples/v2/` is the executable API-evolution trace for the rewrite lane. When `v2` public behavior or naming changes, update these examples in the same patch.

## Recommended Workflow For Agents

1. Read the relevant module plus its matching tests before editing behavior.
2. Identify whether the change is:
   - schema-only,
   - keyed lookup sensitive,
   - row-moving,
   - event-visible,
   - or proc-macro affecting.
3. If row movement or visibility changes are involved, audit:
   - key-slot updates,
   - `table.count` ordering,
   - event emission,
   - query retry/revalidation paths.
4. If a public behavior changes, update tests in the same patch.
5. If the public `v2` surface changes, update `examples/v2/` in the same patch so the runnable sample usage stays current.
6. After any significant code, API, invariant, workflow, or roadmap change, update this `AGENTS.md` in the same change set. Do not let it drift.
7. If you create a commit, use a conventional-commit title and a detailed body.
   The body should explain the high-level change, highlight non-obvious details, give a concise rationale, and point a newcomer at the relevant documentation when terminology needs context.

## Required End-Of-Task Checks

When finishing meaningful work in this repository, run at minimum:

```bash
cargo fmt
cargo build
cargo clippy --all-targets --all-features
```

Strongly recommended in addition:

```bash
cargo test
```

If any of those checks are skipped, blocked, or fail, say so explicitly.
