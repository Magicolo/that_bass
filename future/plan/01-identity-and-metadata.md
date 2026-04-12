# Task 01: Meta, Table, And Resource Identity

This task defines the metadata model that every other rewrite task depends on.

Read this file together with `future/plan/specification.md` and `future/plan/standards.md`. A
correct storage engine and a correct scheduler both require a precise notion of what `Meta` is,
what a row is, what a column is, what a chunk is, what a table is, and what the smallest
schedulable resource is.

## Purpose

Define the rewrite's core identity types and metadata registries:

- store identity,
- meta identity,
- table identity,
- chunk identity,
- column identity,
- scheduler resource identity.

This task is about names, indexing, and metadata ownership, not yet about executing queries or
running workers.

## Required Reading

- `future/plan/specification.md`
- `future/plan/standards.md`
- `future/03-chunking-locks-and-query-plans.md`
- `future/07-keyless-and-user-keyed-tables.md`
- `AGENTS.md`
- `src/v1/mod.rs`
- `src/v1/table.rs`
- `src/v1/template.rs`
- `src/v1/row.rs`

## Key Design Decision

For now, the rewrite keeps a single column concept.

The rewrite distinguishes:

- `Meta`,
- `Row`,
- `Column`,
- `Chunk`,
- `Table`,
- scheduler resources.

Why this matters:

- the selected rewrite no longer needs a logical-versus-physical split in the early milestones,
- the scheduler still needs precise resource identifiers at column granularity,
- the API stays smaller and easier to reason about,
- future decomposition can be reintroduced later through a dedicated macro-driven design instead of
  forcing that complexity into the current metadata model.

## Deliverables

1. A table-shape registry or interner.
2. A table descriptor type.
3. A `Meta` descriptor type.
4. A resource identity type usable by the scheduler.
5. A stable mapping from typed query requests to column accesses.

## Suggested Core Types

The exact names are negotiable. The responsibilities are not.

```rust
struct StoreId(u32);
struct TableId(u32);
struct ChunkId(u32);
struct ColumnId(u16);

enum ResourceId {
    Store(StoreId),
    Table { store: StoreId, table: TableId },
    Chunk { store: StoreId, table: TableId, chunk: ChunkId },
    Column {
        store: StoreId,
        table: TableId,
        chunk: ChunkId,
        column: ColumnId,
    },
}
```

Potential additions:

- `ColumnStorageClass`,
- future decomposition metadata in a later task.

## Table Descriptor Responsibilities

The table descriptor should answer:

- what `Meta` descriptors this table owns,
- what chunks this table owns,
- which metas are inline,
- whether the table includes extension-visible types such as `Key`,
- how `Row` bit packing is interpreted for this table,
- what the target chunk capacity is for this table.

This descriptor should be cheap to clone or reference, because both storage and scheduling will use
it heavily.

## `Meta` Responsibilities

Each `Meta` should carry:

- the type identifier,
- the type name,
- the type layout,
- copy/drop hooks,
- whether the storage is inline or sidecar.

This is the type metadata that one runtime `Column` wrapper refers to.

## Scheduler Resource Identity

The selected granularity floor is:

- one column of one chunk of one table in one store.

That means the scheduler resource model must be able to name:

- store `0`,
- chunk `7` of table `3` of store `0`,
- `Position` column for chunk `7` of table `3`,
- `Velocity` column for chunk `7` of table `3`,
- `Key` column for chunk `7` of table `3`, if the table carries `Meta::of::<Key>()`.

This is the minimum conflict scope in the rewrite, not the only legal scope.

The scheduler must also support broader requests.

Examples:

- whole-store write:
  - `Write(store_0)`
- whole-chunk write:
  - `Read(store_0)`
  - `Read(table_3)`
  - `Write(chunk_7)`
- one-column write:
  - `Read(store_0)`
  - `Read(table_3)`
  - `Read(chunk_7)`
  - `Write(column_2)`

To the scheduler, these only need to be identifiers plus access modes. The important rule is that
descendant accesses carry the ancestor reads that let broad and narrow requests conflict correctly.

## Why Resource Identity Must Exist Before The Scheduler

The scheduler's dependency logic depends on resource identity, but so do:

- query access descriptors,
- command buffer targeting,
- future selective barriers.

If this identity model is invented late, later code will hardcode assumptions that become expensive
to undo.

## Identity Extensions In Metadata

The storage metadata should not record a primitive managed-identity mode.

Selected rule:

- storage primitives know about stores, tables, chunks, metas, rows, and columns,
- `Key` is just another stored type if a table carries `Meta::of::<Key>()`,
- future identity extensions such as `Keys` discover `Key` through ordinary metadata,
- later user-keyed support may add extra extension metadata, but it should not contaminate the
  primitive storage descriptor unnecessarily.

This matters because the storage core should stay agnostic to extension behavior while still
exposing enough generic metadata for extensions to synchronize correctly.

## Singleton Tables

Singletons and global settings are modeled as normal tables.

That means:

- no separate singleton metadata path,
- no special resource namespace,
- a singleton table is just a table whose expected cardinality is one.

The metadata model should not special-case singleton tables in Task 01.

## Query Mapping Example

Suppose the user declares:

```rust
query::all((query::write::<Position>(), query::read::<Velocity>()))
    .expect("query declaration should succeed")
```

The metadata layer should be able to answer:

- which tables can satisfy that query,
- which column implements `Position`,
- which column implements `Velocity`,
- which resource IDs those accesses become for each concrete chunk.

## Deferred Decomposition

The current selected design explicitly postpones datum decomposition.

That means:

- `Position` is one column today,
- `Velocity` is one column today,
- future decomposition may split those into several columns,
- but that work should come later through a dedicated task and likely a macro-driven design.

Task 01 should not add speculative complexity for that future.

## Suggested Implementation Checklist

1. Define the core ID newtypes.
2. Define the `Meta` descriptor.
3. Define the `Row` packing model.
4. Define the initial `Chunk` and `Column` runtime types.
5. Define the table descriptor.
6. Define the mapping from typed query request to column access.
7. Define the functions that derive:
   - scheduler resource IDs at every store/table/chunk/column scope,
   - full hierarchical dependency lists for leaf accesses,
   - and broader dependencies for chunk/table/store-wide work.
8. Add tests for:
   - table-shape equality,
   - meta access mapping,
   - ordinary `Key` meta handling,
   - resource-ID generation,
   - hierarchical dependency generation.

## Pitfalls

### Pitfall: Reusing the current `v1` metadata model too literally

The current crate is useful reference material, but the rewrite needs scheduler-aware resource
identity and clearer runtime terminology even in the simplified one-column model.

### Pitfall: Encoding scheduler concerns directly into user-visible type names

Keep internal resource identity precise, but keep user-facing queries simple.

### Pitfall: Smuggling future decomposition into the early metadata layer

If the current milestone only needs one column concept, keep one column concept. Reintroduce a
richer layout model only when there is a real design and benchmark reason.

## Implementation Review

The current repository now implements this task in `src/v2/schema.rs` with:

- a `Catalog` that interns table shapes and constructs new mutable tables,
- explicit index newtypes for stores, tables, chunks, and columns,
- `Meta` descriptors for stored-type metadata,
- `RowLayout` and `Row<'job>` for packed row identity,
- `Column<'a>` and `Chunk` runtime types for the storage vocabulary,
- `Table` descriptors for row packing, chunk planning, and scheduler identity,
- hierarchical `Resource` identifiers from store scope down to column scope,
- `Dependency` generation for:
  - broad store/table/chunk requests,
  - and leaf accesses that expand into read ancestors plus the requested leaf access,
- stable typed access mapping through `Table::map_access::<T>(...)`.

The current implementation also deliberately treats `Key` as an ordinary stored type so the future
`Keys` resource can discover keyed tables through `Meta` rather than through primitive table modes.

## Actions Taken In The Repository

The following concrete actions were taken to satisfy this task:

- expanded `src/v2/schema.rs` from a simple chunk-planning helper into the actual metadata layer,
- aligned the `v2` terminology with the selected runtime vocabulary:
  - `Meta`,
  - `Row`,
  - `Column`,
  - `Chunk`,
  - `Table`,
  - `Store`,
- kept the table-shape registry separate from table registration so equivalent meta sets can still
  be interned,
- added hierarchical store/table/chunk/column resource identifiers,
- added dependency generation so:
  - broad store/table/chunk requests carry the required ancestor reads,
  - and leaf accesses expand to ancestor reads plus leaf accesses,
- kept `Key` as ordinary `Meta` rather than a primitive table policy so the future `Keys` resource
  can discover it through normal metadata inspection,
- added `tests/v2/metadata.rs` to cover table-shape interning, `Meta` handling, `Key` handling,
  `Row` packing, `Column` wrappers, resource generation, hierarchical dependency generation,
  duplicate-meta rejection, and inline-versus-sidecar accounting,
- added `examples/v2/metadata.rs` so the current metadata API is visible in runnable sample usage,
- updated `AGENTS.md` and the `v2` module docs so newcomers can find the new surface quickly.

## Done Criteria

This task is done when:

- a table can be described without ambiguity,
- a typed query request can map to one column access,
- the scheduler has a stable resource naming scheme to build on,
- future identity extensions do not require redesigning the metadata layer.

Current status:

- implemented in the current repository layout.
