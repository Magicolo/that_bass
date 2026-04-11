# Task 01: Schema, Table, And Resource Identity

This task defines the metadata model that every other rewrite task depends on.

Read this file together with `future/plan/specification.md` and `future/plan/standards.md`. A correct storage engine and a correct scheduler both require a precise notion of what a table is, what a physical column is, and what the smallest schedulable resource is.

## Purpose

Define the rewrite's core identity types and metadata registries:

- schema identity,
- table identity,
- chunk identity,
- physical column identity,
- scheduler resource identity.

This task is about names, indexing, and metadata ownership, not yet about executing queries or running workers.

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

The rewrite distinguishes:

- logical datums,
- physical columns,
- scheduler resources.

These are related, but not identical.

Why this matters:

- one logical datum may later decompose into several physical columns,
- the scheduler must reason about conflicts at the physical resource level,
- the user-facing query model still wants logical names and simple syntax.

## Deliverables

1. A schema registry or interner.
2. A table descriptor type.
3. A physical column descriptor type.
4. A resource identity type usable by the scheduler.
5. A stable mapping from logical query requests to physical columns.

## Suggested Core Types

The exact names are negotiable. The responsibilities are not.

```rust
struct SchemaId(u32);
struct TableId(u32);
struct ChunkId(u32);
struct PhysicalColumnId(u16);

struct ResourceId {
    table: TableId,
    chunk: ChunkId,
    column: PhysicalColumnId,
}
```

Potential additions:

- `LogicalDatumId`,
- `TableKind`,
- `IdentityPolicy`,
- `ColumnStorageClass`.

## Table Descriptor Responsibilities

The table descriptor should answer:

- what logical schema this table implements,
- what physical columns exist,
- what each physical column's layout is,
- which columns are inline,
- whether the table is keyless or keyed,
- how `Row` bit packing is interpreted for this table,
- what the target chunk capacity is for this table.

This descriptor should be cheap to clone or reference, because both storage and scheduling will use it heavily.

## Physical Column Descriptor Responsibilities

Each physical column descriptor should carry:

- logical owner datum,
- physical storage index,
- element layout,
- alignment,
- whether the column is inline or sidecar,
- future decomposition metadata if applicable.

Do not assume "logical datum == one column" inside the metadata model even if that is true in MVP storage.

## Scheduler Resource Identity

The selected granularity floor is:

- one physical column of one chunk of one table.

That means the scheduler resource model must be able to name:

- `Position` column for chunk `7` of table `3`,
- `Velocity` column for chunk `7` of table `3`,
- `Key` column for chunk `7` of table `3`, if the table is keyed.

This is the minimum conflict scope in the rewrite.

## Why Resource Identity Must Exist Before The Scheduler

The scheduler's dependency logic depends on resource identity, but so do:

- query access descriptors,
- command buffer targeting,
- future field decomposition,
- future selective barriers.

If this identity model is invented late, later code will hardcode assumptions that become expensive to undo.

## Identity Policy In Metadata

The table descriptor should record an explicit identity policy.

MVP policies:

- `Keyless`
- `ManagedKeys`

Deferred policy:

- `UserKeyed`

Even though user-keyed tables are deferred, the descriptor should still be shaped so that adding them later does not require breaking everything.

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
```

The metadata layer should be able to answer:

- which table schemas can satisfy that query,
- which physical columns implement `Position`,
- which physical columns implement `Velocity`,
- which resource IDs those accesses become for each concrete chunk.

## Future-Proofing For Decomposition

Future decomposition might map:

```rust
Position { x: f64, y: f64 }
```

to:

- `Position::x`
- `Position::y`

Therefore the logical-to-physical mapping should not assume a one-to-one relationship.

Even if the MVP only stores:

- one `Position` column,

the descriptor API should still allow:

- one logical datum to list several physical columns.

## Suggested Implementation Checklist

1. Define the core ID newtypes.
2. Define the table descriptor.
3. Define the physical column descriptor.
4. Define the identity policy enum.
5. Define the mapping from logical query request to physical column list.
6. Define the function that derives scheduler resource IDs from a chunk plus physical access request.
7. Add tests for:
   - schema equality,
   - column ordering,
   - identity-policy tagging,
   - resource-ID generation.

## Pitfalls

### Pitfall: Reusing the current `Meta` and table model too literally

The current crate's metadata is useful reference material, but the rewrite needs a stronger distinction between logical and physical layout.

### Pitfall: Encoding scheduler concerns directly into user-visible type names

Keep internal resource identity precise, but keep user-facing queries simple.

### Pitfall: Forgetting keyed tables need a physical `Key` column

The `ManagedKeys` path depends on this later. The table descriptor must leave room for it now.

## Done Criteria

This task is done when:

- a table can be described without ambiguity,
- a logical query request can map to physical columns,
- the scheduler has a stable resource naming scheme to build on,
- future decomposition and keyed tables do not require redesigning the metadata layer.
