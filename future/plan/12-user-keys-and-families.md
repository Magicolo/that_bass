# Task 12: User-Keyed Tables And `Families`

This task extends the rewrite's identity model beyond the keyless core and `Keys` MVP.

Read this file together with `future/plan/specification.md` and `future/plan/standards.md`. The selected architecture deliberately postpones some identity complexity. This task picks it back up once the keyless core and `Keys` path are stable.

## Purpose

Implement the two main post-MVP identity extensions:

- user-keyed tables,
- the `Families` resource layered on `Keys`.

## Required Reading

- `future/plan/specification.md`
- `future/plan/standards.md`
- `future/07-keyless-and-user-keyed-tables.md`
- `AGENTS.md`

## Why These Belong Together

They are both identity extensions:

- user-keyed tables answer "the row has stable identity, but the engine should not invent it",
- `Families` answers "`Keys` need a first-class relationship graph".

Neither should block the core rewrite, but both are important enough to deserve explicit follow-up instead of a vague "later".

## User-Keyed Tables

The post-MVP engine should support a table mode where:

- the user provides the identity value,
- the engine stores and indexes it,
- lookups happen by that user key,
- the key policy is explicit rather than accidental.

Open semantic choices this task must decide:

- is the key unique?
- is insertion duplicate-sensitive?
- is upsert supported?
- is the index ordered, hashed, or configurable?

The simplest first answer is:

- unique indexed keys,
- duplicate insertion rejected,
- explicit replace/upsert as separate later operations.

## API Direction

Examples worth considering:

```rust
database.table::<Ghost>().user_key::<GhostId>();
query::find_by::<GhostId>(id);
```

or:

```rust
insert_indexed.one((ghost_id, ghost_row));
```

The exact syntax can change. The important part is that user-keyed storage is a deliberate
 extension layered on the storage core.

## `Families`

The planned `Families` resource should conceptually maintain, for each managed `Key`:

- optional parent,
- optional first child,
- optional next sibling.

This sparse linked representation was already selected conceptually because it:

- avoids `Vec<Key>` child lists,
- allows all family traversals from those three relations,
- leaves room for future lower-level optimization.

## First Implementation Strategy For `Families`

Do not start with atomic micro-optimizations.

Start with:

- scheduled or batched updates,
- explicit maintenance rules,
- strong consistency under add/remove/reparent operations.

Only later should the project investigate whether some relationship updates can use atomics to relax access requirements.

## Consistency Rules

This task must define and test what happens for:

- adoption of a child by a parent,
- rejection or detachment of a child,
- destruction of a keyed row with children,
- moving a child between parents,
- removal of siblings from the linked list.

Do not leave these as implied behavior.

## Relationship To Keyed Storage

`Families` depends on `Keys`, not on keyless rows and not necessarily on user-keyed tables.

That means:

- the resource should layer cleanly on Task 08,
- it should not leak into the keyless fast path,
- queries over families should remain explicit.

## Implementation Checklist

1. Define the metadata and indexing extension needed for `UserKeyed`.
2. Implement user-key index storage and lookup.
3. Define duplicate/replace semantics.
4. Implement the `Families` resource on `Keys`.
5. Define reparenting and removal rules.
6. Add tests for:
   - user-key lookup,
   - duplicate user-key handling,
   - family traversal,
   - reparenting correctness,
   - key removal and family cleanup.

## Pitfalls

### Pitfall: Smuggling user-keyed overhead into all tables

Keep it opt-in just like `Keys`.

### Pitfall: Starting `Families` with low-level atomic tricks

Correctness and explicit semantics come first.

### Pitfall: Leaving duplicate or removal semantics underspecified

Identity extensions are defined by their edge cases.

## Done Criteria

This task is done when:

- user-keyed tables are a real identity extension,
- `Families` exists with explicit maintenance rules,
- both features are clearly layered on top of the stable core rewrite rather than tangled into it.
