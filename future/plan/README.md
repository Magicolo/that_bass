# Rewrite Plan Index

This folder is the concrete rewrite plan for `that_bass`.

If you are new to the rewrite, read in this order:

1. [specification.md](/home/goulade/Projects/rust/that_bass/future/plan/specification.md)
2. [standards.md](/home/goulade/Projects/rust/that_bass/future/plan/standards.md)
3. [00-foundation.md](/home/goulade/Projects/rust/that_bass/future/plan/00-foundation.md)
4. [01-identity-and-metadata.md](/home/goulade/Projects/rust/that_bass/future/plan/01-identity-and-metadata.md) through [10-validation-and-migration.md](/home/goulade/Projects/rust/that_bass/future/plan/10-validation-and-migration.md) for the MVP rewrite
5. [11-direct-exclusive-api.md](/home/goulade/Projects/rust/that_bass/future/plan/11-direct-exclusive-api.md) and later for post-MVP extensions

Before reading these files, use:

- [AGENTS.md](/home/goulade/Projects/rust/that_bass/AGENTS.md) for the current crate
- [future/README.md](/home/goulade/Projects/rust/that_bass/future/README.md) for the proposal landscape

## File Map

- [specification.md](/home/goulade/Projects/rust/that_bass/future/plan/specification.md)
  - The selected architecture, design decisions, non-goals, and open questions.
- [standards.md](/home/goulade/Projects/rust/that_bass/future/plan/standards.md)
  - Mandatory coding standards, testing rules, unsafe-code rules, and preserved codebase habits.
- [00-foundation.md](/home/goulade/Projects/rust/that_bass/future/plan/00-foundation.md)
  - Rewrite lane, benchmarks, instrumentation, and guardrails.
- [01-identity-and-metadata.md](/home/goulade/Projects/rust/that_bass/future/plan/01-identity-and-metadata.md)
  - Schema, table, physical column, and scheduler resource identity.
- [02-chunk-layout.md](/home/goulade/Projects/rust/that_bass/future/plan/02-chunk-layout.md)
  - Chunk sizing, growth, and single-allocation layout.
- [03-keyless-rows.md](/home/goulade/Projects/rust/that_bass/future/plan/03-keyless-rows.md)
  - Keyless row handles and row-targeted structural commands.
- [04-query-surface.md](/home/goulade/Projects/rust/that_bass/future/plan/04-query-surface.md)
  - Query surface, chunk views, and access analysis.
- [05-schedule-builder.md](/home/goulade/Projects/rust/that_bass/future/plan/05-schedule-builder.md)
  - Schedule building, dependency graph, and happens-before rules.
- [06-executor-runtime.md](/home/goulade/Projects/rust/that_bass/future/plan/06-executor-runtime.md)
  - Executor runtime, stealable jobs, and dynamic injection.
- [07-command-resolution.md](/home/goulade/Projects/rust/that_bass/future/plan/07-command-resolution.md)
  - Command buffers, batched resolution, and visibility.
- [08-managed-keys.md](/home/goulade/Projects/rust/that_bass/future/plan/08-managed-keys.md)
  - Managed keys, inline `Key` columns, and reverse mapping.
- [09-global-tables.md](/home/goulade/Projects/rust/that_bass/future/plan/09-global-tables.md)
  - Table-backed globals and `query::one(...)`.
- [10-validation-and-migration.md](/home/goulade/Projects/rust/that_bass/future/plan/10-validation-and-migration.md)
  - Validation, migration, and documentation discipline.
- [11-direct-exclusive-api.md](/home/goulade/Projects/rust/that_bass/future/plan/11-direct-exclusive-api.md)
  - Direct exclusive API and the non-scheduled escape hatch.
- [12-user-keys-and-families.md](/home/goulade/Projects/rust/that_bass/future/plan/12-user-keys-and-families.md)
  - User-keyed tables and `Families`.
- [13-advanced-queries.md](/home/goulade/Projects/rust/that_bass/future/plan/13-advanced-queries.md)
  - Relational operators, deferred value writes, and advanced nested access.
- [14-events.md](/home/goulade/Projects/rust/that_bass/future/plan/14-events.md)
  - Event model reintroduction.
- [15-layout-specialization.md](/home/goulade/Projects/rust/that_bass/future/plan/15-layout-specialization.md)
  - Optional column decomposition and layout specialization.

## Reading Strategy

If you are implementing the rewrite from scratch:

1. Read the specification.
2. Read the standards.
3. Implement tasks `00` through `10` in order unless a task explicitly says otherwise.
4. Treat tasks `11` and later as important follow-up work, not prerequisites for the first coherent milestone.

## Maintenance Rule

If a significant architectural decision changes:

- update [specification.md](/home/goulade/Projects/rust/that_bass/future/plan/specification.md),
- update [standards.md](/home/goulade/Projects/rust/that_bass/future/plan/standards.md) if implementation standards changed,
- update the relevant task file,
- update [AGENTS.md](/home/goulade/Projects/rust/that_bass/AGENTS.md),
- and update any proposal docs in `future/` that are now stale.
