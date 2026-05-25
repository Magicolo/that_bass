# Task 02: Per-Module-Group Aliasing Safety

## Goal

Prevent modules within a composed group from producing aliased `&mut`
references that violate Rust's aliasing rules.  Currently `(Write<T>,
Write<T>)` compiles silently, and `Store::with` carries a TODO acknowledging
the unsoundness.

## The Problem

`Module for (M0, M1)` validates each sub-module independently:

```rust
impl<M0: Module, M1: Module> Module for (M0, M1) {
    fn initialize(&self, store: &mut Store) -> Result<Self::State, Error> {
        Ok((self.0.initialize(store)?, self.1.initialize(store)?))
    }
}
```

Each sub-module's `initialize` succeeds on its own.  No cross-module check
happens.  If `M0` and `M1` both declare write access to column `T` in the
same table, the composed group will produce two `&mut [T]` slices pointing
at the same memory — undefined behavior.

The same issue exists for query modules specifically (two `Write<T>` in one
`Access` tuple), though the current `Access for (A0, A1)` impl at least
shares the same `&table::Table` and could detect the conflict at
`initialize` time within a single query.  The cross-module case is harder
because different module types (query, insert, remove) access the store
through different mechanisms.

## Existing Patterns

### v1: `DeclareContext` (src/v1/row.rs:79-118)

Every `Row` implementation must call `declare` with a `DeclareContext`
before `initialize`.  The context is a `HashSet<Access>` (where `Access` is
`Read(TypeId)` or `Write(TypeId)`).  Within a single `Row` type:

- Read-Write conflict → error
- Write-Write conflict → error
- Read + Read → ok

This validates one declaration tree.  The pattern can be extended to
validate multiple trees against each other.

### v2: `Dependency` paths (src/v2/schema.rs:211-244)

Each unit of work declares a `Dependency` with a path (e.g. `[store(x),
table(y), chunk(None), column(col_z)]`) and an access mode (`Read`/`Write`).
Two dependencies conflict if:

1. Their access modes conflict (Write vs Read, Write vs Write, Read vs
   Write — all conflict except Read vs Read).
2. Their paths overlap (both point at the same resource, or one has a
   wildcard at a shared depth).

### v2: `conflicts_with` on `Analysis` (src/v2/query.rs:478-507)

Two analyzed queries conflict if any declared access pair shares an
identifier AND conflicting modes (with a fast-path rejection via disjoint
filter proofs).

## v4's Architecture Advantage

Unlike v1/v2 where arbitrary operations can interleave freely, v4's `State`
system provides a structural guarantee:

> **A `State` handles exactly one module group at a time.**

This means validation only needs to happen *within* a single
`Module::initialize` call for the composed group.  The `State` itself is the
locking/sequencing point.  No two groups are active simultaneously, so
cross-group conflicts are impossible by construction.

The problem reduces to: **validate that all sub-modules within one composed
module group have non-conflicting accesses to the store.**

## Proposed Design

### Step 1: Add a `declare` method to `Module`

```rust
pub trait Module {
    type Item<'a> where Self: 'a;
    type State;

    /// Declare what resources this module accesses and how.
    /// Called before `initialize`.  Modules that don't access
    /// store resources return an empty declaration.
    fn declare(&self) -> Declaration { Declaration::none() }

    fn initialize(&self, store: &mut Store) -> Result<Self::State, Error>;
    fn update(&self, state: &mut Self::State, store: &Store) -> Result<bool, Error>;
    fn get<'a>(&'a self, state: &'a Self::State, store: &'a Store) -> Self::Item<'a>
        where Self: 'a;
}
```

### Step 2: Define a `Declaration` type

```rust
/// A set of resource accesses declared by a module.
pub struct Declaration {
    accesses: Vec<DeclaredAccess>,
}

pub struct DeclaredAccess {
    /// The kind of resource: "column of type T", "table by schema", etc.
    resource: Resource,
    access: Access,
}

pub enum Access {
    Read,
    Write,
}

pub enum Resource {
    /// A column identified by TypeId in any table with that column.
    Column(TypeId),
    /// A specific table identified by its schema (sorted TypeIds).
    Table(Vec<TypeId>),
    /// Any table (for operations that scan all tables).
    AllTables,
}

impl Access {
    /// Two accesses conflict if one is Write.
    pub const fn conflicts_with(self, other: Self) -> bool {
        matches!(
            (self, other),
            (Self::Write, Self::Read) | (Self::Write, Self::Write) | (Self::Read, Self::Write)
        )
    }
}

impl Resource {
    /// Two resources may overlap (i.e. could refer to the same store data).
    pub fn may_overlap(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Column(a), Self::Column(b)) => a == b,
            (Self::Table(a), Self::Table(b)) => a == b,
            (Self::AllTables, _) | (_, Self::AllTables) => true,
            _ => false,
        }
    }
}

impl Declaration {
    pub fn conflicts_with(&self, other: &Self) -> bool {
        self.accesses.iter().any(|a| {
            other.accesses.iter().any(|b| {
                a.resource.may_overlap(&b.resource)
                    && a.access.conflicts_with(b.access)
            })
        })
    }
}
```

### Step 3: Validate composed modules

Override the tuple impl to call `declare` on each sub-module and check for
conflicts:

```rust
impl<M0: Module, M1: Module> Module for (M0, M1) {
    fn declare(&self) -> Declaration {
        let mut d = self.0.declare();
        d.merge(self.1.declare());
        d
    }

    fn initialize(&self, store: &mut Store) -> Result<Self::State, Error> {
        // Validate before initializing
        let d0 = self.0.declare();
        let d1 = self.1.declare();
        if d0.conflicts_with(&d1) {
            return Err(Error::AccessConflict);
        }
        Ok((self.0.initialize(store)?, self.1.initialize(store)?))
    }
    // ...
}
```

### Step 4: Implement `declare` on existing modules

- `()` → `Declaration::none()` (default impl)
- `query::Read<T>` → `Declaration::single(Resource::Column(TypeId::of::<T>()), Access::Read)`
- `query::Write<T>` → `Declaration::single(Resource::Column(TypeId::of::<T>()), Access::Write)`
- `query::Row` → `Declaration::single(Resource::AllTables, Access::Read)`
- `query::Table` → `Declaration::single(Resource::AllTables, Access::Read)`
- `query::ReadWith` → `Declaration::single(Resource::Column(meta.identifier), Access::Read)`
- `template::Insert<T>` → `Declaration` derived from `T::declare()` (each column gets a `Write` entry)
- `Remove` → `Declaration::single(Resource::AllTables, Access::Write)`

The `query::Module<A>` wrapper delegates `declare` to the inner access
pattern:

```rust
impl<A: Access> module::Module for query::Module<A> {
    fn declare(&self) -> Declaration {
        // Walk the Access tuple tree, collect all column claims
        collect_accesses::<A>()
    }
}
```

### Step 5: Build the `Declaration` from `Access` trait automatically

The `Access` trait already knows which columns it accesses.  We can add a
companion function or an associated constant to surface this:

```rust
pub trait Access {
    type State;
    type Item<'a> where Self: 'a;

    fn initialize(&self, table: &table::Table) -> Option<Self::State>;
    fn get<'a>(&'a self, state: &'a Self::State, table: &'a table::Table) -> Self::Item<'a>
        where Self: 'a;

    /// Declare what this access pattern reads/writes.
    fn declare(&self, out: &mut Vec<DeclaredAccess>);
}
```

Then `collect_accesses` recursively calls `declare` on each element of the
tuple.

## Challenges

1. **Granularity**.  Should we track access at the column level, table
   level, or both?  Column-level is most precise.  Table-level is simpler and
   catches the common cases.  Start with column-level and refine if needed.

2. **Dynamic resources**.  `ReadWith(Meta)` and `WriteWith` have runtime
   `Meta`, not compile-time `TypeId`.  `Resource::Column` can use `TypeId`
   from the `Meta` — this works because the conflict check is structural
   (same TypeId + conflicting mode).

3. **Table schema as a resource**.  An insert that creates a new table
   changes which tables exist.  Should this be declared?  Probably not for
   conflict detection (table creation is additive and doesn't alias existing
   data), but it matters for ordering (Task 01 touch on this).

4. **Error types**.  The conflict error should identify which modules
   conflict and on which resource, for debuggability.  Use `Box<dyn Error>`
   or a dedicated `Conflict { left: &'static str, right: &'static str,
   resource: TypeId }` variant.

5. **`state::Module<M>` double-wrap**.  The `state::Module<M>` wrapper
   implements `module::Module` for `Module<(H, T)>` using `ref_cast`
   splitting.  The `declare` method must also work through this wrapper.
   Since `split_ref` gives `&Module<H>` and `&Module<T>`, we can call
   `declare` on each to build the merged declaration.

6. **Relation to `Store::with`**.  Once `Module for (M0, M1)` validates
   declarations, `Store::with` becomes safe for composed modules.  The TODO
   can be resolved.

## Resources to Study

- `src/v1/row.rs:79-118` — `DeclareContext` API and RW/WW conflict detection
- `src/v2/schema.rs:211-244` — `Dependency` path-based resource model
- `src/v2/schema.rs:166-208` — `Resource` enum with wildcard support
- `src/v2/schema.rs:1632-1692` — `ColumnAccess::dependency` bridging
- `src/v2/query.rs:478-507` — `Analysis::conflicts_with` with filter
  fast-path
- `src/v2/schedule.rs:958-1007` — `conflict` and `covers` functions

## Success Criteria

- [ ] `Module` trait gains a `declare` method (with default returning empty).
- [ ] `Declaration` type with resource + access tracking.
- [ ] `Module for (M0, M1)` validates declarations before initializing.
- [ ] Composing two `Write<T>` modules via `State::build().push(...)` returns
      `Err(Error::AccessConflict)`.
- [ ] Composing `Read<T>` + `Write<T>` for different T succeeds.
- [ ] Composing `Read<T>` + `Write<U>` for different U succeeds.
- [ ] Composing `Write<T>` + `Read<T>` returns `Err(Error::AccessConflict)`.
- [ ] The TODO on `Store::with` is removed or updated to note it's now safe.
- [ ] Build passes with no new warnings.
