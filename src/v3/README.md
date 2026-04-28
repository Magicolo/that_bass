# that_bass v3: Fragment-Based ECS Architecture

This directory contains the experimental `v3` architecture for `that_bass`. It transitions from the chunked storage of `v2` to a **fragment-based** storage model and introduces a new hierarchical declaration API for dependency analysis and execution planning.

## Core Concepts

### 1. Fragment-Based Storage
In `v3`, tables are no longer partitioned into fixed-size chunks. Instead, they use a **Fragment** model.
- **Contiguous Columns**: Each column in a table is a contiguous allocation.
- **Gaps (Free List)**: Instead of compacting rows immediately on removal, `v3` maintains a "free list" of ranges (gaps).
- **Fragments**: A fragment is defined as a contiguous range of inhabited rows between two gaps or between a gap and the end of the column.
- **Dynamic Merging**: Fragments are created when a row is removed (splitting a fragment) and merged when a gap is filled by an insertion or compaction.

#### Trade-offs:
- **Pros**: Zero-copy removals (just add to the free list), potentially better cache locality for sparse-but-contiguous workloads.
- **Cons**: Requires more sophisticated iteration logic (skipping gaps); "Swiss cheese" fragmentation can hurt performance if not managed.

### 2. Hierarchical Declarations (`Declare` API)
The `Declare` API allows tasks to specify exactly what they access and at what **granularity** they execute.

#### Granularity & Scope:
The structure of the declaration determines how many times a task executes:
- **`store().read()`**: Executes **once** per schedule run. Accesses the entire `Store`.
- **`store().table().read()`**: Executes **once per table**.
- **`store().table().fragment().read()`**: Executes **once per fragment** (per table).
- **`store().table().fragment().row().read()`**: Executes **once per row**.

#### The Path Model:
Declarations build a tree of `Node`s. The scheduler walks this tree to extract **Resource Paths**.
- `Branch`: Navigates the hierarchy (`Store -> Tables -> Table -> Fragments -> Fragment -> Columns -> Column -> Row`).
- `Leaf`: The terminal access mode (`read`, `write`, `has`).

### 3. Resource Dependency Analysis
The scheduler analyzes declared paths to prevent data races and optimize execution order.
- **Monotone Paths**: All dependencies are expressed as paths from the `Store` root.
- **Conflict Rules**:
    - `Read` vs `Read`: Never conflicts.
    - `Write` vs `Anything`: Conflicts if one path is a prefix of another or if they are identical.
- **Wildcards**: Declarations like `table()` act as wildcards, representing "all resources of this type at this level".

## Execution Model

1. **Planning**: `Schedule::push` analyzes the `Declare` tree and extracts resource dependencies.
2. **Expansion**: At runtime, the scheduler expands wildcard declarations into concrete jobs based on the current state of the `Store` (e.g., how many tables and fragments currently exist).
3. **Execution**: The executor runs jobs according to the dependency graph. Tasks receive their data through a `prepare::Context` which provides slices (for fragments) or references (for rows/tables).

## Why v3?
`v3` aims for a more "honest" representation of memory. By making fragments first-class citizens, we allow the scheduler to reason about the actual contiguous blocks of data available, which is critical for SIMD optimizations and efficient multi-threading in complex simulations.
