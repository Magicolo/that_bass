# Field Granularity And Layout

This document explores how far `that_bass` should go toward more granular storage than today's "one `Datum` = one column" model.

## Current Situation

Today the storage model is:

- one schema type list per table,
- one column per datum type,
- one lock per column,
- one row move copies or drops whole datum values.

That is already good for:

- dense component iteration,
- cache-friendly joins over full datums,
- simple trait derivation.

But it leaves performance on the table when:

- systems only touch a subset of a large datum,
- a large datum contains a hot numeric prefix and a cold heap-heavy suffix,
- locks conflict on a whole datum even if two systems want disjoint fields,
- SIMD/vectorization wants narrow homogeneous lanes.

## Proposal A: Optional Leaf-Column Decomposition

### Idea

Allow selected datum types to be flattened into leaf fields, so:

```rust
struct Position {
    x: f32,
    y: f32,
    z: f32,
}
```

can become storage equivalent to:

- `Position.x: [f32]`
- `Position.y: [f32]`
- `Position.z: [f32]`

instead of one `[Position]`.

### Why it is attractive

- Smaller working sets when only one or two fields are touched.
- Better SIMD and auto-vectorization opportunities.
- Finer-grained locks if lock-based execution remains.
- Better false-conflict reduction between systems.

### Why it is dangerous

- The whole trait system becomes more complex.
- Query ergonomics degrade fast if users have to think in field paths all the time.
- Moving or dropping values becomes much harder for non-trivial nested types.
- You lose the simplicity of "a datum is one runtime `TypeId`".

## Three Variants

## Variant 1: Full Leaf SoA

Every supported struct datum is recursively flattened into leaves.

### Upside

- Maximum selectivity.
- Maximum potential parallelism on field-disjoint workloads.
- Best numeric-kernel friendliness.

### Downside

- Too complex for a first redesign.
- Terrible fit for arbitrary Rust types.
- Hard to keep the API intuitive.
- Hard to preserve event/type semantics when one logical datum becomes many physical columns.

### Verdict

Too radical as a default storage model.

## Variant 2: Hot/Cold Split Datums

A datum can opt into a split like:

- hot inline fields stored in tight columns,
- cold tail stored in a sidecar object or blob column.

Example:

```rust
struct Transform {
    position: Vec3,
    rotation: Quat,
    scale: Vec3,
    debug_name: String,
    authoring_notes: Vec<String>,
}
```

becomes something closer to:

- hot `TransformHot { position, rotation, scale }`
- cold `TransformCold { debug_name, authoring_notes }`

### Upside

- Captures a huge amount of real-world value.
- Keeps hot simulation data packed.
- Avoids moving big heap-owned blobs during archetype transitions.
- Still preserves a sane logical component boundary.

### Downside

- Users must opt in or annotate.
- Accessing a full datum becomes more expensive than accessing the hot prefix.
- The derive surface grows.

### Verdict

This is the most practical first step if the project wants more granular storage without abandoning ergonomic datums.

## Variant 3: Query Packs / Owning Groups

Do not split datums globally. Instead, create extra packed layouts for hot query combinations.

Example:

- The profiler sees `(&mut Position, &Velocity, &Mass)` dominating frame time.
- The engine builds or asks the user to declare a "pack" for that join.
- The pack arranges those columns identically and maybe even stores them in an execution-friendly microchunk.

This is conceptually close to EnTT full-owning groups.

### Upside

- Targets the actual hot path.
- Avoids imposing field-level complexity on every datum.
- Lets rare or cold components stay simple.

### Downside

- Introduces multiple physical layouts for logically related data.
- Needs invalidation, maintenance, and profiling or explicit declaration.
- Easy to overengineer.

### Verdict

Very interesting as an advanced feature, but not a base storage model.

## API Design Options

The API challenge is more important than the storage challenge.

## Option A: Explicit field queries

Users query field paths directly.

Possible shapes:

- `query::<(&mut field!(Position.x), &field!(Velocity.x))>()`
- generated handles like `Position::X`, `Position::Y`, `Position::Z`

### Good

- Honest.
- Lets users write extremely explicit hot code.

### Bad

- Ugly.
- Feels like a kernel DSL, not a normal Rust API.

## Option B: Keep whole-datum queries, specialize under the hood

Users still write:

```rust
query::<(&mut Position, &Velocity)>()
```

but the engine internally materializes row proxies that borrow split fields.

### Good

- Preserves ergonomics.
- Lets most users ignore layout details.

### Bad

- Proxy/reference machinery becomes subtle.
- Harder to make the unsafe boundaries obvious.

## Option C: Dual API

- whole-datum queries for normal users,
- field queries for experts and generated kernels.

### Good

- Best balance.

### Bad

- More surface area.

### Verdict

Dual API is the right answer if field-level storage is adopted at all.

## What Should Count As Splittable?

Do not flatten arbitrary Rust types.

Instead, introduce explicit storage classes.

Example:

- `#[derive(Datum)]` keeps current behavior.
- `#[derive(Datum, HotFields)]` or `#[datum(layout = "split")]` opts into field-aware layout.
- Restrict split layout to types whose fields are themselves supported storage leaves.

Possible leaf classes:

- POD scalars and vectors,
- fixed-size arrays of POD,
- maybe nested structs composed entirely of splittable leaves.

Not good first candidates:

- `String`
- `Vec<T>`
- `HashMap`
- `Box<T>`
- graph-like pointer-rich structures

These should usually live in side storage or cold blobs.

## Locking Implications

If `that_bass` remains lock-based, finer storage changes the conflict model:

- `&mut Position` may lock three or more leaf columns.
- `&mut Position.x` can lock just one.
- `&Position` plus `&mut Position.y` becomes legal if the API allows field-level alias rules.

That sounds great, but it makes access declaration much more complex.

If the project instead moves toward scheduler-first execution, this complexity becomes easier to manage:

- the scheduler reasons on leaf resources,
- runtime locks disappear from the hot path,
- field-split access turns into a scheduling granularity win.

This is another reason not to do full field decomposition before building the scheduler story.

## Structural Move Implications

Field-level layout changes the cost model of add/remove/move:

- copies become more selective,
- drops become more selective,
- cold sidecars can remain stable while hot data migrates,
- large object movement can drop sharply.

This is especially compelling for:

- transform trees,
- animation state,
- physics state,
- render instance data,
- gameplay tags plus hot numeric state.

## Recommended Shape

If `that_bass` explores more granular columns, the best progression is:

1. Add storage classes:
   - dense table,
   - sparse/side storage,
   - hot/cold split.
2. Allow optional field splitting only for explicitly annotated hot POD-style datums.
3. Keep the main API datum-centric.
4. Add an expert field-query API later if needed.
5. Do not flatten arbitrary nested datums by default.

## Strong Conclusion

The project should not immediately turn every struct into leaf columns.

It should instead:

- introduce hot/cold and side-storage classes,
- let a subset of datums opt into field-aware layout,
- and only after chunking plus scheduling are in place, consider deeper field-level parallelism.

That gives most of the benefit without turning the library into a storage compiler too early.
