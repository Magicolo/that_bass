# Radical Ideas

This file intentionally goes beyond "reasonable next step" proposals.

Some of these are likely too big for the near term.
They are still worth documenting because they reveal what kind of library `that_bass` could become if performance was allowed to dominate almost every other concern.

## Idea 1: Profile-Guided Layout Optimization

### Concept

Ship the runtime with instrumentation that records:

- which queries dominate frame time,
- which component sets are co-iterated most,
- which fields are touched,
- where contention or sync points occur,
- where structural churn is highest.

Then use that information to suggest or automatically build:

- owning groups,
- hot query packs,
- better chunk sizes,
- hot/cold splits,
- field-level decomposition.

### Why it is interesting

Most games are not bottlenecked by all data equally.
They are bottlenecked by a small set of systems and component combinations.

Static design guesses are weaker than profile-informed layout choices.

### Why it is scary

- Much more machinery.
- Hard to explain and debug.
- Risk of building a magical system nobody fully understands.

### Best version

Make it advisory first:

- collect profiles,
- emit suggestions,
- let the user opt into packs/layout hints.

## Idea 2: Self-Tuning Owning Groups

Inspired by EnTT full-owning groups.

### Concept

Let users or the profiler declare:

```rust
database.optimize::<(&mut Position, &Velocity, &Mass)>();
```

This creates a specialized packed layout or execution pack for that join.

### Benefits

- Best-possible hot loop performance for top systems.
- Preserves simpler generic storage for everything else.

### Costs

- Additional maintenance during structural edits.
- Hard decisions around ownership and invalidation.

### Opinion

Worth exploring after chunking and scheduler work, not before.

## Idea 3: Epoch Or MVCC World Snapshots

### Concept

Systems read from a stable world snapshot while writes build the next epoch.

Variants:

- full double buffer for hot world state,
- chunk-level copy-on-write,
- versioned chunk metadata with deferred reclamation.

### Benefits

- Lockless reads.
- Natural fit for rollback networking and deterministic replays.
- Strong frame semantics.

### Costs

- Memory blowup.
- Complex reclamation.
- Hard component pointer/reference semantics.

### Opinion

Very attractive for rollback-heavy or networking-heavy games, but likely too much as the default storage engine.

## Idea 4: GPU-Mirrored Hot Columns

### Concept

Allow selected hot numeric columns or chunk groups to maintain a GPU-friendly mirror.

Use cases:

- transforms,
- particles,
- visibility or culling data,
- crowd/boid simulation,
- skinning inputs.

### Benefits

- Lets the storage model become the data backbone for compute pipelines.
- Reduces copy/packing overhead between gameplay and rendering/compute.

### Costs

- Synchronization complexity.
- API becomes hardware-aware.
- Platform portability and debugging become harder.

### Opinion

A long-term differentiator, not a foundational redesign.

## Idea 5: Query Compilation Or Codegen

### Concept

Generate specialized iteration kernels for hot queries.

Possible routes:

- macro/codegen at compile time,
- monomorphized chunk kernels,
- profile-guided precompiled kernels,
- maybe even dynamic JIT in tools/server builds.

### Benefits

- Better inlining and vectorization.
- Less generic dispatch overhead.
- Clean way to specialize field-split proxies.

### Costs

- Build complexity.
- Harder debugging.
- Risk of baking in too much compile-time surface.

### Opinion

Promising only after the data layout stops moving.

## Idea 6: Spatially Sharded Worlds

### Concept

Instead of one giant world, split the world by:

- map region,
- simulation island,
- cell,
- replication shard,
- ownership domain.

Then allow:

- cross-shard references,
- local schedulers per shard,
- occasional merge or transfer.

### Benefits

- Better NUMA and cache locality.
- Less contention.
- Easier streaming and server scaling.

### Costs

- Cross-shard queries become harder.
- Graph-like relationships get more complex.

### Opinion

Very strong for large worlds and servers. Less essential for small single-scene games.

## Idea 7: Relationship Store Separate From Archetype Store

### Concept

Keep pure component archetype storage separate from graph/relationship storage.

Use a dedicated structure for:

- parent/child,
- inventory,
- attachment graphs,
- ownership graphs,
- AI links,
- quest graphs.

### Benefits

- Lets component storage stay dense and simple.
- Makes graph queries explicit rather than overloading components.

### Costs

- Another subsystem to schedule and synchronize.

### Opinion

Probably worthwhile if the project wants to support richer game-state queries without contaminating archetype storage.

## Idea 8: First-Class Frame Transaction Log

### Concept

Make command buffers and merges first-class logs that can also power:

- replay,
- networking replication,
- rollback,
- observer/event generation,
- debugging timelines.

### Benefits

- One mechanism can solve multiple systems problems.
- Easier deterministic debugging.

### Costs

- Requires careful event/command semantics.
- Log volume can explode.

### Opinion

Potentially one of the smartest long-term moves if the project also cares about tooling and networking.

## Idea 9: Storage Classes Beyond Dense Tables

The project should probably evolve toward multiple storage classes:

- dense chunked tables for hot archetype data,
- sparse side stores for volatile or rarely-joined data,
- blob/cold storage for large heap-rich datums,
- maybe compressed vectors for repeated data.

This is less "wild" than some other ideas, but the full version is a major redesign because it changes the assumption that one schema implies one storage shape.

## Idea 10: `that_bass` As A World Compiler

### Concept

At the radical end, the library stops being only a runtime and becomes a compiler of:

- system access graphs,
- data layouts,
- chunk topologies,
- hot packs,
- sync point placement.

The user writes simple systems and layout hints; the engine generates the execution strategy.

### Benefits

- Could produce astonishing performance for stable workloads.

### Costs

- Enormous complexity.
- Hard to keep transparent.

### Opinion

Interesting as a research identity, dangerous as a product identity.

## Strong Conclusion

The most promising radical bets are:

- profile-guided layout suggestions,
- owning-group style hot packs,
- frame transaction logs,
- optional GPU mirrors,
- maybe MVCC snapshots for rollback-heavy uses.

The least advisable near-term radical move is "fully flatten all data and compile everything".

The best version of `that_bass` still needs to feel understandable to engine programmers. Radical ideas should make the hot path faster, not make the whole engine inscrutable.
