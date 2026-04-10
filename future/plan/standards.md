# Coding Standards And Best Practices

This document is mandatory for all rewrite work.

Read it together with:

- `future/plan/specification.md`
- the relevant task file in `future/plan/`
- `AGENTS.md` for the current crate

This file describes how the rewrite must be implemented. The specification describes what is being built. The task files describe in what order and with what deliverables. All three matter.

## Core Mindset

Treat this library as a critical system.

That means:

- write with care and rigor,
- assume performance matters until proven otherwise,
- assume unsafe code must be justified and audited,
- assume concurrency mistakes are correctness bugs,
- prefer explicit invariants over cleverness,
- do not normalize hand-wavy reasoning.

## Naming

Use full descriptive names for identifiers.

Required rules:

- no abbreviations,
- no acronyms unless they are already the canonical type or trait name used by Rust or the ecosystem,
- no single-letter variable names,
- no vague names like `data`, `item`, `value`, or `state` unless the surrounding type makes the meaning precise and unavoidable.

Examples:

- prefer `target_chunk_capacity` over `cap`,
- prefer `physical_column_index` over `column_index` when both logical and physical columns exist,
- prefer `resolve_result` over `result` when several results exist in scope.

If a type, function, or variable is hard to name, that is often a sign the code wants to be split or simplified.

## Functions And Structure

Favor small functions with self-documenting names.

Required rules:

- split large functions into smaller ones,
- use helper names that explain the step being performed,
- do not keep several logically distinct phases in one giant function unless there is a measured reason,
- if a comment can be replaced by a better function name or type name, prefer the refactor.

Good code here should read like a sequence of named invariant-preserving steps.

## Comments And Documentation

Write comments for non-obvious facts only.

Good comment targets:

- safety contracts,
- concurrency or visibility invariants,
- design decisions,
- rejected alternatives,
- unintuitive performance tradeoffs,
- surprising edge cases,
- reasons a particular ordering or data representation must be preserved.

Do not comment obvious syntax.

Comment as if speaking to a newcomer to the library who is careful but has no rewrite context.

When unsafe code exists, safety comments are mandatory.

## Abstractions

Be minimalist with abstractions.

Required rules:

- use concrete types by default,
- abstract only when the abstraction buys meaningful clarity, reuse, or correctness,
- do not add generic layers "just in case",
- do not hide core storage or scheduling behavior behind unnecessary indirection,
- if an abstraction makes performance harder to reason about, it needs a strong justification.

The rewrite should prefer explicit system structure over abstraction-heavy architecture.

## Public API

Keep the public API small.

Required rules:

- every public item must be intentional,
- default to private or crate-scoped items,
- expose only what is needed for the current milestone,
- postpone convenience APIs until the core semantics are stable,
- do not leak internal tuning knobs or temporary design scaffolding into the public surface.

The library will expand carefully over time. It should not start wide.

## Dependencies

Do not reinvent the wheel for problems that are not specific to this library's goals.

Required rules:

- prefer existing well-scoped dependencies over bespoke implementations for generic needs,
- audit each dependency for real value,
- consider compile time cost,
- consider binary size and maintenance cost,
- avoid adding dependencies that hide important performance behavior,
- avoid dependencies that would make unsafe auditing harder without a strong payoff.

This is a performance-oriented library. Dependency count is not the only cost.

## Performance And Memory

Performance is a design constraint, not a final polishing step.

Required rules:

- be thoughtful about allocations,
- if an allocation can be spared without making the code worse, spare it,
- minimize hidden cloning and buffering,
- prefer layouts and algorithms that make cache behavior understandable,
- treat synchronization overhead as part of performance,
- design APIs so benchmarks can exercise every important behavior through the public surface,
- do not add benchmarking-only hooks or benchmark-only code paths to the library.

Benchmarkability is important, but the library source itself must stay about the library, not about benchmarking.

## `core` Versus `std`

Favor `core` over `std` when possible.

This is not a license to make the code awkward. It is a bias:

- use `core` when it keeps the code clear,
- pull in `alloc` or `std` only when actually needed,
- keep low-level building blocks as platform-independent as practical.

## Unsafe Code

Minimize unsafe code paths.

Required rules:

- keep unsafe code as small and contained as possible,
- isolate unsafe operations behind narrow helpers,
- document every unsafe contract,
- make it obvious which invariants callers must uphold,
- prefer safe code unless the unsafe version has a meaningful and understood benefit,
- if a safe refactor can remove unsafe code without hurting the design badly, prefer the safe refactor.

Additional rule:

- any code that directly or indirectly touches unsafe code must be tested under Miri.

This includes:

- direct unsafe blocks,
- helpers called by unsafe code,
- code that depends on aliasing, lifetime, initialization, pointer, or layout invariants maintained by unsafe internals.

## Testing

Code must be testable and tested.

Required rules:

- every meaningful behavior needs tests,
- tests must focus on non-obvious paths and edge cases,
- tests should validate invariants and semantics, not only happy-path examples,
- every time a test would otherwise use an arbitrary value, replace it with generated or sampled data from `checkito`,
- when `checkito` finds a failing minimized case, add a regular regression test for that exact case as well.

The current `checkito` documentation is clear about the intended workflow:

- build generators that describe valid input bounds,
- let shrinking minimize failures,
- rely on exhaustive mode automatically when the domain is small,
- use reproducible sampling when needed,
- keep exact regression tests for known failures.

That should become the default testing style of the rewrite.

## `checkito` Standards

Use `checkito` extensively.

Preferred tools:

- `#[check]` for property-style tests,
- explicit generators for structured inputs,
- shrinking-aware property tests,
- exhaustive small-domain checks,
- reproducible seeded sampling when investigating difficult cases.

Testing rules:

- use generators instead of hand-picked "arbitrary" values,
- encode domain constraints in the generator instead of inside the test body when practical,
- favor generated sequences of operations for state-machine style tests,
- pair generative tests with focused regression tests for discovered edge cases.

The repository already depends on `checkito = "5.0.0"` in `Cargo.toml`. Use it as the primary property-testing tool.

## Miri Standards

Any change that touches unsafe code directly or indirectly must include Miri coverage.

Required rules:

- add or update targeted tests so Miri exercises the unsafe path,
- prefer deterministic and reasonably small tests for Miri,
- do not assume ordinary tests are enough for pointer and aliasing correctness,
- if a change affects a shared unsafe helper, widen Miri coverage to every important caller.

The goal is not symbolic completeness. The goal is aggressive practical auditing of unsafe assumptions.

## Validation And Tooling

Code must compile, be linted, and be formatted at all times.

Required baseline checks after meaningful work:

```bash
cargo fmt
cargo build
cargo clippy --all-targets --all-features
```

Strongly expected:

```bash
cargo test
```

Required when unsafe-touched code changes:

```bash
cargo miri test
```

If those checks are scoped more narrowly for iteration speed, the final patch still needs equivalent coverage for the changed area.

## Current Codebase Standards To Preserve

The current crate already shows several strong engineering habits that should be preserved unless they conflict with the newer standards above.

### Invariants Are First-Class

The existing code treats ordering and uniqueness as semantic invariants:

- metadata lists are sorted and deduplicated,
- query states are sorted,
- lock lists are sorted,
- set-style operations rely on sorted representations.

Carry that forward. Do not treat those properties as incidental optimizations.

### Schema-Level Reasoning Beats Ad Hoc Branching

The current code consistently reasons in terms of schemas, tables, columns, and declared access rather than type-by-type special cases.

Preserve that style in the rewrite.

### Keep Non-Obvious Ordering Comments

The current crate documents strange visibility windows and ordering constraints near the code that enforces them.

That is good practice and should continue.

### Optimize Away Needless Work

The current code already takes zero-sized types and similar special cases seriously.

Keep the same mindset:

- avoid unnecessary locks,
- avoid unnecessary copies and drops,
- avoid unnecessary allocations,
- keep cheap cases cheap.

### Favor Narrow Unsafe Helpers

The current crate often funnels unsafe behavior through narrow helpers such as checked `get_unchecked` wrappers that still validate in debug mode.

That pattern is good:

- small unsafe surface,
- centralized contracts,
- stronger debugging behavior.

### Test Behavior, Not Just Examples

The current `tests/` directory acts as a semantic specification more than as a toy example suite.

The rewrite should preserve that spirit:

- test invariants,
- test movement,
- test visibility windows,
- test conflict behavior,
- test failure modes.

### Reuse Established Coordination Patterns

The current crate uses patterns like `fold_swap` and `try_fold_swap` to prefer progress and fairness over naïve blocking.

The exact mechanisms will change in the rewrite, but the broader standard remains:

- preserve established concurrency reasoning where it is still valid,
- prefer well-understood coordination patterns over ad hoc ones.

### Preserve Coverage For Tuple-Based Surfaces

The current crate relies on tuple macro coverage in several extensible traits.

When the rewrite has similarly extensible surfaces, coverage should remain systematic rather than ad hoc.

## Design For External Verification

The public API must be rich enough that:

- benchmarks,
- property tests,
- regression tests,
- integration tests,

can all be written against public behavior without private hooks.

This is a hard design requirement for the rewrite.

## Final Standard

When in doubt, choose the option that is:

1. easier to reason about,
2. easier to test,
3. easier to benchmark through public behavior,
4. smaller in public surface,
5. smaller in unsafe surface,
6. and no worse for performance.

