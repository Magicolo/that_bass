# Task 14: Event Model Reintroduction, Phase Semantics, And Identity-Aware Payloads

This task brings events back after the core rewrite semantics are stable.

Read this file together with `future/plan/specification.md` and `future/plan/standards.md`. The current crate has a meaningful event system, but the rewrite changes the visibility model enough that event timing and payload identity must be redesigned instead of copied forward blindly.

## Purpose

Reintroduce structural events and observers in a way that matches:

- scheduled execution,
- resolve-job visibility,
- keyless-by-default tables,
- phase-aware semantics.

## Required Reading

- `future/plan/specification.md`
- `future/plan/standards.md`
- `AGENTS.md`
- current event behavior in `src/event.rs`

## Why This Was Deferred

The current event system is tightly coupled to the old structural timing model.

The rewrite changes:

- when structure becomes visible,
- when later queries can observe new rows,
- how identity works for keyless tables,
- how frame phases and resolve jobs define ordering.

That means event timing must be redefined intentionally.

## Core Design Questions

This task must answer:

1. Are events emitted at resolve time, at phase boundaries, or both?
2. What payloads exist for keyless tables?
3. Which event families are worth keeping:
   - `Create`
   - `Destroy`
   - `Modify`
   - `OnAdd`
   - `OnRemove`
4. What buffering or retention semantics still make sense in a scheduler-first engine?

## Identity-Aware Payloads

Keyed tables and keyless tables should not pretend to have the same event identity semantics.

Likely direction:

- keyed tables can emit stable `Key` payloads,
- keyless tables may need:
  - table/chunk summaries,
  - snapshot payloads,
  - or explicit "no stable identity" event forms.

This must be designed carefully so keyless tables do not accidentally promise identity they do not have.

## Suggested Phasing Model

A good first direction is:

- structural commands resolve,
- resolve jobs emit structural event records,
- observers consume those records at defined phase boundaries or immediately after resolve depending on configuration.

This keeps event timing tied to the actual structural visibility model.

## Buffering And Retention

The current crate has meaningful listener retention behavior. The rewrite should re-evaluate, not blindly preserve, that complexity.

First questions:

- do listeners need replay?
- do they need bounded history?
- should retention be frame-based instead of listener-count-based?

The rewrite may want a simpler first event model than the current crate.

## Relationship To Query And Scheduler APIs

If events are reintroduced, they should fit the same scheduler story:

- event readers are scheduled consumers,
- event writers are resolve jobs or explicit emit commands,
- event ordering follows the same phase and happens-before model.

Do not introduce a side-channel that bypasses the scheduler's semantics.

## Implementation Checklist

1. Decide event timing model.
2. Decide payload model for keyed versus keyless tables.
3. Reintroduce a minimal set of structural event families.
4. Define buffering/retention semantics.
5. Add tests for:
   - event timing relative to resolve,
   - keyed payload correctness,
   - keyless payload honesty,
   - retention or replay semantics if kept.

## Pitfalls

### Pitfall: Recreating current event behavior before the new visibility model is stable

That will likely produce the wrong abstractions.

### Pitfall: Pretending keyless tables can emit stable row identity

They cannot.

### Pitfall: Making events an unscheduled side channel

They should respect the same phase semantics as the rest of the engine.

## Done Criteria

This task is done when:

- the rewrite has an explicit event timing model,
- payload semantics differ honestly between keyed and keyless tables,
- events fit the scheduler and resolve model instead of bypassing it.
