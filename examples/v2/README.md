# `v2` Example Usage

This folder contains runnable sample usage for the rewrite lane.

Run the current example with:

```bash
cargo run --example v2
```

Recommended read order:

- `main.rs` for the newcomer-facing `v2` walkthrough,
- `store_planning.rs`, `global_tables.rs`, `schedule_builder.rs`,
  `executor_runtime.rs`, `command_resolution.rs`, and `managed_keys.rs`
  for the intended first-iteration API,
- the remaining files only when you need lower-level or more focused demonstrations.

Why this folder exists:

- it gives a newcomer one concrete place to see the current `v2` public API in use,
- it makes public-surface drift visible during the rewrite,
- it provides a lightweight integration point that should evolve with the rewrite tasks.
- it now includes the newcomer-facing first-iteration surface and a smaller set of focused,
  advanced demonstrations that still help validate ongoing rewrite tasks.

Maintenance rule:

- when the public `v2` API changes, update `examples/v2/` in the same patch,
- when a new public capability becomes important enough for a newcomer to copy, add a focused example for it here.
