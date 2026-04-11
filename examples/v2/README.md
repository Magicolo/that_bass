# `v2` Example Usage

This folder contains runnable sample usage for the rewrite lane.

Run the current example with:

```bash
cargo run --example v2
```

Why this folder exists:

- it gives a newcomer one concrete place to see the current `v2` public API in use,
- it makes public-surface drift visible during the rewrite,
- it provides a lightweight integration point that should evolve with the rewrite tasks.

Maintenance rule:

- when the public `v2` API changes, update `examples/v2/` in the same patch,
- when a new public capability becomes important enough for a newcomer to copy, add a focused example for it here.
