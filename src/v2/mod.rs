//! Rewrite lane for the next major architecture of `that_bass`.
//!
//! `v2` exists so the next storage-and-scheduler design can evolve side by side with the stable
//! `v1` engine. The code in this module should be written as if it lived in a separate crate even
//! though it currently ships inside this repository.
//!
//! Primary design references:
//!
//! - `future/plan/00-foundation.md`
//! - `future/plan/specification.md`
//! - `future/plan/standards.md`
//!
//! Boundary rules:
//!
//! - do not depend on the current runtime modules under `that_bass::v1`,
//! - keep the public surface intentionally small,
//! - keep benchmark-specific logic outside the library in `benches/`,
//! - let module names carry context so public type names can stay short and clear.
//!
//! Module map:
//!
//! - `command`: deferred command vocabulary.
//! - `instrumentation`: measurement categories and public diagnostics hooks.
//! - `key`: stable-identity vocabulary used by later extension resources.
//! - `query`: typed query descriptors, inline dense-slice projections, optional chunk views,
//!   filters, and access analysis.
//! - `schedule`: ordering vocabulary for future schedule construction.
//! - `schema`: the metadata catalog, `Meta` descriptors, chunk layouts, tables, chunks, and
//!   resource mapping.
//! - `store`: the foundation store boundary and chunk planning surface.
//!
//! Glossary:
//!
//! `meta`
//!: Type metadata for one stored type. A table stores one `Meta` per declared column.
//!
//! `table`
//!: The storage owner for one collection of `Meta` descriptors. A table owns metadata and a
//! collection of chunks.
//!
//! `store`
//!: The root owner for tables. Scheduler dependency identifiers are hierarchical from store down
//! to columns.
//!
//! `chunk`
//!: A densely packed, independently allocated subset of one table's rows. Chunks are the minimum
//! unit of storage locality and the floor for potentially parallel work.
//!
//! `column`
//!: A runtime wrapper around one chunk pointer paired with one `Meta`.
//!
//! `row`
//!: An ephemeral locator that is valid only for the current scheduled job epoch. A row is not
//! stable identity and must not be persisted across structural change boundaries.
//!
//! `rows`
//!: A generated chunk-aligned view of transient row handles. `Rows<'job>` behaves like a
//! slice-shaped view even though rows are not stored in a physical column.
//!
//! `key`
//!: A stable identity datum. Storage primitives treat `Key` like any other column, and
//! later extension resources such as `Keys` synchronize tables that choose to store it.
//!
//! `job`
//!: A runtime work unit that the executor may schedule independently. In the selected direction,
//! scheduled functions expand into per-chunk jobs.
//!
//! `resolve job`
//!: A later scheduled step that batches deferred command buffers and applies their structural work
//! according to declared happens-before edges.
//!
//! `happens-before`
//!: An ordering guarantee between jobs or resolve jobs. Outside declared happens-before edges, the
//! runtime is intentionally free to execute non-deterministically.

pub mod command;
pub mod instrumentation;
pub mod key;
pub mod query;
pub mod schedule;
pub mod schema;
pub mod store;

pub use self::store::{ChunkPlan, Configuration, Store};
