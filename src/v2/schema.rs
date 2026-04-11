//! Minimal schema layout metrics for the rewrite lane.
//!
//! Task `00-foundation.md` does not implement full schema construction. It only establishes the
//! layout metrics needed to drive chunk-capacity planning and benchmark matrices.

/// The layout metrics that matter for chunk-capacity planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchemaLayout {
    inline_row_width: usize,
    physical_column_count: usize,
}

impl SchemaLayout {
    pub const fn new(inline_row_width: usize, physical_column_count: usize) -> Self {
        Self {
            inline_row_width,
            physical_column_count,
        }
    }

    pub const fn inline_row_width(self) -> usize {
        self.inline_row_width
    }

    pub const fn physical_column_count(self) -> usize {
        self.physical_column_count
    }
}
