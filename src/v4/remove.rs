use crate::v4::{Row, Store, Table, utility::ranges};

pub struct Remove<'a> {
    rows: Vec<(u32, u32)>,
    tables: &'a mut [Table],
}

impl<'a> Remove<'a> {
    pub fn one(&mut self, row: Row) {
        self.rows.push((row.table(), row.row()));
    }

    pub fn resolve(&mut self) -> u32 {
        self.rows.sort();
        let mut total = 0u32;
        for (table, rows) in ranges(self.rows.drain(..).rev()) {
            if let Some(table) = self.tables.get_mut(table as usize) {
                total = total.saturating_add(rows.end.saturating_sub(rows.start));
                table.release(rows);
            }
        }
        total
    }
}

impl Store {
    pub fn remove(&mut self) -> Remove<'_> {
        Remove {
            rows: Vec::new(),
            tables: &mut self.tables,
        }
    }
}
