use crate::v4::utility::ranges;
use crate::v4::utility::sort;
use crate::v4::{At, Error, Row, Store, Table, query, template};

pub struct Query<'a, Q: query::Query> {
    query: Q,
    states: Box<[(u32, Q::State<'a>)]>,
    tables: &'a [Table],
}

pub struct Insert<'a, T: template::Template> {
    table: u32,
    tables: &'a mut [Table],
    state: T::State,
    template: T,
}

pub struct Remove<'a> {
    rows: Vec<(u32, u32)>,
    tables: &'a mut [Table],
}

impl Store {
    pub fn query<Q: query::Query>(&self, query: Q) -> Query<'_, Q> {
        Query {
            states: self
                .tables
                .iter()
                .enumerate()
                .filter_map(|(index, table)| {
                    let table = At(u32::try_from(index).ok()?, table);
                    Some((table.index(), query.initialize(table)?))
                })
                .collect(),
            tables: &self.tables,
            query,
        }
    }

    pub fn insert<T: template::Template>(&mut self, template: T) -> Result<Insert<'_, T>, Error> {
        let metas = sort(template.declare())?;
        let table = match self.find_table(&metas) {
            Some(index) => index,
            None => {
                let index = self
                    .tables
                    .len()
                    .try_into()
                    .map_err(Error::TablesOverflow)?;
                self.tables.push(Table::new(metas)?);
                index
            }
        };
        let state = template
            .initialize(unsafe { self.tables.get_unchecked_mut(table as usize) })
            .ok_or(Error::FailedToInitialize)?;
        Ok(Insert {
            state,
            table,
            tables: &mut self.tables,
            template,
        })
    }

    pub fn remove(&mut self) -> Remove<'_> {
        Remove {
            rows: Vec::new(),
            tables: &mut self.tables,
        }
    }
}

impl<'a, Q: query::Query> Query<'a, Q> {
    pub fn iter(&self) -> impl Iterator<Item = Q::Item<'a>> + '_ {
        self.states.iter().map(|(_, state)| self.query.get(state))
    }
}

impl<'a, T: template::Template> Insert<'a, T> {
    pub fn one(&mut self, item: T::Item) -> Result<Row<'a>, Error> {
        let table = unsafe { self.tables.get_unchecked_mut(self.table as usize) };
        let row = table.reserve(1)?.next().ok_or(Error::FailedToReserve)?;
        if self.template.defer(&mut self.state, item) {
            Ok(Row::new(row, self.table))
        } else {
            todo!()
        }
    }

    pub fn resolve(mut self) -> Result<(), Error> {
        let table = unsafe { self.tables.get_unchecked_mut(self.table as usize) };
        table.ensure()?;
        unsafe { self.template.resolve(&mut self.state, table) };
        table.commit();
        Ok(())
    }
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
