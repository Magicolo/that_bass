use crate::v4::{
    Error, Row, Store, Table, query, template,
    utility::{ranges, sort},
};
use core::{iter, slice::Iter};

pub struct Query<'a, Q: query::Query> {
    query: Q,
    states: Box<[(u32, Q::State)]>,
    tables: &'a mut [Table],
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
    pub fn query<Q: query::Query>(&mut self, query: Q) -> Query<'_, Q> {
        // TODO: Query constuction must be failible if the same `Meta` is mentioned
        // twice. Perhaps add `query::Query::declare` like in `template::Template`.
        Query {
            states: self
                .tables
                .iter()
                .filter_map(|table| Some((table.index(), query.initialize(table)?)))
                .collect(),
            tables: &mut self.tables,
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
                self.tables.push(Table::new(index, metas)?);
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

pub struct Iterator<'a, Q: query::Query> {
    query: &'a Q,
    states: Iter<'a, (u32, Q::State)>,
    tables: &'a mut [Table],
}

impl<'a, Q: query::Query> iter::Iterator for Iterator<'a, Q> {
    type Item = Q::Item<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        None
        // TODO: The `&mut` requirement and `Iterator` signature makes this
        // iterator impossible...
        // let (table, state) = self.states.next()?;
        // let table = unsafe { self.tables.get_unchecked_mut(*table as usize)
        // }; Some(self.query.get(state, table))
    }
}

impl<'a, Q: query::Query> Query<'a, Q> {
    pub fn iter(&mut self) -> Iterator<'_, Q> {
        Iterator {
            query: &self.query,
            states: self.states.iter(),
            tables: &mut self.tables,
        }
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
