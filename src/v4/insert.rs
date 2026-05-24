use crate::v4::{Error, Row, Store, Table, template};

pub struct Insert<'a, T: template::Template> {
    table: u32,
    tables: &'a mut [Table],
    state: T::State,
    template: T,
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
        unsafe { self.template.resolve(&mut self.state, &*table) };
        table.commit();
        Ok(())
    }
}

impl Store {
    pub fn insert<T: template::Template>(&mut self, template: T) -> Result<Insert<'_, T>, Error> {
        let table = self.find_or_insert_table(template.declare())?;
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
}
