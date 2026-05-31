use crate::v4::{
    Error, Meta, Rows, Store, Table,
    depend::{Access, Depend, Dependency, Resource},
    template::{Column, ColumnWith, Key, Template},
    utility::{IntoNest, Push},
};
use core::{marker::PhantomData, mem::take};

pub struct Build<T>(T);
pub struct Insert<T: Template> {
    template: T,
    count: u32,
    table: Table,
    state: T::State,
}

impl Insert<()> {
    pub const fn builder() -> Build<()> {
        Build(())
    }
}

impl<T: Template> Insert<T> {
    pub fn one<N: IntoNest<Nest = T::Item>>(&mut self, item: N) {
        self.template.defer(&mut self.state, item.into_nest());
        self.count += 1;
    }

    pub fn resolve(&mut self) -> Result<Rows<'_>, Error> {
        match take(&mut self.count) {
            0 => Ok(Rows::new(0..0, &self.table)),
            count => {
                let rows = self.table.insert(count, |rows| unsafe {
                    self.template.resolve(&mut self.state, rows, &self.table)
                })?;
                debug_assert_eq!(count as usize, rows.len());
                Ok(rows)
            }
        }
    }
}

impl<T> Build<T> {
    pub fn key(self) -> Build<T::Out>
    where
        T: Push<Key>,
    {
        Build(self.0.push(Key(())))
    }

    pub fn column<C: 'static>(self) -> Build<T::Out>
    where
        T: Push<Column<C>>,
    {
        Build(self.0.push(Column(PhantomData)))
    }

    pub fn column_with(self, meta: Meta) -> Build<T::Out>
    where
        T: Push<ColumnWith>,
    {
        Build(self.0.push(ColumnWith(meta)))
    }
}

impl<T: Template> Build<T> {
    pub fn build(self, store: &Store) -> Result<Insert<T>, Error> {
        let template = self.0;
        let table = store.tables().find_or_add(template.declare())?;
        let state = template
            .initialize(&table)
            .ok_or(Error::FailedToInitialize)?;
        Ok(Insert {
            template,
            count: 0,
            table,
            state,
        })
    }
}

unsafe impl<T: Template> Depend for Insert<T> {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        self.template.declare().map(|meta| Dependency {
            access: Access::Write,
            resource: Resource::Column2(meta.identifier()),
        })
    }
}
