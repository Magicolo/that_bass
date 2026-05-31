use crate::v4::{
    Error, Meta, Store, Table, Vector,
    depend::{Access, Depend, Dependency, Resource},
    template::{Column, ColumnWith, Key, Template},
    utility::{IntoNest, Push},
};
use core::{
    any::{Any, TypeId},
    iter::{empty, once},
    marker::PhantomData,
    mem::take,
    ptr::NonNull,
};

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

    pub fn resolve(&mut self) -> Result<(), Error> {
        let count = take(&mut self.count);
        if count > 0 {
            let rows = self.table.reserve(count)?;
            unsafe { self.template.resolve(&mut self.state, rows, &self.table) };
            self.table.commit();
        }
        Ok(())
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
