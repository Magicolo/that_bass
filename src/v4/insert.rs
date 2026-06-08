use crate::v4::{
    Error, Meta, Store, Table,
    depend::{Access, Depend, Dependency, Resource},
    template::{Column, ColumnWith, Key, Template},
    utility::{IntoNest, Push},
};
use core::marker::PhantomData;

pub struct Build<T>(T);

pub struct Insert<T: Template> {
    template: T,
    state: T::State,
    items: Vec<T::Item>,
    table: Table,
}

impl Insert<()> {
    pub const fn builder() -> Build<()> {
        Build(())
    }
}

impl<T: Template> Insert<T> {
    pub fn one<N: IntoNest<Nest = T::Item>>(&mut self, item: N) {
        self.items.push(item.into_nest());
    }

    pub fn all<I: IntoIterator<Item: IntoNest<Nest = T::Item>>>(&mut self, items: I) {
        self.items.extend(items.into_iter().map(I::Item::into_nest));
    }

    pub fn resolve(&mut self) -> Result<(), Error> {
        let count = self.items.len().try_into().map_err(Error::ItemsOverflow)?;
        self.table.insert(count, |start| {
            for (index, item) in self.items.drain(..).enumerate() {
                let index = start + index as u32;
                unsafe {
                    self.template
                        .apply(&mut self.state, item, index, &self.table);
                }
            }
        })?;
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
        let insert = Insert {
            template,
            items: Vec::new(),
            table,
            state,
        };
        insert.analyze()?;
        Ok(insert)
    }
}

unsafe impl<T: Template> Depend for Insert<T> {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        self.template.declare().map(|meta| Dependency {
            access: Access::Write,
            resource: Resource::Column(meta.identifier()),
        })
    }
}
