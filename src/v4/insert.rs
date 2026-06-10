use crate::v4::{
    Error, Meta, Store, Table,
    buffer::Buffer,
    depend::Dependency,
    template::{Column, ColumnWith, Key, Template},
    utility::{IntoNest, Push},
};
use core::marker::PhantomData;

pub struct Build<T>(T);

// TODO: Store items in a buffer that mirrors the table's columns such that
// copying the data can be simply done by 'zipping' the buffer's pointers with
// the columns and 'ptr::copy_nonoverlapping'. `Table::insert` can then remain
// non-generic. Add `Template::push` to move the item to the corresponding
// buffer column.
pub struct Insert<T: Template> {
    template: T,
    state: T::State,
    table: Table,
    buffer: Buffer,
}

impl Insert<()> {
    pub const fn builder() -> Build<()> {
        Build(())
    }
}

impl<T: Template> Insert<T> {
    #[inline]
    pub fn one<N: IntoNest<Nest = T::Item>>(&mut self, item: N) -> Result<(), Error> {
        let item = item.into_nest();
        self.buffer.reserve(1)?;
        unsafe { self.template.set(&self.state, item, &mut self.buffer) };
        unsafe { self.buffer.commit() };
        Ok(())
    }

    #[inline]
    pub fn all<I: IntoIterator<IntoIter: ExactSizeIterator, Item: IntoNest<Nest = T::Item>>>(
        &mut self,
        items: I,
    ) -> Result<(), Error> {
        let items = items.into_iter();
        let count = items.len().try_into().map_err(Error::InsertOverflow)?;
        self.buffer.reserve(count)?;
        for item in items.map(I::Item::into_nest) {
            unsafe { self.template.set(&self.state, item, &mut self.buffer) };
            unsafe { self.buffer.commit() };
        }
        Ok(())
    }

    pub fn resolve(&mut self) -> Result<(), Error> {
        self.table.append(&mut self.buffer)
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
        let table = store
            .tables()
            .find_or_add(template.depend().map(Dependency::meta))?;
        let state = template
            .initialize(&table, store)
            .ok_or(Error::FailedToInitialize)?;
        Ok(Insert {
            template,
            state,
            buffer: Buffer::new(table.metas()),
            table,
        })
    }
}
