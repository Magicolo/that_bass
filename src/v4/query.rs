use crate::v4::{
    Error, Meta, Store,
    depend::{Depend, Dependency},
    filter::{Filter, Has, HasWith, Not},
    item::{self, Item, Read, ReadWith, Try, Write, WriteWith},
    table,
    utility::{IntoFlat, Push},
};
use core::{marker::PhantomData, slice};

pub struct Columns<'a, I: Item> {
    query: &'a I,
    states: slice::IterMut<'a, (table::Table, I::State)>,
}

pub struct Build<I, F>(I, F);
pub struct Query<I: Item, F: Filter> {
    item: I,
    filter: F,
    count: u32,
    states: Vec<(table::Table, I::State)>,
    tables: table::Tables,
}

impl Build<(), ()> {
    pub const fn new() -> Self {
        Self((), ())
    }
}

impl<I, F> Build<I, F> {
    fn push_item<J>(self, item: J) -> Build<I::Out, F>
    where
        I: Push<J>,
    {
        Build(self.0.push(item), self.1)
    }

    fn push_filter<G>(self, filter: G) -> Build<I, F::Out>
    where
        F: Push<G>,
    {
        Build(self.0, self.1.push(filter))
    }
}

impl<I: Item, F: Filter> Build<I, F> {
    pub fn build(self, store: &Store) -> Result<Query<I, F>, Error> {
        self.0.analyze()?;
        Ok(Query {
            item: self.0,
            filter: self.1,
            count: 0,
            states: Vec::new(),
            tables: store.tables.clone(),
        })
    }
}

impl Query<(), ()> {
    pub const fn builder() -> Build<(), ()> {
        Build::new()
    }
}

impl<I: Item, F: Filter> Query<I, F> {
    fn update(&mut self) {
        while let Some(table) = self.tables.get(self.count) {
            self.count += 1;
            if self.filter.filter(&table)
                && let Some(query) = self.item.initialize(&table)
            {
                self.states.push((table, query));
            }
        }
    }
}

unsafe impl<I: Item, F: Filter> Depend for Query<I, F> {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        self.item.depend()
    }
}

impl<'a, I: Item<Guard<'a>: IntoFlat>, F: Filter> IntoIterator for &'a mut Query<I, F> {
    type IntoIter = Columns<'a, I>;
    type Item = <I::Guard<'a> as IntoFlat>::Flat;

    fn into_iter(self) -> Self::IntoIter {
        self.update();
        Columns {
            query: &self.item,
            states: self.states.iter_mut(),
        }
    }
}

impl<'a, I: Item<Guard<'a>: IntoFlat>> Iterator for Columns<'a, I> {
    type Item = <I::Guard<'a> as IntoFlat>::Flat;

    fn next(&mut self) -> Option<Self::Item> {
        let (table, state) = self.states.next()?;
        Some(self.query.get(state, table.count(), table).into_flat())
    }
}

impl<I, F> Build<I, F> {
    pub fn key(self) -> Build<I::Out, F>
    where
        I: Push<item::Key>,
    {
        self.push_item(item::Key(()))
    }

    pub fn row(self) -> Build<I::Out, F>
    where
        I: Push<item::Row>,
    {
        self.push_item(item::Row(()))
    }

    pub fn table(self) -> Build<I::Out, F>
    where
        I: Push<item::Table>,
    {
        self.push_item(item::Table(()))
    }

    pub fn try_read<T: 'static>(self) -> Build<I::Out, F>
    where
        I: Push<Try<Read<T>>>,
    {
        self.push_item(Try(Read(PhantomData)))
    }

    pub fn read<T: 'static>(self) -> Build<I::Out, F>
    where
        I: Push<Read<T>>,
    {
        self.push_item(Read(PhantomData))
    }

    pub fn try_read_with(self, meta: Meta) -> Build<I::Out, F>
    where
        I: Push<Try<ReadWith>>,
    {
        self.push_item(Try(ReadWith(meta)))
    }

    pub fn read_with(self, meta: Meta) -> Build<I::Out, F>
    where
        I: Push<ReadWith>,
    {
        self.push_item(ReadWith(meta))
    }

    pub fn try_write<T: 'static>(self) -> Build<I::Out, F>
    where
        I: Push<Try<Write<T>>>,
    {
        self.push_item(Try(Write(PhantomData)))
    }

    pub fn try_write_with(self, meta: Meta) -> Build<I::Out, F>
    where
        I: Push<Try<WriteWith>>,
    {
        self.push_item(Try(WriteWith(meta)))
    }

    pub fn write<T: 'static>(self) -> Build<I::Out, F>
    where
        I: Push<Write<T>>,
    {
        self.push_item(Write(PhantomData))
    }

    pub fn write_with(self, meta: Meta) -> Build<I::Out, F>
    where
        I: Push<WriteWith>,
    {
        self.push_item(WriteWith(meta))
    }

    pub fn has<T: 'static>(self) -> Build<I, F::Out>
    where
        F: Push<Has<T>>,
    {
        self.push_filter(Has(PhantomData))
    }

    pub fn has_with(self, meta: Meta) -> Build<I, F::Out>
    where
        F: Push<HasWith>,
    {
        self.push_filter(HasWith(meta))
    }

    pub fn not<T: 'static>(self) -> Build<I, F::Out>
    where
        F: Push<Not<Has<T>>>,
    {
        self.push_filter(Not(Has(PhantomData)))
    }

    pub fn not_with(self, meta: Meta) -> Build<I, F::Out>
    where
        F: Push<Not<HasWith>>,
    {
        self.push_filter(Not(HasWith(meta)))
    }
}
