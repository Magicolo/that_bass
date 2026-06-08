#[cfg(debug_assertions)]
use crate::v4::table::Lock;
use crate::v4::{
    Error, Meta, Store,
    depend::{Depend, Dependency},
    filter::{Filter, Has, HasWith, Not},
    item::{self, Context, Item, Read, ReadWith, Try, Write, WriteWith},
    table,
    utility::{IntoFlat, Push, is_unique},
};
use core::{cell::RefCell, marker::PhantomData, slice};

pub struct Guard<'a, I: Item> {
    item: &'a I,
    state: &'a mut I::State,
    table: &'a table::Table,
    count: u32,
    remove: &'a RefCell<Vec<u32>>,
    #[cfg(debug_assertions)]
    locks: Vec<Lock>,
}

pub struct Tables<'a, I: Item> {
    item: &'a I,
    remove: &'a RefCell<Vec<u32>>,
    states: slice::IterMut<'a, (table::Table, I::State)>,
}

pub struct Build<I, F>(I, F);

#[derive(Debug, Clone)]
pub struct Query<I: Item, F: Filter> {
    item: I,
    filter: F,
    count: u32,
    states: Vec<(table::Table, I::State)>,
    remove: RefCell<Vec<u32>>,
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
        let query = Query {
            item: self.0,
            filter: self.1,
            count: 0,
            states: Vec::new(),
            remove: RefCell::new(Vec::new()),
            tables: store.tables.clone(),
        };
        query.analyze()?;
        Ok(query)
    }
}

impl Query<(), ()> {
    pub const fn builder() -> Build<(), ()> {
        Build::new()
    }
}

impl<I: Item, F: Filter> Query<I, F> {
    pub fn tables(&mut self) -> Tables<'_, I> {
        self.update();
        Tables {
            item: &self.item,
            remove: &self.remove,
            states: self.states.iter_mut(),
        }
    }

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

impl<'a, I: Item, F: Filter> IntoIterator for &'a mut Query<I, F> {
    type IntoIter = Tables<'a, I>;
    type Item = <Self::IntoIter as Iterator>::Item;

    fn into_iter(self) -> Self::IntoIter {
        self.tables()
    }
}

impl<'a, I: Item> Iterator for Tables<'a, I> {
    type Item = Guard<'a, I>;

    fn next(&mut self) -> Option<Self::Item> {
        let (table, state) = self.states.next()?;
        let count = unsafe { table.lock(self.item.declare(state)) };
        #[cfg(debug_assertions)]
        let locks = self.item.declare(state).collect::<Vec<_>>();
        #[cfg(debug_assertions)]
        debug_assert!(locks.is_sorted() && is_unique(&locks));
        Some(Guard {
            #[cfg(debug_assertions)]
            locks,
            count: count.unwrap_or_else(|| table.count()),
            item: self.item,
            remove: self.remove,
            state,
            table,
        })
    }
}

impl<I: Item> Guard<'_, I> {
    pub fn table(&self) -> &table::Table {
        &self.table
    }

    pub fn count(&self) -> u32 {
        self.count
    }

    pub fn get<'a>(&'a mut self) -> <I::Item<'a> as IntoFlat>::Flat
    where
        I::Item<'a>: IntoFlat,
    {
        #[cfg(debug_assertions)]
        let context = Context::new(self.count, self.table, self.remove, &self.locks);
        #[cfg(not(debug_assertions))]
        let context = Context::new(self.count, self.table, self.remove);
        unsafe { self.item.get(&mut self.state, context) }.into_flat()
    }
}

impl<'a, I: Item> Drop for Guard<'a, I> {
    fn drop(&mut self) {
        let resolve = unsafe {
            self.table.unlock(
                self.item.declare(self.state),
                &mut *self.remove.borrow_mut(),
            )
        };
        if resolve {
            let _ = self.table.resolve();
        }
    }
}

impl<I, F> Build<I, F> {
    pub fn key(self) -> Build<I::Out, F>
    where
        I: Push<item::Key>,
    {
        self.push_item(item::Key(()))
    }

    pub fn rows(self) -> Build<I::Out, F>
    where
        I: Push<item::Rows>,
    {
        self.push_item(item::Rows(()))
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
