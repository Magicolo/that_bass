use crate::v4::{
    Error, Meta,
    depend::{Access, Depend, Dependency},
    filter::{Filter, Has, HasWith, Not},
    item::{self, Context, Item, Read, ReadWith, Try, Write, WriteWith},
    key::{self, Keys},
    table::{self, Lock},
    utility::{IntoFlat, Push},
};
use core::{any::TypeId, cell::RefCell, marker::PhantomData};

pub struct All<'a, I: Item> {
    count: u32,
    state: &'a (table::Table, I::State),
    query: &'a Inner<I>,
}

pub struct One<'a, I: Item> {
    state: &'a (table::Table, I::State),
    query: &'a Inner<I>,
}

pub struct Tables<'a, I: Item> {
    index: u32,
    query: &'a Inner<I>,
}

pub struct Build<I, F>(I, F);

pub struct Query<I: Item, F: Filter> {
    inner: Inner<I>,
    filter: F,
}

struct Inner<I: Item> {
    item: I,
    count: u32,
    states: Vec<(table::Table, I::State)>,
    remove: RefCell<Vec<u32>>,
    tables: table::Tables,
    keys: Keys,
}

impl<I: Item> Drop for One<'_, I> {
    fn drop(&mut self) {
        let _ = unsafe {
            self.state.0.unlock(
                self.query.item.declare(&self.state.1),
                &mut *self.query.remove.borrow_mut(),
            )
        };
    }
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
    pub fn build(self, store: &crate::v4::Store) -> Result<Query<I, F>, Error> {
        let query = Query {
            filter: self.1,
            inner: Inner {
                item: self.0,
                count: 0,
                states: Vec::new(),
                remove: RefCell::new(Vec::new()),
                tables: store.tables().clone(),
                keys: store.keys().clone(),
            },
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
            index: 0,
            query: &self.inner,
        }
    }

    fn update(&mut self) {
        while let Some(table) = self.inner.tables.get(self.inner.count) {
            self.inner.count += 1;
            if self.filter.filter(&table)
                && let Some(query) = self.inner.item.initialize(&table)
            {
                self.inner.states.push((table, query));
            }
        }
    }

    pub fn get(&mut self, key: key::Key) -> Option<One<'_, I>> {
        let (table_index, row) = self.inner.keys.lookup(key)?;
        let index = self
            .inner
            .states
            .iter()
            .position(|(t, _)| t.index() == table_index)?;
        let entry = &mut self.inner.states[index];
        let table = &entry.0;
        let state = &mut entry.1;
        let count =
            unsafe { table.lock(self.inner.item.declare(state)) }.unwrap_or_else(|| table.count());
        if row >= count {
            let _ = unsafe { table.unlock(self.inner.item.declare(state), &mut Vec::new()) };
            return None;
        }
        let key_column = table.column(TypeId::of::<key::Key>())?;
        let stored_key = unsafe {
            &*table
                .columns()
                .get_unchecked(key_column as usize)
                .get_all::<key::Key>(row + 1)
                .as_ptr()
                .add(row as usize)
        };
        if *stored_key != key {
            let _ = unsafe { table.unlock(self.inner.item.declare(state), &mut Vec::new()) };
            return None;
        }
        Some(One {
            state: index,
            query: &self.inner,
        })
    }
}

unsafe impl<I: Item, F: Filter> Depend for Query<I, F> {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        self.inner.depend()
    }
}

impl<'a, I: Item, F: Filter> IntoIterator for &'a mut Query<I, F> {
    type IntoIter = Tables<'a, I>;
    type Item = <Self::IntoIter as Iterator>::Item;

    fn into_iter(self) -> Self::IntoIter {
        self.tables()
    }
}

impl<I: Item<State: Clone> + Clone, F: Filter + Clone> Clone for Query<I, F> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            filter: self.filter.clone(),
        }
    }
}

unsafe impl<I: Item> Depend for Inner<I> {
    fn depend(&self) -> impl Iterator<Item = Dependency> {
        self.item.depend()
    }
}

impl<I: Item<State: Clone> + Clone> Clone for Inner<I> {
    fn clone(&self) -> Self {
        Self {
            item: self.item.clone(),
            count: self.count,
            states: self.states.clone(),
            remove: self.remove.clone(),
            tables: self.tables.clone(),
            keys: self.keys.clone(),
        }
    }
}

impl<'a, I: Item> Iterator for Tables<'a, I> {
    type Item = All<'a, I>;

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.index;
        let pair = self.query.states.get(index as usize)?;
        self.index += 1;
        let count = unsafe { pair.0.lock(self.query.item.declare(&pair.1)) }
            .unwrap_or_else(|| pair.0.count());
        Some(All {
            count,
            query: self.query,
            state: pair,
        })
    }
}

impl<I: Item> All<'_, I> {
    pub fn table(&self) -> &table::Table {
        &self.state.0
    }

    pub fn count(&self) -> u32 {
        self.count
    }

    pub fn all<'a>(&'a mut self) -> <I::All<'a> as IntoFlat>::Flat
    where
        I::All<'a>: IntoFlat,
    {
        let context = Context::new(self.count, &self.state.0, &self.query.remove);
        unsafe { self.query.item.all(&self.state.1, context) }.into_flat()
    }

    pub fn one<'a>(&'a mut self, row: u32) -> Option<<I::One<'a> as IntoFlat>::Flat>
    where
        I::One<'a>: IntoFlat,
    {
        if row < self.count {
            let context = Context::new(row, &self.state.0, &self.query.remove);
            Some(unsafe { self.query.item.one(&self.state.1, context) }.into_flat())
        } else {
            None
        }
    }
}

impl<'a, I: Item> Drop for All<'a, I> {
    fn drop(&mut self) {
        let freed = self.snapshot_freed_keys();
        let _ = unsafe {
            self.state.0.unlock(
                self.query.item.declare(&self.state.1),
                &mut *self.query.remove.borrow_mut(),
            )
        };
        for key in freed {
            self.query.keys.free(key);
        }
    }
}

impl<I: Item> All<'_, I> {
    fn snapshot_freed_keys(&self) -> Vec<key::Key> {
        let has_key_column = self
            .state
            .0
            .columns()
            .first()
            .is_some_and(|c| c.meta().is_key());
        if !has_key_column {
            return Vec::new();
        }
        let has_key_access = self
            .query
            .item
            .declare(&self.state.1)
            .any(|lock| lock == Lock::Column(0, Access::Read));
        if !has_key_access {
            return Vec::new();
        }
        self.query
            .remove
            .borrow()
            .iter()
            .map(|&row| unsafe {
                &*self
                    .state
                    .0
                    .columns()
                    .get_unchecked(0)
                    .get_all::<key::Key>(row + 1)
                    .as_ptr()
                    .add(row as usize)
            })
            .copied()
            .collect()
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
