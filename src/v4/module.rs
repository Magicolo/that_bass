use crate::v4::{Error, Row, Store, Table, template, utility::ranges};

pub trait Module {
    type Item<'a>
    where
        Self: 'a;
    type State;

    fn initialize(&self, store: &mut Store) -> Result<Self::State, Error>;
    fn update(&self, state: &mut Self::State, store: &Store) -> Result<bool, Error>;
    fn get<'a>(&'a self, state: &'a Self::State, store: &'a Store) -> Self::Item<'a>
    where
        Self: 'a;
}

impl<M: Module> Module for &mut M {
    type Item<'a>
        = M::Item<'a>
    where
        Self: 'a;
    type State = M::State;

    fn initialize(&self, store: &mut Store) -> Result<Self::State, Error> {
        M::initialize(self, store)
    }

    fn update(&self, state: &mut Self::State, store: &Store) -> Result<bool, Error> {
        M::update(self, state, store)
    }

    fn get<'a>(&'a self, state: &'a Self::State, store: &'a Store) -> Self::Item<'a>
    where
        Self: 'a,
    {
        M::get(self, state, store)
    }
}

impl<M: Module> Module for &M {
    type Item<'a>
        = M::Item<'a>
    where
        Self: 'a;
    type State = M::State;

    fn initialize(&self, store: &mut Store) -> Result<Self::State, Error> {
        M::initialize(self, store)
    }

    fn update(&self, state: &mut Self::State, store: &Store) -> Result<bool, Error> {
        M::update(self, state, store)
    }

    fn get<'a>(&'a self, state: &'a Self::State, store: &'a Store) -> Self::Item<'a>
    where
        Self: 'a,
    {
        M::get(self, state, store)
    }
}

impl Module for () {
    type Item<'a>
        = ()
    where
        Self: 'a;
    type State = ();

    fn initialize(&self, _: &mut Store) -> Result<Self::State, Error> {
        Ok(())
    }

    fn update(&self, _: &mut Self::State, _: &Store) -> Result<bool, Error> {
        Ok(false)
    }

    fn get<'a>(&'a self, _: &'a Self::State, _: &'a Store) -> Self::Item<'a>
    where
        Self: 'a,
    {
        ()
    }
}

impl<M0: Module, M1: Module> Module for (M0, M1) {
    type Item<'a>
        = (M0::Item<'a>, M1::Item<'a>)
    where
        Self: 'a;
    type State = (M0::State, M1::State);

    fn initialize(&self, store: &mut Store) -> Result<Self::State, Error> {
        Ok((self.0.initialize(store)?, self.1.initialize(store)?))
    }

    fn update(&self, state: &mut Self::State, store: &Store) -> Result<bool, Error> {
        Ok(self.0.update(&mut state.0, store)? | self.1.update(&mut state.1, store)?)
    }

    fn get<'a>(&'a self, state: &'a Self::State, store: &'a Store) -> Self::Item<'a>
    where
        Self: 'a,
    {
        (self.0.get(&state.0, store), self.1.get(&state.1, store))
    }
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
    // TODO: This method is not safe because of the implementation `Module for (M0,
    // M1)` which currently validates `M0` and `M1` individually rather than as a
    // single unit. This means that one can get two `Query<Write<T>>` items and
    // alias a reference to the same location, thus violating rust's invariants.
    pub fn with<M: Module, T, F: FnOnce(M::Item<'_>) -> T>(
        &mut self,
        module: M,
        with: F,
    ) -> Result<T, Error> {
        let state = module.initialize(self)?;
        Ok(with(module.get(&state, self)))
    }

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

    pub fn remove(&mut self) -> Remove<'_> {
        Remove {
            rows: Vec::new(),
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
