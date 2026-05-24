use crate::v4::{Error, Store};

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
}
