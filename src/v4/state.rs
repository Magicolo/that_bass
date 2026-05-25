use crate::v4::{Error, Store, module, remove::Remove, utility::Push};
use ref_cast::RefCast;

#[derive(RefCast)]
#[repr(transparent)]
pub struct Module<M = ()>(M);

pub struct State<'a, M: module::Module = ()> {
    store: &'a mut Store,
    module: M,
    state: M::State,
}

pub struct Guard<'a, M: module::Module = ()> {
    module: &'a M,
    state: &'a mut M::State,
    store: &'a mut Store,
}

impl State<'_> {
    pub const fn build() -> Module {
        Module::new()
    }
}

impl<M: module::Module> State<'_, M> {
    pub fn guard(&mut self) -> Guard<'_, M> {
        Guard {
            module: &self.module,
            state: &mut self.state,
            store: &mut *self.store,
        }
    }
}

impl Store {
    pub fn state<M: module::Module>(&mut self, module: Module<M>) -> Result<State<'_, M>, Error> {
        Ok(State {
            state: module.0.initialize(self)?,
            store: self,
            module: module.0,
        })
    }
}

impl<'a, H: module::Module, T: module::Module> Guard<'a, (H, T)> {
    pub fn get(&mut self) -> Result<H::Item<'_>, Error> {
        self.update()?;
        Ok(self.module.0.get(&mut self.state.0, self.store))
    }

    pub fn next(self) -> Result<Guard<'a, T>, Error> {
        let Self {
            module,
            state,
            store,
        } = self;
        module.0.resolve(&mut state.0, store)?;
        Ok(Guard {
            module: &module.1,
            state: &mut state.1,
            store,
        })
    }

    pub fn with<F: FnOnce(H::Item<'_>)>(mut self, with: F) -> Result<Guard<'a, T>, Error> {
        with(self.get()?);
        self.next()
    }

    fn update(&mut self) -> Result<bool, Error> {
        let mut did = false;
        while self.module.0.update(&mut self.state.0, self.store)? {
            did = true;
        }
        Ok(did)
    }
}

impl Module {
    pub(crate) const fn new() -> Self {
        Self(())
    }
}

impl<M: module::Module> Module<M> {
    pub fn push<N: module::Module>(self, module: N) -> Module<M::Out>
    where
        M: Push<N, Out: module::Module>,
    {
        Module(self.0.push(module))
    }
}

impl<H, T> Module<(H, T)> {
    fn split_own(self) -> (Module<H>, Module<T>) {
        let Self((head, tail)) = self;
        (Module(head), Module(tail))
    }

    fn split_ref(&self) -> (&Module<H>, &Module<T>) {
        let Self((head, tail)) = self;
        (Module::ref_cast(head), Module::ref_cast(tail))
    }

    #[allow(dead_code)]
    fn split_mut(&mut self) -> (&mut Module<H>, &mut Module<T>) {
        let Self((head, tail)) = self;
        (Module::ref_cast_mut(head), Module::ref_cast_mut(tail))
    }
}

impl module::Module for Module<()> {
    type Item<'a>
        = <() as module::Module>::Item<'a>
    where
        Self: 'a;
    type State = <() as module::Module>::State;

    fn initialize(&self, store: &mut Store) -> Result<Self::State, Error> {
        self.0.initialize(store)
    }

    fn update(&self, state: &mut Self::State, store: &Store) -> Result<bool, Error> {
        self.0.update(state, store)
    }

    fn get<'a>(&'a self, state: &'a mut Self::State, store: &'a Store) -> Self::Item<'a>
    where
        Self: 'a,
    {
        self.0.get(state, store)
    }

    fn resolve(&self, state: &mut Self::State, store: &mut Store) -> Result<(), Error> {
        self.0.resolve(state, store)
    }
}

impl<H: module::Module, T: module::Module> module::Module for Module<(H, T)>
where
    Module<T>: module::Module,
{
    type Item<'a>
        = (H::Item<'a>, <Module<T> as module::Module>::Item<'a>)
    where
        Self: 'a;
    type State = (H::State, <Module<T> as module::Module>::State);

    fn initialize(&self, store: &mut Store) -> Result<Self::State, Error> {
        let (head, tail) = self.split_ref();
        Ok((head.0.initialize(store)?, tail.initialize(store)?))
    }

    fn update(&self, state: &mut Self::State, store: &Store) -> Result<bool, Error> {
        let (head, tail) = self.split_ref();
        Ok(head.0.update(&mut state.0, store)? | tail.update(&mut state.1, store)?)
    }

    fn get<'a>(&'a self, state: &'a mut Self::State, store: &'a Store) -> Self::Item<'a>
    where
        Self: 'a,
    {
        let (head, tail) = self.split_ref();
        (
            head.0.get(&mut state.0, store),
            tail.get(&mut state.1, store),
        )
    }

    fn resolve(&self, state: &mut Self::State, store: &mut Store) -> Result<(), Error> {
        let (head, tail) = self.split_ref();
        head.0.resolve(&mut state.0, store)?;
        tail.resolve(&mut state.1, store)
    }
}

#[test]
fn access() -> Result<(), Error> {
    use crate::v4::query::Query;
    let mut s = Store::new();
    let mut state = s.state(
        State::build()
            .push(Query::build().read::<char>().write::<String>())
            .push((Query::build().read::<isize>(), Remove::build()))
            .push(Query::build().read::<[u32; 100]>())
            .push(Query::build().read::<u8>())
            .push(Query::build().read::<usize>())
            .push(Query::build().read::<char>())
            .push(Query::build().read::<i32>()),
    )?;
    let guard = state.guard();
    let guard = guard.next()?;
    let guard = guard.next()?;
    let guard = guard.next()?;
    let guard = guard.next()?;
    let guard = guard.next()?;
    let guard = guard.next()?;
    let _guard = guard.next()?;
    Ok(())
}
