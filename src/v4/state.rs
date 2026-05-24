use crate::v4::{Error, Store, module, utility::Push};
use ref_cast::RefCast;

#[derive(RefCast)]
#[repr(transparent)]
pub struct Module<M = ()>(M);

pub struct State<'a, M: module::Module = ()> {
    store: &'a mut Store,
    module: M,
    state: M::State,
}

pub struct Rest<'a, M: module::Module = ()> {
    store: &'a Store,
    module: &'a mut M,
    state: &'a mut M::State,
}

impl State<'_> {
    pub const fn build() -> Module {
        Module::new()
    }
}

impl<M: module::Module> State<'_, M> {
    pub fn get(&mut self) -> Result<M::Item<'_>, Error> {
        self.update()?;
        Ok(self.module.get(&self.state, self.store))
    }

    fn update(&mut self) -> Result<bool, Error> {
        let mut did = false;
        while self.module.update(&mut self.state, self.store)? {
            did = true;
        }
        Ok(did)
    }
}

impl<H: module::Module, T: module::Module> State<'_, (H, T)> {
    pub fn next<'b>(&'b mut self) -> (H::Item<'b>, Rest<'b, T>) {
        let Self {
            store,
            module,
            state,
        } = self;
        (
            module.0.get(&state.0, store),
            Rest {
                store,
                module: &mut module.1,
                state: &mut state.1,
            },
        )
    }
}

impl<H: module::Module, T: module::Module> Rest<'_, (H, T)> {
    pub fn next<'b>(&'b mut self) -> (H::Item<'b>, Rest<'b, T>) {
        let Self {
            store,
            module,
            state,
        } = self;
        (
            module.0.get(&state.0, store),
            Rest {
                store,
                module: &mut module.1,
                state: &mut state.1,
            },
        )
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

    fn get<'a>(&'a self, state: &'a Self::State, store: &'a Store) -> Self::Item<'a>
    where
        Self: 'a,
    {
        self.0.get(state, store)
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

    fn get<'a>(&'a self, state: &'a Self::State, store: &'a Store) -> Self::Item<'a>
    where
        Self: 'a,
    {
        let (head, tail) = self.split_ref();
        (head.0.get(&state.0, store), tail.get(&state.1, store))
    }
}

#[test]
fn boba() -> Result<(), Error> {
    use crate::v4::query::Query;
    let mut s = Store::new();
    let mut b = s.state(
        State::build()
            .push(Query::build().read::<char>().write::<String>())
            .push(Query::build().read::<isize>())
            .push(Query::build().read::<[u32; 100]>())
            .push(Query::build().read::<u8>())
            .push(Query::build().read::<usize>())
            .push(Query::build().read::<char>())
            .push(Query::build().read::<i32>()),
    )?;
    let (i, mut b) = b.next();
    let (i, mut b) = b.next();
    let (i, mut b) = b.next();
    let (i, mut b) = b.next();
    let (i, mut b) = b.next();
    let (i, mut b) = b.next();
    let (i, mut b) = b.next();
    Ok(())
}
