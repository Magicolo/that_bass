pub mod add;
pub mod create;
pub mod derive;
pub mod destroy;
pub mod event;
pub mod query;
pub mod remove;

use crate::{self as that_bass};
use checkito::{any, CheckParallel, FullGenerate, Generate, Prove};
use std::{collections::HashSet, error, fmt, marker::PhantomData};
use that_bass::{
    filter::{Any, Filter, Has, Is, Not},
    key::Key,
    query::By,
    template::Template,
    Database, Datum, Error, Filter, Template,
};

#[derive(Debug, Clone, Copy, Default, Datum)]
pub struct A;
#[derive(Debug, Clone, Copy, Default, Datum)]
pub struct B;
#[derive(Debug, Clone, Copy, Default, Datum)]
pub struct C(usize);

const COUNT: usize = 37;

#[test]
fn boba() -> Result<(), Box<dyn error::Error>> {
    #[derive(Debug, Clone, Copy)]
    enum Action {
        CreateA(usize, bool),
        CreateB(bool),
        CreateC(bool),
        AddA(bool),
        AddB(bool),
        AddC(bool),
        RemoveA(bool),
        RemoveB(bool),
        RemoveC(bool),
        Destroy(usize, bool),
    }
    struct Proof<P>(&'static str, P);
    impl<P> fmt::Debug for Proof<P> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(&self.0)
        }
    }
    impl<P> fmt::Display for Proof<P> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            fmt::Debug::fmt(&self.0, f)
        }
    }
    impl<P> error::Error for Proof<P> {}
    macro_rules! prove {
        ($prove:expr) => {{
            let prove = $prove;
            if prove.prove() {
                Ok(())
            } else {
                Err(Proof(stringify!($prove), prove))
            }
        }};
    }

    let count = ..100usize;
    let resolve = <bool>::generator();
    let actions = any::Any((
        (count.clone(), resolve.clone()).map(|(count, resolve)| Action::CreateA(count, resolve)),
        <bool>::generator().map(Action::CreateB),
        <bool>::generator().map(Action::CreateC),
        <bool>::generator().map(Action::AddA),
        <bool>::generator().map(Action::AddB),
        <bool>::generator().map(Action::AddC),
        <bool>::generator().map(Action::RemoveA),
        <bool>::generator().map(Action::RemoveB),
        <bool>::generator().map(Action::RemoveC),
        (count.clone(), resolve.clone()).map(|(count, resolve)| Action::Destroy(count, resolve)),
    ))
    .collect_with::<_, Vec<Action>>(..256usize);
    actions.check_parallel_with(u16::MAX as _, u8::MAX as _, None, |actions| {
        let database = Database::new();
        let mut create_a = None;
        let mut destroy = None;
        let mut query_keys = database.query::<Key>()?;
        for &action in actions {
            match action {
                Action::CreateA(count, resolve) => {
                    let create = match &mut create_a {
                        Some(create) => create,
                        None => create_a.insert(database.create::<A>()?),
                    };
                    let keys: Vec<_> = create.all(vec![A; count]).iter().copied().collect();
                    prove!(keys.len() == count)?;
                    if resolve {
                        let resolved = create.resolve();
                        prove!(keys.len() <= resolved)?;
                        for key in keys {
                            prove!(database.keys().get(key).is_ok())?;
                        }
                    } else {
                        for key in keys {
                            prove!(database.keys().get(key).is_err())?;
                        }
                    }
                }
                Action::Destroy(count, resolve) => {
                    let destroy = match &mut destroy {
                        Some(destroy) => destroy,
                        None => destroy.insert(database.destroy()),
                    };
                    let mut keys = Vec::new();
                    query_keys.try_each(|key| {
                        if keys.len() < count {
                            keys.push(key);
                            true
                        } else {
                            false
                        }
                    });
                    destroy.all(keys.iter().copied());
                    if resolve {
                        let resolved = destroy.resolve();
                        prove!(keys.len() <= count)?;
                        prove!(keys.len() <= resolved)?;
                        for key in keys {
                            prove!(database.keys().get(key).is_err())?;
                        }
                    } else {
                        for key in keys {
                            prove!(database.keys().get(key).is_ok())?;
                        }
                    }
                }
                _ => {}
            }
        }
        Ok::<(), Box<dyn error::Error + Send + Sync>>(())
    })?;
    Ok(())
}

fn create_one(database: &Database, template: impl Template) -> Result<Key, Error> {
    let mut create = database.create()?;
    let key = create.one(template);
    assert_eq!(create.resolve(), 1);
    Ok(key)
}

fn create_n<const N: usize>(
    database: &Database,
    templates: [impl Template; N],
) -> Result<[Key; N], Error> {
    let mut create = database.create()?;
    let keys = create.all_n(templates);
    assert_eq!(create.resolve(), N);
    Ok(keys)
}

fn destroy_one(database: &Database, key: Key) -> Result<(), Error> {
    let mut destroy = database.destroy();
    destroy.one(key);
    assert_eq!(destroy.resolve(), 1);
    Ok(())
}

fn destroy_all(database: &Database, keys: &[Key]) {
    let mut destroy = database.destroy();
    destroy.all(keys.iter().copied());
    assert_eq!(destroy.resolve(), keys.len());
}
