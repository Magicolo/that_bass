pub mod common;
use common::*;

#[derive(Debug, Clone, Copy)]
pub enum Type {
    Unit,
    A,
    B,
    C,
    AB,
    AC,
    BC,
    ABC,
}

#[derive(Debug, Clone)]
pub enum Action {
    Create(usize, Type, bool),
    Add(usize, Type, bool),
    Remove(usize, Type, bool),
    Destroy(usize, Type, bool),
}

#[test]
#[cfg(not(miri))]
fn check() -> Result<(), Box<dyn error::Error>> {
    let count = ..256usize;
    let r#type = (
        Same(Type::Unit),
        Same(Type::A),
        Same(Type::B),
        Same(Type::C),
        Same(Type::AB),
        Same(Type::AC),
        Same(Type::BC),
        Same(Type::ABC),
    )
        .any()
        .map(Unify::unify);
    let resolve = <bool>::generator();
    (
        (&count, &r#type, &resolve).map(|values| Action::Create(values.0, values.1, values.2)),
        (&count, &r#type, &resolve).map(|values| Action::Add(values.0, values.1, values.2)),
        (&count, &r#type, &resolve).map(|values| Action::Remove(values.0, values.1, values.2)),
        (&count, &r#type, &resolve).map(|values| Action::Destroy(values.0, values.1, values.2)),
    )
        .any()
        .map(Unify::unify)
        .collect_with::<_, Vec<Action>>(..256usize)
        .check(1000, |actions| run(actions.into_iter().cloned()))?;
    Ok(())
}

#[test]
fn removed_column_is_no_longer_queried() -> Result<(), Box<dyn error::Error + Send + Sync>> {
    let database = Database::new();
    let mut create = database.create::<(A, B)>()?;
    let mut remove = database.remove::<B>()?;
    let mut query = database.query::<()>()?.filter::<Has<B>>();
    let key = create.one((A, B(0)));
    assert!(query.find(key, |_| ()).is_err());
    create.resolve();
    assert!(query.find(key, |_| ()).is_ok());
    remove.one(key);
    assert!(query.find(key, |_| ()).is_ok());
    remove.resolve();
    assert!(query.find(key, |_| ()).is_err());
    Ok(())
}

#[test]
fn does_not_resolve_when_no_apply_and_no_move() -> Result<(), Box<dyn error::Error + Send + Sync>> {
    let database = Database::new();
    let mut create = database.create::<(A, B, C)>()?;
    let mut add = database.add::<()>()?;
    let keys = create.defaults_n::<68>();
    create.resolve();
    add.all(keys);
    let resolved = add.resolve();
    assert_eq!(resolved, 0);
    Ok(())
}

#[test]
fn run_empty() -> Result<(), Box<dyn error::Error + Send + Sync>> {
    run([])
}

fn run(
    actions: impl IntoIterator<Item = Action>,
) -> Result<(), Box<dyn error::Error + Send + Sync>> {
    let database = Database::new();
    let mut queries = (
        database.query::<()>()?,
        database.query::<()>()?.filter::<Has<A>>(),
        database.query::<()>()?.filter_with(has::<B>()),
        database.query::<&C>()?,
        database.query::<&A>()?.filter::<Has<B>>(),
        database.query::<&A>()?.filter_with(has::<C>()),
        database
            .query::<()>()?
            .filter_with((has::<B>(), has::<C>())),
        database.query::<(&A, &B, &C)>()?,
        database.query::<()>()?.filter_with(false),
    );

    let mut creates = (
        database.create::<()>()?,
        database.create::<A>()?,
        database.create::<B>()?,
        database.create::<C>()?,
        database.create::<(A, B)>()?,
        database.create::<(A, C)>()?,
        database.create::<(B, C)>()?,
        database.create::<(A, B, C)>()?,
    );

    let mut adds = (
        database.add::<()>()?,
        database.add::<A>()?,
        database.add::<B>()?,
        database.add::<C>()?,
        database.add::<(A, B)>()?,
        database.add::<(A, C)>()?,
        database.add::<(B, C)>()?,
        database.add::<(A, B, C)>()?,
    );

    let mut removes = (
        database.remove::<()>()?,
        database.remove::<A>()?,
        database.remove::<B>()?,
        database.remove::<C>()?,
        database.remove::<(A, B)>()?,
        database.remove::<(A, C)>()?,
        database.remove::<(B, C)>()?,
        database.remove::<(A, B, C)>()?,
    );

    let mut destroys = (
        database.destroy(),
        database.destroy().filter::<Has<A>>(),
        database.destroy().filter_with(has::<B>()),
        database.destroy().filter::<Has<C>>(),
        database.destroy().filter_with((has::<A>(), has::<B>())),
        database.destroy().filter::<Has<(A, C)>>(),
        database.destroy().filter_with((has::<B>(), has::<C>())),
        database.destroy().filter::<Has<(A, B, C)>>(),
    );

    fn create<T: Template + Default>(
        count: usize,
        resolve: bool,
        create: &mut Create<T>,
        database: &Database,
    ) -> Result<(), Box<dyn error::Error + Sync + Send>> {
        let keys: Vec<_> = create.defaults(count).iter().copied().collect();
        let mut query = database.query::<()>()?.filter::<Has<T>>();
        let mut by = By::new();
        by.keys(keys.iter().copied());

        prove!(keys.len() == count)?;
        prove!(by.len() == count)?;
        prove!(keys.iter().all(|&key| key != Key::NULL))?;

        if resolve {
            let resolved = create.resolve();
            prove!(keys.len() <= resolved)?;
            prove!(database
                .keys()
                .get_all(keys.iter().copied())
                .all(|(_, slot)| slot.is_ok()))?;
            prove!(query.count_by(&by) == count)?;
            for &key in keys.iter() {
                prove!(database.keys().get(key).is_ok())?;
                prove!(query.find(key, |_| ()))??;
            }
        } else {
            for &key in keys.iter() {
                prove!(database.keys().get(key).is_err())?;
                prove!(query.find(key, |_| ()).is_err())?;
            }
        }
        Ok(())
    }

    fn add<T: Template + Default>(
        count: usize,
        resolve: bool,
        add: &mut Add<T, impl Filter>,
        query: &mut Query<impl Row, impl Filter>,
        database: &Database,
    ) -> Result<(), Box<dyn error::Error + Sync + Send>> {
        let keys: Vec<_> = query.keys();
        let keys = &keys[..count.min(keys.len())];
        add.all(keys.iter().copied());
        if resolve {
            add.resolve();
            for &key in keys {
                prove!(database.keys().get(key).is_ok())?;
                prove!(query.find(key, |_| ()))??;
            }
        }
        Ok(())
    }

    fn remove<T: Template + Default>(
        count: usize,
        resolve: bool,
        remove: &mut Remove<T, impl Filter>,
        query: &mut Query<impl Row, impl Filter>,
        database: &Database,
    ) -> Result<(), Box<dyn error::Error + Sync + Send>> {
        let keys: Vec<_> = query.keys();
        let keys = &keys[..count.min(keys.len())];
        remove.all(keys.iter().copied());
        if resolve {
            remove.resolve();
            for &key in keys {
                prove!(database.keys().get(key).is_ok())?;
                prove!(query.find(key, |_| ()).is_err())?;
            }
        }
        Ok(())
    }

    fn destroy(
        count: usize,
        resolve: bool,
        destroy: &mut Destroy<impl Filter>,
        query: &mut Query<impl Row, impl Filter>,
        database: &Database,
    ) -> Result<(), Box<dyn error::Error + Sync + Send>> {
        let keys: Vec<_> = query.keys();
        let keys = &keys[..count.min(keys.len())];
        destroy.all(keys.iter().copied());
        if resolve {
            let resolved = destroy.resolve();
            prove!(keys.len() <= resolved)?;
            for &key in keys {
                prove!(database.keys().get(key).is_err())?;
                prove!(query.find(key, |_| ()).is_err())?;
            }
        } else {
            for &key in keys {
                prove!(database.keys().get(key).is_ok())?;
                prove!(query.find(key, |_| ()))??;
            }
        }
        Ok(())
    }

    for action in actions {
        match action {
            Action::Create(count, Type::Unit, resolve) => {
                create(count, resolve, &mut creates.0, &database)?
            }
            Action::Create(count, Type::A, resolve) => {
                create(count, resolve, &mut creates.1, &database)?
            }
            Action::Create(count, Type::B, resolve) => {
                create(count, resolve, &mut creates.2, &database)?
            }
            Action::Create(count, Type::C, resolve) => {
                create(count, resolve, &mut creates.3, &database)?
            }
            Action::Create(count, Type::AB, resolve) => {
                create(count, resolve, &mut creates.4, &database)?
            }
            Action::Create(count, Type::AC, resolve) => {
                create(count, resolve, &mut creates.5, &database)?
            }
            Action::Create(count, Type::BC, resolve) => {
                create(count, resolve, &mut creates.6, &database)?
            }
            Action::Create(count, Type::ABC, resolve) => {
                create(count, resolve, &mut creates.7, &database)?
            }
            Action::Add(count, Type::Unit, resolve) => {
                add(count, resolve, &mut adds.0, &mut queries.0, &database)?
            }
            Action::Add(count, Type::A, resolve) => {
                add(count, resolve, &mut adds.1, &mut queries.2, &database)?
            }
            Action::Add(count, Type::B, resolve) => {
                add(count, resolve, &mut adds.2, &mut queries.1, &database)?
            }
            Action::Add(count, Type::C, resolve) => {
                add(count, resolve, &mut adds.3, &mut queries.1, &database)?
            }
            Action::Remove(count, Type::Unit, resolve) => {
                remove(count, resolve, &mut removes.0, &mut queries.8, &database)?
            }
            Action::Remove(count, Type::A, resolve) => {
                remove(count, resolve, &mut removes.1, &mut queries.1, &database)?
            }
            Action::Remove(count, Type::B, resolve) => {
                remove(count, resolve, &mut removes.2, &mut queries.4, &database)?
            }
            Action::Remove(count, Type::C, resolve) => {
                remove(count, resolve, &mut removes.3, &mut queries.6, &database)?
            }
            Action::Destroy(count, Type::Unit, resolve) => {
                destroy(count, resolve, &mut destroys.0, &mut queries.0, &database)?
            }
            Action::Destroy(count, Type::A, resolve) => {
                destroy(count, resolve, &mut destroys.1, &mut queries.1, &database)?
            }
            Action::Destroy(count, Type::B, resolve) => {
                destroy(count, resolve, &mut destroys.2, &mut queries.2, &database)?
            }
            Action::Destroy(count, Type::C, resolve) => {
                destroy(count, resolve, &mut destroys.3, &mut queries.3, &database)?
            }
            Action::Destroy(count, Type::AB, resolve) => {
                destroy(count, resolve, &mut destroys.4, &mut queries.4, &database)?
            }
            Action::Destroy(count, Type::AC, resolve) => {
                destroy(count, resolve, &mut destroys.5, &mut queries.5, &database)?
            }
            Action::Destroy(count, Type::BC, resolve) => {
                destroy(count, resolve, &mut destroys.6, &mut queries.6, &database)?
            }
            Action::Destroy(count, Type::ABC, resolve) => {
                destroy(count, resolve, &mut destroys.7, &mut queries.7, &database)?
            }
            _ => {}
        }
    }
    Ok::<(), Box<dyn error::Error + Send + Sync>>(())
}
