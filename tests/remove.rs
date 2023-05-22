pub mod common;
use common::*;

#[test]
fn remove_resolve_is_0() -> Result<(), Error> {
    let database = Database::new();
    assert_eq!(database.add::<()>()?.resolve(), 0);
    Ok(())
}

#[test]
fn remove_no_double_resolve() -> Result<(), Error> {
    let database = Database::new();
    let key = create_one(&database, A)?;

    let mut remove = database.remove::<A>()?;
    remove.one(key);
    assert_eq!(remove.resolve(), 1);
    assert_eq!(remove.resolve(), 0);
    Ok(())
}

#[test]
fn remove_simple_template() -> Result<(), Error> {
    let database = Database::new();
    let key = create_one(&database, A)?;
    let mut remove = database.remove::<A>()?;
    remove.one(key);
    assert!(database.query::<&A>()?.has(key));
    assert_eq!(remove.resolve(), 1);
    assert!(!database.query::<&A>()?.has(key));
    Ok(())
}

#[test]
fn remove_simple_template_twice() -> Result<(), Error> {
    let database = Database::new();
    let key = create_one(&database, (A, B(1)))?;
    let mut remove_a = database.remove::<A>()?;
    let mut remove_b = database.remove::<B>()?;

    remove_a.one(key);
    remove_b.one(key);
    assert!(database.query::<()>()?.has(key));
    assert!(database.query::<&A>()?.has(key));
    assert!(database.query::<&B>()?.has(key));
    assert!(database.query::<(&A, &B)>()?.has(key));

    assert_eq!(remove_a.resolve(), 1);
    assert!(database.query::<()>()?.has(key));
    assert!(!database.query::<&A>()?.has(key));
    assert!(database.query::<&B>()?.has(key));
    assert!(!database.query::<(&A, &B)>()?.has(key));

    assert_eq!(remove_b.resolve(), 1);
    assert!(database.query::<()>()?.has(key));
    assert!(!database.query::<&A>()?.has(key));
    assert!(!database.query::<&B>()?.has(key));
    assert!(!database.query::<(&A, &B)>()?.has(key));
    Ok(())
}

#[test]
fn remove_composite_template() -> Result<(), Error> {
    let database = Database::new();
    let key = create_one(&database, (A, B(1)))?;
    let mut remove = database.remove::<(A, B)>()?;

    remove.one(key);
    assert!(database.query::<()>()?.has(key));
    assert!(database.query::<&A>()?.has(key));
    assert!(database.query::<&B>()?.has(key));
    assert!(database.query::<(&A, &B)>()?.has(key));

    assert_eq!(remove.resolve(), 1);
    assert!(database.query::<()>()?.has(key));
    assert!(!database.query::<&A>()?.has(key));
    assert!(!database.query::<&B>()?.has(key));
    assert!(!database.query::<(&A, &B)>()?.has(key));
    Ok(())
}
