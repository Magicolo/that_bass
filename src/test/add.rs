use super::*;

#[test]
fn add_resolve_is_0() -> Result<(), Error> {
    let database = Database::new();
    assert_eq!(database.add::<()>()?.resolve(), 0);
    Ok(())
}

#[test]
fn add_no_double_resolve() -> Result<(), Error> {
    let database = Database::new();
    let key = create_one(&database, ())?;
    let mut add = database.add()?;
    add.one_with(key, A);
    assert_eq!(add.resolve(), 1);
    assert_eq!(add.resolve(), 0);
    Ok(())
}

#[test]
fn add_simple_template() -> Result<(), Error> {
    let database = Database::new();
    let key = create_one(&database, ())?;
    let mut add = database.add()?;
    add.one_with(key, A);
    assert!(!database.query::<&A>()?.has(key));
    assert_eq!(add.resolve(), 1);
    assert!(database.query::<&A>()?.has(key));
    Ok(())
}

#[test]
fn add_simple_template_twice() -> Result<(), Error> {
    let database = Database::new();
    let key = create_one(&database, ())?;
    let mut add_a = database.add()?;
    let mut add_b = database.add()?;

    add_a.one_with(key, A);
    add_b.one_with(key, B);
    assert!(database.query::<()>()?.has(key));
    assert!(!database.query::<&A>()?.has(key));
    assert!(!database.query::<&B>()?.has(key));
    assert!(!database.query::<(&A, &B)>()?.has(key));

    assert_eq!(add_a.resolve(), 1);
    assert!(database.query::<()>()?.has(key));
    assert!(database.query::<&A>()?.has(key));
    assert!(!database.query::<&B>()?.has(key));
    assert!(!database.query::<(&A, &B)>()?.has(key));

    assert_eq!(add_b.resolve(), 1);
    assert!(database.query::<()>()?.has(key));
    assert!(database.query::<&A>()?.has(key));
    assert!(database.query::<&B>()?.has(key));
    assert!(database.query::<(&A, &B)>()?.has(key));
    Ok(())
}

#[test]
fn add_composite_template() -> Result<(), Error> {
    let database = Database::new();
    let key = create_one(&database, ())?;
    let mut add = database.add()?;
    add.one_with(key, (A, B));
    assert!(!database.query::<&A>()?.has(key));
    assert!(!database.query::<&B>()?.has(key));
    assert!(!database.query::<(&A, &B)>()?.has(key));

    assert_eq!(add.resolve(), 1);
    assert!(database.query::<&A>()?.has(key));
    assert!(database.query::<&B>()?.has(key));
    assert!(database.query::<(&A, &B)>()?.has(key));
    Ok(())
}
