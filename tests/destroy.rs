pub mod common;
use common::*;

#[test]
fn destroy_one_fails_with_null_key() {
    let database = Database::new();
    let mut destroy = database.destroy();
    assert_eq!(
        database.keys().get(Key::NULL).err(),
        Some(Error::InvalidKey(Key::NULL))
    );
    destroy.one(Key::NULL);
    assert_eq!(destroy.resolve(), 0);
    assert_eq!(
        database.keys().get(Key::NULL).err(),
        Some(Error::InvalidKey(Key::NULL))
    );
}

#[test]
fn destroy_one_true_with_create_one() -> Result<(), Error> {
    let database = Database::new();
    let key = create_one(&database, ())?;
    let mut destroy = database.destroy();
    assert!(database.keys().get(key).is_ok());
    destroy.one(key);
    assert!(database.keys().get(key).is_ok());
    assert_eq!(destroy.resolve(), 1);
    assert_eq!(database.keys().get(key).err(), Some(Error::InvalidKey(key)));
    Ok(())
}

#[test]
fn destroy_all_n_with_create_all_n_resolves_n() -> Result<(), Error> {
    let database = Database::new();
    let count = |created: &[Key]| {
        database
            .keys()
            .get_all(created.iter().copied())
            .filter(|(_, result)| result.is_ok())
            .count()
    };

    let keys = create_n(&database, [(); COUNT])?;
    let mut destroy = database.destroy();
    assert_eq!(count(&keys), COUNT);
    destroy.all(keys.clone());
    assert_eq!(count(&keys), COUNT);
    assert_eq!(destroy.resolve(), COUNT);
    assert_eq!(count(&keys), 0);
    Ok(())
}

#[test]
fn destroy_all_filter() -> Result<(), Error> {
    let database = Database::new();
    create_n(&database, [(); COUNT])?;
    create_n(&database, [A; COUNT])?;
    assert_eq!(database.destroy_all().filter::<Has<A>>().resolve(), COUNT);
    assert_eq!(database.destroy_all().resolve(), COUNT);
    Ok(())
}

#[test]
fn destroy_all_resolve_0() -> Result<(), Error> {
    let database = Database::new();
    assert_eq!(database.destroy_all().resolve(), 0);
    Ok(())
}

#[test]
fn destroy_all_resolve_100() -> Result<(), Error> {
    let database = Database::new();
    create_n(&database, [(); COUNT])?;
    assert_eq!(database.destroy_all().resolve(), COUNT);
    Ok(())
}
