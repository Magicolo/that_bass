pub mod common;
use common::*;

#[test]
fn create_adds_a_table() -> Result<(), Error> {
    let database = Database::new();
    assert_eq!(database.tables().len(), 0);
    database.create::<()>()?;
    assert_eq!(database.tables().len(), 1);
    Ok(())
}

#[test]
fn create_adds_a_table_with_datum() -> Result<(), Error> {
    let database = Database::new();
    assert_eq!(database.tables().len(), 0);
    database.create::<A>()?;
    assert!(database.tables().get(0)?.has::<A>());
    Ok(())
}

#[test]
fn create_adds_a_table_with_data() -> Result<(), Error> {
    let database = Database::new();
    assert_eq!(database.tables().len(), 0);
    database.create::<(A, B)>()?;
    assert!(database.tables().get(0)?.has::<A>());
    assert!(database.tables().get(0)?.has::<B>());
    Ok(())
}

#[test]
fn create_one_returns_non_null_key() -> Result<(), Error> {
    let database = Database::new();
    assert_ne!(create_one(&database, ())?, Key::NULL);
    Ok(())
}

#[test]
fn create_all_returns_no_null_key() -> Result<(), Error> {
    let database = Database::new();
    assert!(create_n(&database, [(); COUNT])?.iter().all(Key::valid));
    Ok(())
}

#[test]
fn create_all_returns_distinct_keys() -> Result<(), Error> {
    let database = Database::new();
    let mut set = HashSet::new();
    let mut create = database.create()?;
    assert!(create.all([(); COUNT]).iter().all(|&key| set.insert(key)));
    Ok(())
}

#[test]
fn create_fail_with_duplicate_datum_in_template() {
    let database = Database::new();
    let result = database.create::<(A, A)>();
    assert_eq!(result.err(), Some(Error::DuplicateMeta));
}

#[test]
fn create_destroy_create_reuses_key_index() -> Result<(), Error> {
    let database = Database::new();
    let key1 = create_one(&database, ())?;
    assert_eq!(destroy_one(&database, key1), Ok(()));
    let key2 = create_one(&database, ())?;
    assert_eq!(key1.index(), key2.index());
    assert_ne!(key1.generation(), key2.generation());
    Ok(())
}
