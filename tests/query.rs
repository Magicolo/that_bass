pub mod common;
use common::*;

#[test]
fn query_is_some_create_one_key() -> Result<(), Error> {
    let database = Database::new();
    let key = create_one(&database, ())?;
    assert!(database.query::<()>()?.has(key));
    assert_eq!(database.query::<()>()?.find(key, |_| true), Ok(true));
    Ok(())
}

#[test]
fn query_is_none_destroy_one_key() -> Result<(), Error> {
    let database = Database::new();
    let key = create_one(&database, ())?;
    destroy_one(&database, key)?;
    let mut query = database.query::<()>()?;
    assert_eq!(query.find(key, |_| {}).err(), Some(Error::InvalidKey(key)));
    Ok(())
}

#[test]
fn query_is_some_all_create_all() -> Result<(), Error> {
    let database = Database::new();
    let keys = create_n(&database, [(); COUNT])?;
    let mut query = database.query::<()>()?;
    assert!(keys.iter().all(|&key| query.find(key, |_| {}).is_ok()));
    Ok(())
}

#[test]
fn query_is_some_remain_destroy_all() -> Result<(), Error> {
    let database = Database::new();
    let keys = create_n(&database, [(); COUNT])?;
    destroy_all(&database, &keys[..COUNT / 2]);
    let mut query = database.query::<()>()?;
    assert!(keys[..COUNT / 2]
        .iter()
        .all(|&key| query.find(key, |_| {}).is_err()));
    assert!(keys[COUNT / 2..]
        .iter()
        .all(|&key| query.find(key, |_| {}).is_ok()));
    Ok(())
}

#[test]
fn query_is_err_with_write_write() {
    let database = Database::new();
    assert_eq!(
        database.query::<(&mut A, &mut A)>().err(),
        Some(Error::WriteWriteConflict(TypeId::of::<A>()))
    );
}

#[test]
fn query_is_err_with_read_write() {
    let database = Database::new();
    assert_eq!(
        database.query::<(&A, &mut A)>().err(),
        Some(Error::ReadWriteConflict(TypeId::of::<A>()))
    );
    assert_eq!(
        database.query::<(&mut A, &A)>().err(),
        Some(Error::ReadWriteConflict(TypeId::of::<A>()))
    );
}

#[test]
fn query_with_false_is_always_empty() -> Result<(), Error> {
    let database = Database::new();
    let mut query = database.query::<()>()?.filter_with(false);
    assert_eq!(query.count(), 0);
    let key = create_one(&database, ())?;
    assert_eq!(query.count(), 0);
    assert_eq!(
        query.find(key, |_| {}).err(),
        Some(Error::KeyNotInQuery(key))
    );
    Ok(())
}

#[test]
fn query_with_wrong_filter_is_empty() -> Result<(), Error> {
    let database = Database::new();
    let mut query = database.query::<()>()?.filter::<Has<A>>();
    assert_eq!(query.count(), 0);
    let keys = create_n(&database, [(); COUNT])?;
    let mut by = By::new();
    assert_eq!(by.len(), 0);
    by.keys(keys.iter().copied());
    assert_eq!(by.len(), COUNT);
    assert_eq!(query.count(), 0);
    assert_eq!(query.count_by(&by), 0);

    for &key in keys.iter() {
        assert!(database.keys().get(key).is_ok());
        assert_eq!(
            query.find(key, |_| true).err(),
            Some(Error::KeyNotInQuery(key))
        );
    }
    Ok(())
}

#[test]
fn query_reads_same_datum_as_create_one() -> Result<(), Error> {
    let database = Database::new();
    let key = create_one(&database, B(1))?;
    assert_eq!(database.query::<&B>()?.find(key, |b| b.0), Ok(1));
    Ok(())
}

#[test]
fn query1_reads_datum_written_by_query2() -> Result<(), Error> {
    let database = Database::new();
    let key = create_one(&database, B(1))?;
    database.query::<&mut B>()?.find(key, |b| b.0 += 1)?;
    assert_eq!(database.query::<&B>()?.find(key, |b| b.0), Ok(2));
    Ok(())
}

#[test]
fn query_option_read() -> Result<(), Error> {
    let database = Database::new();
    let key1 = create_one(&database, ())?;
    let key2 = create_one(&database, A)?;
    let mut query = database.query::<Option<&A>>()?;
    assert_eq!(query.count(), 2);
    assert_eq!(query.find(key1, |a| a.is_some()), Ok(false));
    assert_eq!(query.find(key2, |a| a.is_some()), Ok(true));
    assert_eq!(query.chunk().count(), 2);
    Ok(())
}

#[test]
fn query_split_item_on_multiple_threads() -> Result<(), Error> {
    let database = Database::new();
    create_n(&database, [C(0.0); COUNT])?;
    create_n(&database, [(A, C(0.0)); COUNT * 2])?;
    create_n(&database, [(B(0), C(0.0)); COUNT * 3])?;
    create_n(&database, [(A, B(0), C(0.0)); COUNT * 4])?;
    let mut query = database.query::<&mut C>()?;
    assert_eq!(query.split().len(), 4);
    assert!(query
        .split()
        .enumerate()
        .all(|(i, split)| split.count() == (i + 1) * COUNT));

    scope(|scope| {
        for (i, split) in query.split().enumerate() {
            assert_eq!(split.count(), (i + 1) * COUNT);
            scope.spawn(move || split.each(|c| c.0 += 1.0));
        }
    });
    query.each(|c| assert_eq!(c.0, 1.0));
    Ok(())
}

#[test]
fn query_split_chunk_on_multiple_threads() -> Result<(), Error> {
    let database = Database::new();
    create_n(&database, [C(0.0); COUNT])?;
    create_n(&database, [(A, C(0.0)); COUNT * 2])?;
    create_n(&database, [(B(0), C(0.0)); COUNT * 3])?;
    create_n(&database, [(A, B(0), C(0.0)); COUNT * 4])?;
    let mut query = database.query::<&mut C>()?.chunk();
    assert_eq!(query.count(), 4);
    assert_eq!(query.split().len(), 4);

    scope(|scope| {
        for (i, split) in query.split().enumerate() {
            scope.spawn(move || {
                let value = split.map(|c| {
                    assert_eq!(c.len(), (i + 1) * COUNT);
                    for c in c {
                        c.0 += 1.0;
                    }
                });
                assert_eq!(value, Some(()));
            });
        }
    });
    Ok(())
}

#[test]
fn query_multi_join() -> Result<(), Error> {
    #[derive(Datum)]
    struct Join(Vec<Key>);

    let database = Database::new();
    let a = create_one(&database, Join(vec![]))?;
    let b = create_one(&database, Join(vec![a, a, Key::NULL]))?;
    create_one(&database, Join(vec![Key::NULL, Key::NULL, a, b]))?;

    let mut query = database.query::<&Join>()?;
    assert_eq!(query.count(), 3);
    let mut by = By::new();
    assert_eq!(by.len(), 0);
    query.each(|a| by.keys(a.0.iter().copied()));
    assert_eq!(by.len(), 7);
    assert_eq!(query.count_by(&by), 4);

    let mut query = database.query::<Key>()?;
    assert_eq!(query.count(), 3);
    let sum = query.fold_by(&mut by, 0, |sum, _, item| match item {
        Ok(key) if key == a || key == b => sum + 1,
        _ => sum,
    });
    assert_eq!(by.len(), 0);
    assert_eq!(sum, 4);
    Ok(())
}

#[test]
fn query_copy_to() -> Result<(), Error> {
    #[derive(Datum)]
    struct CopyTo(Key);

    let database = Database::new();
    let mut a = create_one(&database, B(1))?;
    let mut create = database.create()?;
    for i in 0..COUNT {
        a = create.one((B(i), CopyTo(a)));
    }
    assert_eq!(create.resolve(), COUNT);

    let mut sources = database.query::<(&B, &CopyTo)>()?;
    let mut targets = database.query::<&mut B>()?;
    let mut by = By::new();
    sources.each(|(b, copy)| by.pair(copy.0, b.0));
    targets.each_by_ok(&mut by, |value, b| b.0 = value);
    // TODO: Add assertions.
    Ok(())
}

#[test]
fn query_copy_from() -> Result<(), Error> {
    #[derive(Datum)]
    struct CopyFrom(Key);

    let database = Database::new();
    let a = create_one(&database, B(1))?;
    let mut create = database.create()?;
    for i in 0..COUNT {
        create.one((B(i), CopyFrom(a)));
    }
    assert_eq!(create.resolve(), COUNT);

    let mut copies = database.query::<(Key, &CopyFrom)>()?;
    let mut sources = database.query::<&B>()?;
    let mut targets = database.query::<&mut B>()?;
    let mut by_source = By::new();
    let mut by_target = By::new();

    copies.each(|(key, copy)| by_source.pair(copy.0, key));
    assert_eq!(by_source.len(), COUNT);
    sources.each_by_ok(&mut by_source, |target, b| by_target.pair(target, b.0));
    assert_eq!(by_source.len(), 0);
    assert_eq!(by_target.len(), COUNT);
    targets.each_by_ok(&mut by_target, |value, c| c.0 = value);
    assert_eq!(by_target.len(), 0);

    assert_eq!(copies.count(), COUNT);
    assert_eq!(sources.count(), COUNT + 1);
    assert_eq!(targets.count(), COUNT + 1);
    assert_eq!(by_source.len(), 0);
    assert_eq!(by_target.len(), 0);
    sources.each(|c| assert_eq!(c.0, 1));
    targets.each(|c| assert_eq!(c.0, 1));
    Ok(())
}

#[test]
fn query_swap() -> Result<(), Error> {
    #[derive(Datum)]
    struct Swap(Key, Key);

    let database = Database::new();
    let mut a = create_one(&database, B(1))?;
    let mut b = create_one(&database, B(2))?;
    let mut create = database.create()?;
    for i in 0..COUNT {
        let c = create.one((B(i), Swap(a, b)));
        a = b;
        b = c;
    }
    assert_eq!(create.resolve(), COUNT);

    let mut swaps = database.query::<&Swap>()?;
    let mut sources = database.query::<&B>()?;
    let mut targets = database.query::<&mut B>()?;
    let mut by_source = By::new();
    let mut by_target = By::new();
    swaps.each(|swap| by_source.pairs([(swap.0, swap.1), (swap.1, swap.0)]));
    sources.each_by_ok(&mut by_source, |target, b| by_target.pair(target, b.0));
    targets.each_by_ok(&mut by_target, |value, b| b.0 = value);
    // TODO: Add assertions.
    Ok(())
}
