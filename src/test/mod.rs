use crate as that_bass;
use std::{collections::HashSet, marker::PhantomData, thread::scope};
use that_bass::{
    filter::{Filter, Has, Is, Not},
    key::Key,
    query::By,
    template::Template,
    Database, Datum, Error, Filter, Template,
};

#[derive(Debug, Clone, Copy, Datum)]
pub struct A;
#[derive(Debug, Clone, Copy, Datum)]
pub struct B;
#[derive(Debug, Clone, Copy, Datum)]
pub struct C(usize);

pub mod filter {
    use super::*;

    #[derive(Filter)]
    pub struct UnitStruct;
    #[derive(Filter)]
    pub struct EmptyTupleStruct();
    #[derive(Filter)]
    pub struct EmptyMapStruct {}
    #[derive(Filter)]
    pub struct GenericTupleStruct<T: Filter, U>(T, PhantomData<U>);
    #[derive(Filter)]
    pub struct TupleStruct(
        UnitStruct,
        EmptyTupleStruct,
        EmptyMapStruct,
        Has<A>,
        (Is<B>, Not<Has<C>>),
        (),
        GenericTupleStruct<UnitStruct, [bool; 32]>,
    );
    #[derive(Filter)]
    pub struct MapStruct {
        a: UnitStruct,
        b: EmptyTupleStruct,
        c: TupleStruct,
        d: EmptyMapStruct,
        e: Has<A>,
        f: (Is<B>, Not<Has<C>>),
        h: (),
        i: GenericTupleStruct<(), usize>,
    }
    #[derive(Filter)]
    pub enum GenericEnum<T: Filter, U> {
        A(T),
        B(PhantomData<U>),
    }
    #[derive(Filter)]
    pub enum Enum {
        A,
        B(),
        C(Has<A>),
        D(
            Is<B>,
            Not<Has<C>>,
            TupleStruct,
            MapStruct,
            GenericEnum<MapStruct, (char,)>,
        ),
        E {},
        F {
            a: UnitStruct,
            b: EmptyTupleStruct,
            c: TupleStruct,
            d: MapStruct,
            e: GenericTupleStruct<EmptyTupleStruct, i32>,
        },
    }
}

pub mod template {
    use super::*;

    #[derive(Template)]
    pub struct UnitStruct;
    #[derive(Template)]
    pub struct EmptyTupleStruct();
    #[derive(Template)]
    pub struct EmptyMapStruct {}
    #[derive(Template)]
    pub struct GenericTupleStruct<T: Template, U: 'static>(T, PhantomData<U>);
    #[derive(Template)]
    pub struct TupleStruct(
        UnitStruct,
        EmptyTupleStruct,
        EmptyMapStruct,
        A,
        (B, C),
        (),
        GenericTupleStruct<UnitStruct, [bool; 32]>,
    );
    #[derive(Template)]
    pub struct MapStruct {
        a: UnitStruct,
        b: EmptyTupleStruct,
        c: TupleStruct,
        d: EmptyMapStruct,
        e: A,
        f: (B, C),
        h: (),
        i: GenericTupleStruct<(), usize>,
    }
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
    let table = database.tables().get(0).unwrap();
    assert!(table.has::<A>());
    Ok(())
}

#[test]
fn create_adds_a_table_with_data() -> Result<(), Error> {
    let database = Database::new();
    assert_eq!(database.tables().len(), 0);
    database.create::<(A, B)>()?;
    let table = database.tables().get(0).unwrap();
    assert!(table.has::<A>());
    assert!(table.has::<B>());
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
    assert!(create_n(&database, [(); 1000])?.iter().all(Key::valid));
    Ok(())
}

#[test]
fn create_all_returns_distinct_keys() -> Result<(), Error> {
    let database = Database::new();
    let mut set = HashSet::new();
    let mut create = database.create()?;
    assert!(create.all([(); 1000]).iter().all(|&key| set.insert(key)));
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

#[test]
fn destroy_one_fails_with_null_key() {
    let database = Database::new();
    let mut destroy = database.destroy();
    assert_eq!(
        database.keys().get(Key::NULL).err(),
        Some(Error::InvalidKey)
    );
    destroy.one(Key::NULL);
    assert_eq!(destroy.resolve(), 0);
    assert_eq!(
        database.keys().get(Key::NULL).err(),
        Some(Error::InvalidKey)
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
    assert_eq!(database.keys().get(key).err(), Some(Error::InvalidKey));
    Ok(())
}

#[test]
fn destroy_all_n_with_create_all_n_resolves_n() -> Result<(), Error> {
    let database = Database::new();
    let count = |keys: &[Key]| {
        database
            .keys()
            .get_all(keys.iter().copied())
            .filter(|(_, result)| result.is_ok())
            .count()
    };

    let keys = create_n(&database, [(); 1000])?;
    let mut destroy = database.destroy();
    assert_eq!(count(&keys), 1000);
    destroy.all(keys.clone());
    assert_eq!(count(&keys), 1000);
    assert_eq!(destroy.resolve(), 1000);
    assert_eq!(count(&keys), 0);
    Ok(())
}

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
    add.one(key, A);
    assert_eq!(add.resolve(), 1);
    assert_eq!(add.resolve(), 0);
    Ok(())
}

#[test]
fn add_simple_template() -> Result<(), Error> {
    let database = Database::new();
    let key = create_one(&database, ())?;
    let mut add = database.add()?;
    add.one(key, A);
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

    add_a.one(key, A);
    add_b.one(key, B);
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
    add.one(key, (A, B));
    assert!(!database.query::<&A>()?.has(key));
    assert!(!database.query::<&B>()?.has(key));
    assert!(!database.query::<(&A, &B)>()?.has(key));

    assert_eq!(add.resolve(), 1);
    assert!(database.query::<&A>()?.has(key));
    assert!(database.query::<&B>()?.has(key));
    assert!(database.query::<(&A, &B)>()?.has(key));
    Ok(())
}

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
    let key = create_one(&database, (A, B))?;
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
    let key = create_one(&database, (A, B))?;
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

#[test]
fn destroy_all_resolve_0() -> Result<(), Error> {
    let database = Database::new();
    assert_eq!(database.destroy_all().resolve(), 0);
    Ok(())
}

#[test]
fn destroy_all_resolve_100() -> Result<(), Error> {
    let database = Database::new();
    create_n(&database, [(); 100])?;
    assert_eq!(database.destroy_all().resolve(), 100);
    Ok(())
}

#[test]
fn destroy_all_filter() -> Result<(), Error> {
    let database = Database::new();
    create_n(&database, [(); 100])?;
    create_n(&database, [A; 100])?;
    assert_eq!(database.destroy_all().filter::<Has<A>>().resolve(), 100);
    assert_eq!(database.destroy_all().resolve(), 100);
    Ok(())
}

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
    assert_eq!(query.find(key, |_| {}).err(), Some(Error::InvalidKey));
    Ok(())
}

#[test]
fn query_is_some_all_create_all() -> Result<(), Error> {
    let database = Database::new();
    let keys = create_n(&database, [(); 1000])?;
    let mut query = database.query::<()>()?;
    assert!(keys.iter().all(|&key| query.find(key, |_| {}).is_ok()));
    Ok(())
}

#[test]
fn query_is_some_remain_destroy_all() -> Result<(), Error> {
    let database = Database::new();
    let keys = create_n(&database, [(); 1000])?;
    destroy_all(&database, &keys[..500]);
    let mut query = database.query::<()>()?;
    assert!(keys[..500]
        .iter()
        .all(|&key| query.find(key, |_| {}).is_err()));
    assert!(keys[500..]
        .iter()
        .all(|&key| query.find(key, |_| {}).is_ok()));
    Ok(())
}

#[test]
fn query_is_err_with_write_write() {
    let database = Database::new();
    assert_eq!(
        database.query::<(&mut A, &mut A)>().err(),
        Some(Error::WriteWriteConflict)
    );
}

#[test]
fn query_is_err_with_read_write() {
    let database = Database::new();
    assert_eq!(
        database.query::<(&A, &mut A)>().err(),
        Some(Error::ReadWriteConflict)
    );
    assert_eq!(
        database.query::<(&mut A, &A)>().err(),
        Some(Error::ReadWriteConflict)
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
fn query_reads_same_datum_as_create_one() -> Result<(), Error> {
    let database = Database::new();
    let key = create_one(&database, C(1))?;
    assert_eq!(database.query::<&C>()?.find(key, |c| c.0), Ok(1));
    Ok(())
}

#[test]
fn query1_reads_datum_written_by_query2() -> Result<(), Error> {
    let database = Database::new();
    let key = create_one(&database, C(1))?;
    database.query::<&mut C>()?.find(key, |c| c.0 += 1)?;
    assert_eq!(database.query::<&C>()?.find(key, |c| c.0), Ok(2));
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
    create_n(&database, [C(0); 25])?;
    create_n(&database, [(A, C(0)); 50])?;
    create_n(&database, [(B, C(0)); 75])?;
    create_n(&database, [(A, B, C(0)); 100])?;
    let mut query = database.query::<&mut C>()?;
    assert_eq!(query.split().len(), 4);
    assert!(query
        .split()
        .enumerate()
        .all(|(i, split)| split.count() == (i + 1) * 25));

    scope(|scope| {
        for split in query.split() {
            scope.spawn(move || split.each(|c| c.0 += 1));
        }
    });
    query.each(|c| assert_eq!(c.0, 1));
    Ok(())
}

#[test]
fn query_split_chunk_on_multiple_threads() -> Result<(), Error> {
    let database = Database::new();
    create_n(&database, [C(0); 25])?;
    create_n(&database, [(A, C(0)); 50])?;
    create_n(&database, [(B, C(0)); 75])?;
    create_n(&database, [(A, B, C(0)); 100])?;
    let mut query = database.query::<&mut C>()?.chunk();
    assert_eq!(query.count(), 4);
    assert_eq!(query.split().len(), 4);

    scope(|scope| {
        for split in query.split() {
            scope.spawn(move || {
                let value = split.map(|c| {
                    for c in c {
                        c.0 += 1;
                    }
                });
                assert_eq!(value, Some(()));
            });
        }
    });
    Ok(())
}

#[test]
fn multi_join() -> Result<(), Error> {
    struct A(Vec<Key>);
    impl Datum for A {}

    let database = Database::new();
    let a = create_one(&database, A(vec![]))?;
    let b = create_one(&database, A(vec![a, a, Key::NULL]))?;
    create_one(&database, A(vec![Key::NULL, Key::NULL, a, b]))?;

    let mut query = database.query::<&A>()?;
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
fn copy_to() -> Result<(), Error> {
    struct CopyTo(Key);
    impl Datum for CopyTo {}
    const COUNT: usize = 10;

    let database = Database::new();
    let mut a = create_one(&database, C(1))?;
    let mut create = database.create()?;
    for i in 0..COUNT {
        a = create.one((C(i), CopyTo(a)));
    }
    assert_eq!(create.resolve(), COUNT);

    let mut sources = database.query::<(&C, &CopyTo)>()?;
    let mut targets = database.query::<&mut C>()?;
    let mut by = By::new();
    sources.each(|(c, copy)| by.pair(copy.0, c.0));
    targets.each_by_ok(&mut by, |value, c| c.0 = value);
    // TODO: Add assertions.
    Ok(())
}

#[test]
fn copy_from() -> Result<(), Error> {
    struct CopyFrom(Key);
    impl Datum for CopyFrom {}
    const COUNT: usize = 10;

    let database = Database::new();
    let a = create_one(&database, C(1))?;
    let mut create = database.create()?;
    for i in 0..COUNT {
        create.one((C(i), CopyFrom(a)));
    }
    assert_eq!(create.resolve(), COUNT);

    let mut copies = database.query::<(Key, &CopyFrom)>()?;
    let mut sources = database.query::<&C>()?;
    let mut targets = database.query::<&mut C>()?;
    let mut by_source = By::new();
    let mut by_target = By::new();

    copies.each(|(key, copy)| by_source.pair(copy.0, key));
    assert_eq!(by_source.len(), COUNT);
    sources.each_by_ok(&mut by_source, |target, c| by_target.pair(target, c.0));
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
fn swap() -> Result<(), Error> {
    struct Swap(Key, Key);
    impl Datum for Swap {}
    const COUNT: usize = 10;

    let database = Database::new();
    let mut a = create_one(&database, C(1))?;
    let mut b = create_one(&database, C(2))?;
    let mut create = database.create()?;
    for i in 0..COUNT {
        let c = create.one((C(i), Swap(a, b)));
        a = b;
        b = c;
    }
    assert_eq!(create.resolve(), COUNT);

    let mut swaps = database.query::<&Swap>()?;
    let mut sources = database.query::<&C>()?;
    let mut targets = database.query::<&mut C>()?;
    let mut by_source = By::new();
    let mut by_target = By::new();
    swaps.each(|swap| by_source.pairs([(swap.0, swap.1), (swap.1, swap.0)]));
    sources.each_by_ok(&mut by_source, |target, c| by_target.pair(target, c.0));
    targets.each_by_ok(&mut by_target, |value, c| c.0 = value);
    // TODO: Add assertions.
    Ok(())
}
