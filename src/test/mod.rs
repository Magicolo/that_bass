use crate as that_bass;
use std::{any::TypeId, collections::HashSet, marker::PhantomData, thread::scope};
use that_bass::{
    filter::{Filter, Has, Is, Not},
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

pub mod filter {
    use crate::filter::Any;

    use super::*;

    #[derive(Filter, Default)]
    pub struct UnitStruct;
    #[derive(Filter, Default)]
    pub struct EmptyTupleStruct();
    #[derive(Filter, Default)]
    pub struct EmptyMapStruct {}
    #[derive(Filter, Default)]
    pub struct GenericTupleStruct<T: Filter, U>(T, PhantomData<U>);
    #[derive(Filter, Default)]
    pub struct TupleStruct(
        UnitStruct,
        EmptyTupleStruct,
        EmptyMapStruct,
        Has<A>,
        (Is<B>, Not<Has<C>>),
        (),
        GenericTupleStruct<UnitStruct, [bool; 32]>,
    );
    #[derive(Filter, Default)]
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
    pub enum EmptyEnum {}
    #[derive(Filter, Default)]
    pub enum GenericEnum<T: Filter, U> {
        A(T),
        B(PhantomData<U>),
        #[default]
        C,
    }
    #[derive(Filter, Default)]
    pub enum Enum {
        #[default]
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

    #[test]
    fn compiles() -> Result<(), Error> {
        let database = Database::new();
        database.destroy().filter::<UnitStruct>();
        database.destroy().filter::<EmptyTupleStruct>();
        database.destroy().filter::<Any<EmptyMapStruct>>();
        database.destroy().filter::<(Enum, Enum)>();
        database.destroy().filter::<Any<(TupleStruct, MapStruct)>>();
        database
            .destroy()
            .filter::<GenericTupleStruct<UnitStruct, bool>>();
        database.destroy().filter::<GenericEnum<Any<()>, bool>>();
        Ok(())
    }
}

pub mod template {
    use super::*;

    #[derive(Template, Default)]
    pub struct UnitStruct;
    #[derive(Template, Default)]
    pub struct EmptyTupleStruct();
    #[derive(Template, Default)]
    pub struct EmptyMapStruct {}
    #[derive(Template, Default)]
    pub struct GenericTupleStruct<T: Template, U: 'static>(T, PhantomData<U>);
    #[derive(Template, Default)]
    pub struct TupleStruct(
        UnitStruct,
        EmptyTupleStruct,
        EmptyMapStruct,
        A,
        (B, C),
        (),
        GenericTupleStruct<UnitStruct, [bool; 32]>,
    );
    #[derive(Template, Default)]
    pub struct MapStruct {
        a: UnitStruct,
        b: EmptyTupleStruct,
        c: EmptyMapStruct,
        d: A,
        e: (B, C),
        f: (),
        g: GenericTupleStruct<(), usize>,
    }

    #[test]
    fn compiles() -> Result<(), Error> {
        let database = Database::new();
        database.create::<UnitStruct>()?.defaults(1);
        database.create::<EmptyTupleStruct>()?.defaults(1);
        database.create::<EmptyMapStruct>()?.defaults(1);
        database
            .create::<GenericTupleStruct<A, bool>>()?
            .defaults(1);
        database.create::<TupleStruct>()?.defaults(1);
        database.create::<MapStruct>()?.defaults(1);
        Ok(())
    }
}

// pub mod row {
//     use super::*;

//     struct Boba<'a>(&'a ());
//     trait Fett {}
//     impl<'a> Fett for Boba<'static> {}

//     #[derive(Row)]
//     pub struct UnitStruct;
//     #[derive(Row)]
//     pub struct EmptyTupleStruct();
//     #[derive(Row)]
//     pub struct EmptyMapStruct {}
//     #[derive(Row)]
//     pub struct GenericTupleStruct<T: Row, U: 'static>(T, PhantomData<U>);
//     #[derive(Row)]
//     pub struct TupleStruct<'a>(
//         UnitStruct,
//         EmptyTupleStruct,
//         EmptyMapStruct,
//         &'a A,
//         (&'a mut B, PhantomData<&'a C>),
//         (),
//         GenericTupleStruct<UnitStruct, [bool; 32]>,
//     );
//     #[derive(Row)]
//     pub struct MapStruct<'a> {
//         _a: UnitStruct,
//         _b: EmptyTupleStruct,
//         _c: TupleStruct<'a>,
//         _d: EmptyMapStruct,
//         _e: &'a A,
//         _f: (&'a B, &'a mut C),
//         _h: (),
//         _i: GenericTupleStruct<(), usize>,
//     }

//     #[test]
//     fn compiles() -> Result<(), Error> {
//         let database = Database::new();
//         database.query::<UnitStruct>()?.each(|_item| {});
//         database.query::<EmptyTupleStruct>()?.each(|_item| {});
//         database.query::<EmptyMapStruct>()?.each(|_item| {});
//         database
//             .query::<GenericTupleStruct<&A, bool>>()?
//             .each(|_item| {});
//         database.query::<TupleStruct>()?.each(|_item| {});
//         database.query::<MapStruct>()?.each(|_item| {});
//         Ok(())
//     }
// }

const COUNT: usize = 37;

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
    create_n(&database, [(); COUNT])?;
    assert_eq!(database.destroy_all().resolve(), COUNT);
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
    create_n(&database, [C(0); COUNT])?;
    create_n(&database, [(A, C(0)); COUNT * 2])?;
    create_n(&database, [(B, C(0)); COUNT * 3])?;
    create_n(&database, [(A, B, C(0)); COUNT * 4])?;
    let mut query = database.query::<&mut C>()?;
    assert_eq!(query.split().len(), 4);
    assert!(query
        .split()
        .enumerate()
        .all(|(i, split)| split.count() == (i + 1) * COUNT));

    scope(|scope| {
        for (i, split) in query.split().enumerate() {
            assert_eq!(split.count(), (i + 1) * COUNT);
            scope.spawn(move || split.each(|c| c.0 += 1));
        }
    });
    query.each(|c| assert_eq!(c.0, 1));
    Ok(())
}

#[test]
fn query_split_chunk_on_multiple_threads() -> Result<(), Error> {
    let database = Database::new();
    create_n(&database, [C(0); COUNT])?;
    create_n(&database, [(A, C(0)); COUNT * 2])?;
    create_n(&database, [(B, C(0)); COUNT * 3])?;
    create_n(&database, [(A, B, C(0)); COUNT * 4])?;
    let mut query = database.query::<&mut C>()?.chunk();
    assert_eq!(query.count(), 4);
    assert_eq!(query.split().len(), 4);

    scope(|scope| {
        for (i, split) in query.split().enumerate() {
            scope.spawn(move || {
                let value = split.map(|c| {
                    assert_eq!(c.len(), (i + 1) * COUNT);
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

#[test]
fn broadcast_on_add() -> Result<(), Error> {
    #[derive(Default, Debug)]
    struct A;
    impl Datum for A {}

    let database = Database::new();
    let mut create = database.create::<()>()?;
    let mut destroy = database.destroy_all();
    let mut on_add1 = database.on_add().with_key().with_type::<A>();
    let mut on_add2 = database.on_add().with_key().with_type::<A>();
    let mut on_add3 = database.on_add().with_key().with_type::<A>();
    let mut keys2 = Vec::new();
    let mut keys3 = Vec::new();

    for i in 0..0 {
        assert!(on_add1.next().is_none());
        let keys = create.defaults(i).to_vec();
        keys2.extend(keys.iter().copied());
        keys3.extend(keys.iter().copied());
        let on_add4 = database.on_add().with_key().with_type::<A>();
        assert_eq!(create.resolve(), i);
        assert!(on_add1.next().is_none());
        let mut add = database.add::<A>()?;
        add.all(keys.iter().copied());
        assert_eq!(add.resolve(), i);
        assert!((&mut on_add1).map(|e| e.key).eq(keys.iter().copied()));
        assert!(on_add4.map(|e| e.key).eq(keys.iter().copied()));
        assert!(database
            .on_add()
            .with_key()
            .with_type::<A>()
            .next()
            .is_none());

        if i % 13 == 0 {
            on_add3.clear();
            keys3.clear();
        }
        if i % 11 == 0 {
            assert!((&mut on_add3).map(|e| e.key).eq(keys3.drain(..)));
        }
        if i % 7 == 0 {
            on_add2.clear();
            keys2.clear();
        }
        if i % 3 == 0 {
            assert!((&mut on_add2).map(|e| e.key).eq(keys2.drain(..)));
        }

        assert_eq!(destroy.resolve(), i);
    }
    Ok(())
}
