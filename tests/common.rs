pub use checkito::{FullGenerate, Generate, any::Unify, prove, same::Same};
pub use std::{any::TypeId, collections::HashSet, error, marker::PhantomData, thread::scope};
pub use that_bass::{
    Database, Datum, Error, Filter, Template,
    create::Create,
    destroy::Destroy,
    filter::{Any, Filter, Has, Is, Not, has},
    key::Key,
    modify::{Add, Remove},
    query::{By, Query},
    row::Row,
    template::Template,
};

#[derive(Debug, Clone, Copy, Default, Datum)]
pub struct A;
#[derive(Debug, Clone, Copy, Default, Datum)]
pub struct B(pub usize);
#[derive(Debug, Clone, Copy, Default, Datum)]
pub struct C(pub f64);

pub const COUNT: usize = 37;

pub fn create_one(database: &Database, template: impl Template) -> Result<Key, Error> {
    let mut create = database.create()?;
    let key = create.one(template);
    assert_eq!(create.resolve(), 1);
    Ok(key)
}

pub fn create_n<const N: usize>(
    database: &Database,
    templates: [impl Template; N],
) -> Result<[Key; N], Error> {
    let mut create = database.create()?;
    let keys = create.all_n(templates);
    assert_eq!(create.resolve(), N);
    Ok(keys)
}

pub fn destroy_one(database: &Database, key: Key) -> Result<(), Error> {
    let mut destroy = database.destroy();
    destroy.one(key);
    assert_eq!(destroy.resolve(), 1);
    Ok(())
}

pub fn destroy_all(database: &Database, keys: &[Key]) {
    let mut destroy = database.destroy();
    destroy.all(keys.iter().copied());
    assert_eq!(destroy.resolve(), keys.len());
}
