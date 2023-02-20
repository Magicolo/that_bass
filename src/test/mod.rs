pub mod add;
pub mod create;
pub mod derive;
pub mod destroy;
pub mod event;
pub mod query;
pub mod remove;

use crate::{self as that_bass};
use std::{collections::HashSet, marker::PhantomData};
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
