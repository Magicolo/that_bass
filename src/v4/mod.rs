pub mod depend;
pub mod error;
pub mod filter;
pub mod insert;
pub mod item;
pub mod meta;
pub mod module;
pub mod query;
pub mod remove;
pub mod slice;
pub mod state;
pub mod table;
pub mod template;
pub mod utility;
pub mod vector;

use crate::v4::{
    depend::{Access, Dependency, Resource},
    module::Module,
    table::Tables,
};
use core::iter::once;
pub use error::Error;
pub use meta::Meta;
use std::collections::{HashMap, hash_map::Entry};
pub use table::{Row, Rows, Table};
pub use vector::Vector;

pub struct Store {
    tables: Tables,
}

impl Store {
    pub fn new() -> Self {
        Self {
            tables: Tables::new(),
        }
    }

    pub const fn tables(&self) -> &Tables {
        &self.tables
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v4::{insert::Insert, query::Query, remove::Remove, utility::IteratorExtension};
    use itertools::izip;
    use std::{
        sync::atomic::{AtomicU32, Ordering},
        thread::scope,
    };

    #[test]
    fn access() -> anyhow::Result<()> {
        fn wait_for(signal: &AtomicU32, threshold: u32) {
            loop {
                let value = signal.load(Ordering::Relaxed);
                if value < threshold {
                    atomic_wait::wait(signal, value);
                } else {
                    break;
                }
            }
        }

        let store = Store::new();
        let mut query1 = Query::builder()
            .read::<char>()
            .try_write::<String>()
            .build(&store)?;
        let mut query2 = Query::builder()
            .read::<char>()
            .write::<u32>()
            .not::<String>()
            .build(&store)?;
        let mut insert = Insert::builder().key().column::<char>().build(&store)?;
        let mut remove = Remove::builder().build(&store);
        {
            let signal = AtomicU32::new(0);
            scope(|scope| {
                let signal = &signal;
                scope.spawn(move || {
                    for (a, b) in query1.columns().into_flat() {
                        let Some(b) = b else { continue };
                        for (a, b) in izip!(a, b) {
                            b.push(*a);
                        }
                    }
                    signal.store(1, Ordering::Relaxed);
                    atomic_wait::wake_all(signal);
                });
                scope.spawn(move || {
                    for (a, b) in query2.columns().into_flat() {
                        for (a, b) in izip!(a, b) {
                            *b = *a as u32;
                        }
                    }
                    signal.store(1, Ordering::Relaxed);
                    atomic_wait::wake_all(signal);
                });
                scope.spawn(move || {
                    wait_for(&signal, 10);
                    insert.one(((), 'a'));
                    anyhow::Ok(())
                });
                scope.spawn(move || {
                    wait_for(&signal, 1);
                    remove.all();
                    anyhow::Ok(())
                });
                anyhow::Ok(())
            })?;
        }
        Ok(())
    }

    // #[test]
    // fn read_write_conflict() -> Result<(), Error> {
    //     let mut store = Store::new();
    //     let mut state = store.state(
    //         State::build()
    //             .push((Insert::build().column::<u8>(),))
    //             .push((query().read::<u8>().write::<u8>(),)),
    //     )?;
    //     let guard = state.guard();
    //     let mut guard = guard.next()?;
    //     assert!(matches!(guard.get(), Err(Error::ReadWriteConflict(_, _))));
    //     Ok(())
    // }
}
