pub mod depend;
pub mod error;
pub mod filter;
pub mod insert;
pub mod item;
pub mod meta;
pub mod query;
pub mod remove;
pub mod slice;
pub mod table;
pub mod template;
pub mod utility;
pub mod vector;

use crate::v4::table::Tables;
pub use error::Error;
pub use meta::Meta;
pub use table::{Row, Rows, Table};
pub use vector::Vector;

#[derive(Default)]
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
    use crate::v4::{depend::Analysis, insert::Insert, query::Query, remove::Remove};
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
            .has::<u8>()
            .build(&store)?;
        let mut query2 = Query::builder()
            .read::<char>()
            .write::<u32>()
            .not::<String>()
            .build(&store)?;
        let mut query3 = query1.clone();
        let mut query4 = query2.clone();
        let mut insert = Insert::builder().key().column::<char>().build(&store)?;
        let mut remove = Remove::builder().build(&store)?;
        {
            let signal = AtomicU32::new(0);
            scope(|scope| {
                let signal = &signal;
                scope.spawn(|| {
                    Analysis::new().add(&query3).add(&query4).analyze()?;
                    for mut outer in query3.tables() {
                        let (a, b) = outer.columns();
                        let Some(b) = b else { continue };

                        for mut inner in query4.tables() {
                            let (c, d) = inner.columns();
                            for (a, b, c, d) in izip!(&*a, &mut *b, &*c, &*d) {
                                b.push(*a);
                                b.push(*c);
                                b.extend(char::from_u32(*d));
                            }
                        }
                    }
                    signal.store(1, Ordering::Relaxed);
                    atomic_wait::wake_all(signal);
                    anyhow::Ok(())
                });
                scope.spawn(|| {
                    for mut guard in query1.tables() {
                        let (a, b) = guard.columns();
                        let Some(b) = b else { continue };
                        for (a, b) in izip!(a, b) {
                            b.push(*a);
                        }
                    }
                    signal.store(1, Ordering::Relaxed);
                    atomic_wait::wake_all(signal);
                });
                scope.spawn(move || {
                    for mut guard in query2.tables() {
                        let (a, b) = guard.columns();
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
