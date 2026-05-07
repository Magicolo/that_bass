use checkito::Check;
use std::num::NonZeroUsize;
use that_bass::v2::{
    Configuration, Store,
    query::{self, Access, Error},
    schedule::{Builder, Error as ScheduleError},
    schema::{Meta, Resource, ResourceId},
    store::GlobalError,
};

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Position {
    x: f32,
    y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Time {
    seconds: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Gravity {
    y: f32,
}

#[test]
fn initialize_global_creates_one_row_and_query_one_reads_it() -> Result<(), String> {
    if cfg!(miri) {
        for seconds in [0.0f32, 1.0, 16.0, 32.5] {
            assert_global_roundtrip(seconds);
        }

        return Ok(());
    }

    let seconds_generator = 0u16..512u16;
    seconds_generator
        .check(|raw_seconds| assert_global_roundtrip(raw_seconds as f32 / 8.0))
        .map_or(Ok(()), |failure| Err(format!("{failure:?}")))
}

#[test]
fn initialize_global_overwrites_the_existing_singleton_row() {
    let mut store = Store::with_configuration(test_configuration());
    let initial_table_index = store
        .initialize_global(Time { seconds: 1.0 })
        .expect("initial global initialization should succeed");
    let replacement_table_index = store
        .initialize_global(Time { seconds: 2.5 })
        .expect("replacement global initialization should succeed");
    let global_time = query::one::<Time>()
        .get(&store)
        .expect("singleton query should read the replacement value");

    assert_eq!(initial_table_index, replacement_table_index);
    assert_eq!(*global_time, Time { seconds: 2.5 });
    assert_eq!(row_count_for_table(&store, replacement_table_index), 1);
}

#[test]
fn query_one_rejects_missing_or_ambiguous_matching_tables() {
    let mut store = Store::with_configuration(test_configuration());

    assert_eq!(
        query::one::<Time>().get(&store),
        Err(Error::MissingOneTable {
            type_name: core::any::type_name::<Time>(),
        })
    );

    store
        .register_table([Meta::of::<Time>()])
        .expect("table registration should succeed");
    store
        .register_table([Meta::of::<Time>(), Meta::of::<Gravity>()])
        .expect("table registration should succeed");

    assert_eq!(
        query::one::<Time>().table_index(&store),
        Err(Error::MultipleOneTables {
            type_name: core::any::type_name::<Time>(),
            table_indices: store
                .tables()
                .iter()
                .filter(|table| table.inline_meta_for::<Time>().is_some())
                .map(|table| table.index())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        })
    );
}

#[test]
fn query_one_rejects_tables_with_zero_or_many_rows() {
    let mut empty_store = Store::with_configuration(test_configuration());
    let empty_table_index = empty_store
        .register_table([Meta::of::<Time>()])
        .expect("table registration should succeed");

    assert_eq!(
        query::one::<Time>().get(&empty_store),
        Err(Error::InvalidOneRowCount {
            type_name: core::any::type_name::<Time>(),
            table_index: empty_table_index,
            count: 0,
        })
    );
    assert_eq!(
        query::one::<Time>().table_index(&empty_store),
        Err(Error::InvalidOneRowCount {
            type_name: core::any::type_name::<Time>(),
            table_index: empty_table_index,
            count: 0,
        })
    );

    let mut store = Store::with_configuration(test_configuration());
    let table_index = store
        .register_table([Meta::of::<Time>()])
        .expect("table registration should succeed");
    let table = store
        .table_mut(table_index)
        .expect("registered table should stay addressable");
    let chunk_index = push_chunk_with_minimum_capacity(table, 2);
    unsafe {
        table
            .write::<Time>(chunk_index, 0, Time { seconds: 1.0 })
            .expect("direct write should succeed");
        table
            .write::<Time>(chunk_index, 1, Time { seconds: 2.0 })
            .expect("direct write should succeed");
        table
            .assume_initialized_prefix(chunk_index, 2)
            .expect("initialized prefix declaration should succeed");
    }

    assert_eq!(
        query::one::<Time>().get(&store),
        Err(Error::InvalidOneRowCount {
            type_name: core::any::type_name::<Time>(),
            table_index,
            count: 2,
        })
    );
    assert_eq!(
        query::one::<Time>().table_index(&store),
        Err(Error::InvalidOneRowCount {
            type_name: core::any::type_name::<Time>(),
            table_index,
            count: 2,
        })
    );
}

#[test]
fn initialize_global_rejects_existing_tables_with_more_than_one_row() {
    let mut store = Store::with_configuration(test_configuration());
    let table_index = store
        .register_table([Meta::of::<Time>()])
        .expect("table registration should succeed");
    let table = store
        .table_mut(table_index)
        .expect("registered table should stay addressable");
    let chunk_index = push_chunk_with_minimum_capacity(table, 2);
    unsafe {
        table
            .write::<Time>(chunk_index, 0, Time { seconds: 1.0 })
            .expect("direct write should succeed");
        table
            .write::<Time>(chunk_index, 1, Time { seconds: 2.0 })
            .expect("direct write should succeed");
        table
            .assume_initialized_prefix(chunk_index, 2)
            .expect("initialized prefix declaration should succeed");
    }

    assert_eq!(
        store.initialize_global(Time { seconds: 3.0 }),
        Err(GlobalError::InvalidRowCount {
            table_index,
            count: 2,
        })
    );
}

#[test]
fn schedule_builder_accepts_stream_and_singleton_inputs_as_separate_injections() {
    let mut store = Store::with_configuration(test_configuration());
    let position_table_index = store
        .register_table([Meta::of::<Position>()])
        .expect("table registration should succeed");
    let time_table_index = store
        .initialize_global(Time { seconds: 0.5 })
        .expect("global initialization should succeed");
    let mut builder = Builder::new(&mut store);
    let function_index = builder
        .push(
            "integrate",
            (query::write::<Position>(), query::one::<Time>()),
        )
        .expect("mixed stream and singleton inputs should plan successfully");
    let schedule = builder.build();
    let function = schedule
        .function(function_index)
        .expect("scheduled function should exist");

    assert_eq!(function.known_tables(), &[position_table_index]);
    assert_eq!(
        function
            .globals()
            .iter()
            .map(|global| (global.table_index(), global.type_name(), global.access()))
            .collect::<Vec<_>>(),
        vec![(
            time_table_index,
            core::any::type_name::<Time>(),
            Access::Read
        )]
    );
    assert_eq!(
        function.static_job_dependencies(),
        &[that_bass::v2::schema::Dependency::read([
            Resource::store(Some(schedule.root_identifier())),
            Resource::table(Some(time_table_index.into())),
            Resource::chunk(None),
            Resource::column::<Time>(Some(ResourceId::new(0))),
        ])]
    );
}

#[test]
fn schedule_builder_rejects_singleton_inputs_with_invalid_cardinality() {
    let mut store = Store::with_configuration(test_configuration());
    let table_index = store
        .register_table([Meta::of::<Time>()])
        .expect("table registration should succeed");
    let mut builder = Builder::new(&mut store);

    assert_eq!(
        builder.push("tick", query::one::<Time>()),
        Err(ScheduleError::Query(Error::InvalidOneRowCount {
            type_name: core::any::type_name::<Time>(),
            table_index,
            count: 0,
        }))
    );
}

#[test]
fn singleton_conflicts_are_planned_like_other_table_accesses() {
    let mut store = Store::with_configuration(test_configuration());
    store
        .initialize_global(Time { seconds: 1.0 })
        .expect("global initialization should succeed");
    let mut builder = Builder::new(&mut store);
    let first_function_index = builder
        .push("read time", query::one::<Time>())
        .expect("singleton read should plan successfully");
    let second_function_index = builder
        .push("write time", query::one_mut::<Time>())
        .expect("singleton write should plan successfully");
    let schedule = builder.build();

    assert!(schedule.edges().iter().any(|edge| {
        edge.from()
            == that_bass::v2::schedule::Node::Resolve(
                schedule
                    .function(first_function_index)
                    .expect("first function should exist")
                    .resolve_index(),
            )
            && edge.to() == that_bass::v2::schedule::Node::Function(second_function_index)
    }));
}

fn assert_global_roundtrip(seconds: f32) {
    let mut store = Store::with_configuration(test_configuration());
    let table_index = store
        .initialize_global(Time { seconds })
        .expect("global initialization should succeed");
    let global_time = query::one::<Time>()
        .get(&store)
        .expect("singleton query should read the initialized value");

    assert_eq!(*global_time, Time { seconds });
    assert_eq!(query::one::<Time>().table_index(&store), Ok(table_index));
    assert_eq!(row_count_for_table(&store, table_index), 1);
}

fn row_count_for_table(store: &Store, table_index: that_bass::v2::schema::TableIndex) -> usize {
    store
        .table(table_index)
        .expect("registered table should stay addressable")
        .chunks()
        .iter()
        .map(|chunk| chunk.count())
        .sum()
}

fn push_chunk_with_minimum_capacity(
    table: &mut that_bass::v2::schema::Table,
    minimum_capacity: usize,
) -> that_bass::v2::schema::ChunkIndex {
    loop {
        let chunk_index = table.push_chunk();
        let chunk = table
            .chunk(chunk_index)
            .expect("newly pushed chunk must be addressable");
        if chunk.capacity() >= minimum_capacity {
            return chunk_index;
        }
    }
}

fn test_configuration() -> Configuration {
    Configuration::default().with_target_chunk_byte_count(
        NonZeroUsize::new(1024).expect("test constant must be non-zero"),
    )
}
