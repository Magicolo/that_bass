use checkito::Check;
use std::num::NonZeroUsize;
use that_bass::v2::{
    command, query,
    schedule::{self, Builder, Edge, Node, Ordering, Reason},
    schema::{Catalog, Dependency, Meta, Resource, ResourceId, Table},
    Configuration,
};

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Position {
    x: f64,
    y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Velocity {
    x: f64,
    y: f64,
}

#[test]
fn pure_reads_do_not_conflict() {
    let left_dependency = Dependency::read([
        Resource::store(Some(ResourceId::new(0))),
        Resource::table(Some(ResourceId::new(1))),
        Resource::chunk(Some(ResourceId::new(2))),
        Resource::column::<Position>(Some(ResourceId::new(3))),
    ]);
    let right_dependency = Dependency::read([
        Resource::store(Some(ResourceId::new(0))),
        Resource::table(Some(ResourceId::new(1))),
        Resource::chunk(Some(ResourceId::new(2))),
        Resource::column::<Position>(Some(ResourceId::new(3))),
    ]);

    assert!(!schedule::conflict(&left_dependency, &right_dependency));
}

#[test]
fn broad_writers_conflict_with_descendant_accesses_but_not_disjoint_chunks() {
    let chunk_write = Dependency::write([
        Resource::store(Some(ResourceId::new(0))),
        Resource::table(Some(ResourceId::new(1))),
        Resource::chunk(Some(ResourceId::new(2))),
    ]);
    let descendant_read = Dependency::read([
        Resource::store(Some(ResourceId::new(0))),
        Resource::table(Some(ResourceId::new(1))),
        Resource::chunk(Some(ResourceId::new(2))),
        Resource::column::<Position>(Some(ResourceId::new(3))),
    ]);
    let disjoint_chunk_read = Dependency::read([
        Resource::store(Some(ResourceId::new(0))),
        Resource::table(Some(ResourceId::new(1))),
        Resource::chunk(Some(ResourceId::new(9))),
        Resource::column::<Position>(Some(ResourceId::new(3))),
    ]);
    let store_write = Dependency::write([Resource::store(Some(ResourceId::new(0)))]);

    assert!(schedule::conflict(&chunk_write, &descendant_read));
    assert!(!schedule::conflict(&chunk_write, &disjoint_chunk_read));
    assert!(schedule::conflict(&store_write, &descendant_read));
}

#[test]
fn empty_path_barriers_follow_the_selected_conflict_rules() {
    let read_dependency = Dependency::read([
        Resource::store(Some(ResourceId::new(0))),
        Resource::table(Some(ResourceId::new(1))),
        Resource::chunk(Some(ResourceId::new(2))),
        Resource::column::<Position>(Some(ResourceId::new(3))),
    ]);

    assert!(schedule::conflict(&Dependency::barrier(), &read_dependency));
    assert!(schedule::conflict(
        &Dependency::read([]),
        &Dependency::barrier()
    ));
    assert!(!schedule::conflict(&Dependency::read([]), &read_dependency));
}

#[test]
fn covers_handles_wildcards_and_prefix_paths() {
    let general_dependency = Dependency::write([
        Resource::store(None),
        Resource::table(Some(ResourceId::new(1))),
    ]);
    let specific_dependency = Dependency::read([
        Resource::store(Some(ResourceId::new(0))),
        Resource::table(Some(ResourceId::new(1))),
        Resource::chunk(Some(ResourceId::new(2))),
        Resource::column::<Position>(Some(ResourceId::new(3))),
    ]);
    let mismatched_table_dependency = Dependency::read([
        Resource::store(Some(ResourceId::new(0))),
        Resource::table(Some(ResourceId::new(8))),
        Resource::chunk(Some(ResourceId::new(2))),
    ]);
    let read_general_dependency = Dependency::read([
        Resource::store(Some(ResourceId::new(0))),
        Resource::table(Some(ResourceId::new(1))),
    ]);

    assert!(schedule::covers(&general_dependency, &specific_dependency));
    assert!(!schedule::covers(
        &general_dependency,
        &mismatched_table_dependency
    ));
    assert!(!schedule::covers(
        &read_general_dependency,
        &Dependency::write([
            Resource::store(Some(ResourceId::new(0))),
            Resource::table(Some(ResourceId::new(1))),
            Resource::chunk(Some(ResourceId::new(2))),
        ])
    ));
}

#[test]
fn schedule_builder_caches_known_tables_for_matching_queries() {
    let mut catalog = Catalog::new();
    let mut tables = make_tables(
        &mut catalog,
        [
            [Meta::of::<Position>()].as_slice(),
            [Meta::of::<Velocity>()].as_slice(),
        ],
    );
    let position_table_index = tables[0].index();
    let mut builder = Builder::new(&mut catalog, &mut tables, test_configuration());
    let function_index = builder.push_query(
        "read position",
        query::all(query::read::<Position>()).expect("query declaration should succeed"),
    );
    let schedule = builder.build();

    assert_eq!(
        schedule
            .function(function_index)
            .expect("scheduled function should exist")
            .known_tables(),
        &[position_table_index]
    );
}

#[test]
fn insert_descriptor_exposes_one_known_target_table_before_first_execution() {
    let mut catalog = Catalog::new();
    let mut tables = make_tables(&mut catalog, [[Meta::of::<Position>()].as_slice()]);
    let mut builder = Builder::new(&mut catalog, &mut tables, test_configuration());
    let function_index = builder.push_query(
        "spawn",
        query::all(query::rows()).expect("query declaration should succeed"),
    );

    let target_table_index = builder
        .add_insert(
            function_index,
            command::Insert::<(Position, Velocity)>::new(),
        )
        .expect("typed insert should resolve to one known table");
    let schedule = builder.build();
    let function = schedule
        .function(function_index)
        .expect("scheduled function should exist");
    let resolve = schedule
        .resolve_for_function(function_index)
        .expect("paired resolve should exist");

    assert!(function.known_tables().contains(&target_table_index));
    assert_eq!(function.command_kinds(), &[command::Kind::Insert]);
    assert_eq!(
        resolve.command_plans(),
        &[command::Plan::Insert {
            table_index: target_table_index
        }]
    );
    assert_eq!(
        resolve.dependencies(),
        &[Dependency::write([
            Resource::store(Some(schedule.root_identifier())),
            Resource::table(Some(target_table_index.into())),
        ])]
    );
}

#[test]
fn conflicting_functions_produce_resolve_to_function_edges() {
    let mut catalog = Catalog::new();
    let mut tables = make_tables(&mut catalog, [[Meta::of::<Position>()].as_slice()]);
    let mut builder = Builder::new(&mut catalog, &mut tables, test_configuration());
    let integrate_index = builder.push_query(
        "integrate",
        query::all(query::write::<Position>()).expect("query declaration should succeed"),
    );
    let clamp_index = builder.push_query(
        "clamp",
        query::all(query::write::<Position>()).expect("query declaration should succeed"),
    );
    let schedule = builder.build();

    assert!(schedule.edges().contains(&Edge::new(
        Node::Resolve(
            schedule
                .function(integrate_index)
                .expect("integrate function should exist")
                .resolve_index()
        ),
        Node::Function(clamp_index),
        Ordering::ImplicitDeclarationOrder,
        Reason::Conflict,
    )));
}

#[test]
fn conflicting_resolve_families_produce_resolve_to_resolve_edges() {
    let mut catalog = Catalog::new();
    let mut tables = make_tables(&mut catalog, [[Meta::of::<Position>()].as_slice()]);
    let mut builder = Builder::new(&mut catalog, &mut tables, test_configuration());
    let first_function_index = builder.push_query(
        "spawn first",
        query::all(query::rows()).expect("query declaration should succeed"),
    );
    builder
        .add_insert(first_function_index, command::Insert::<(Position,)>::new())
        .expect("typed insert should resolve to one known table");
    let second_function_index = builder.push_query(
        "spawn second",
        query::all(query::rows()).expect("query declaration should succeed"),
    );
    builder
        .add_insert(second_function_index, command::Insert::<(Position,)>::new())
        .expect("typed insert should resolve to one known table");
    let schedule = builder.build();

    assert!(schedule.edges().contains(&Edge::new(
        Node::Resolve(
            schedule
                .function(first_function_index)
                .expect("first function should exist")
                .resolve_index()
        ),
        Node::Resolve(
            schedule
                .function(second_function_index)
                .expect("second function should exist")
                .resolve_index()
        ),
        Ordering::ImplicitDeclarationOrder,
        Reason::Conflict,
    )));
}

#[test]
fn schedule_reuse_is_stable_after_chunk_count_changes() -> Result<(), String> {
    let additional_chunk_count_generator = 0usize..8usize;

    additional_chunk_count_generator
        .check(|additional_chunk_count| {
            let mut catalog = Catalog::new();
            let mut tables = make_tables(&mut catalog, [[Meta::of::<Position>()].as_slice()]);
            let table_index = tables[0].index();
            let mut builder = Builder::new(&mut catalog, &mut tables, test_configuration());
            let function_index = builder.push_query(
                "read position",
                query::all(query::read::<Position>()).expect("query declaration should succeed"),
            );
            let schedule = builder.build();
            let initial_edge_count = schedule.edge_count();

            for _ in 0..additional_chunk_count {
                tables[0].push_chunk();
            }

            assert_eq!(
                schedule
                    .function(function_index)
                    .expect("scheduled function should exist")
                    .known_tables(),
                &[table_index]
            );
            assert_eq!(schedule.edge_count(), initial_edge_count);
        })
        .map_or(Ok(()), |failure| Err(format!("{failure:?}")))
}

fn make_tables<const TABLE_COUNT: usize>(
    catalog: &mut Catalog,
    meta_groups: [&[Meta]; TABLE_COUNT],
) -> Vec<Table> {
    meta_groups
        .into_iter()
        .map(|metas| {
            catalog
                .register_table(metas.iter().copied(), test_configuration())
                .expect("table registration should succeed")
        })
        .collect()
}

fn test_configuration() -> Configuration {
    Configuration::default().with_target_chunk_byte_count(
        NonZeroUsize::new(1024).expect("test constant must be non-zero"),
    )
}
