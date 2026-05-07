use parking_lot::Mutex;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use that_bass::v2::{
    Builder, Callbacks, Configuration, Executor, FunctionContext, FunctionIndex, Key, Options,
    Store, command, query,
};

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Position {
    x: f32,
    y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Velocity {
    x: f32,
    y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Lifetime {
    frames_remaining: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Time {
    seconds: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
struct Spawner {
    count: u32,
}

// The current direct chunk-projection helpers still live at the lower-level schema layer.
type TableIndex = that_bass::v2::schema::TableIndex;
type ChunkIndex = that_bass::v2::schema::ChunkIndex;
type Table = that_bass::v2::schema::Table;

type DirectPositionQuery = query::All<(
    query::RowsRequest,
    query::Write<Position>,
    query::OptionQuery<query::Read<Velocity>>,
)>;

const TARGET_CHUNK_BYTE_COUNT: usize = 128;
const EXECUTOR_WORKER_COUNT: usize = 2;

fn main() {
    println!("that_bass v2 cheat sheet");
    println!();

    // Keep the chunk target small so the example naturally creates multiple chunks when it grows.
    let configuration = Configuration::default()
        .with_target_chunk_byte_count(non_zero_usize(TARGET_CHUNK_BYTE_COUNT));
    let mut store = Store::with_configuration(configuration);

    // Tables are schema-driven. The same logical concept can live in several table shapes.
    let position_only_table_index = store
        .register::<Position>()
        .expect("position-only table registration should succeed");
    let moving_table_index = store
        .register_row::<(Position, Velocity)>()
        .expect("moving-table registration should succeed");
    let keyed_table_index = store
        .ensure_row::<(Key, Position, Velocity, Lifetime)>()
        .expect("keyed table registration should succeed");

    // Globals are ordinary one-row tables rather than a separate resource world.
    store
        .initialize_global(Time { seconds: 0.016 })
        .expect("time global initialization should succeed");
    store
        .initialize_global(Spawner { count: 3 })
        .expect("spawner global initialization should succeed");

    seed_position_rows(
        &mut store,
        position_only_table_index,
        &[Position { x: 1.0, y: 1.0 }, Position { x: 2.0, y: 2.0 }],
    );
    seed_position_velocity_rows(
        &mut store,
        moving_table_index,
        &[
            (Position { x: 10.0, y: 10.0 }, Velocity { x: 0.5, y: 1.0 }),
            (Position { x: 20.0, y: 20.0 }, Velocity { x: -1.0, y: 0.25 }),
        ],
    );

    print_store_setup(
        &store,
        position_only_table_index,
        moving_table_index,
        keyed_table_index,
    );
    println!();

    demonstrate_direct_chunk_queries(&mut store, position_only_table_index, moving_table_index);
    println!();

    demonstrate_global_tables(&mut store);
    println!();

    let schedule_summary = run_executor_walkthrough(&mut store, keyed_table_index);
    println!();

    explain_why_it_matters(&store, keyed_table_index, &schedule_summary);
}

fn print_store_setup(
    store: &Store,
    position_only_table_index: TableIndex,
    moving_table_index: TableIndex,
    keyed_table_index: TableIndex,
) {
    let sample_row_width = core::mem::size_of::<(Key, Position, Velocity, Lifetime)>();
    let chunk_plan = store.plan_chunk_capacity_for_row_width(sample_row_width);

    println!("1. Setup and chunk planning");
    println!(
        "   target chunk bytes: {}",
        store.configuration().target_chunk_byte_count().get()
    );
    println!("   keyed row width: {sample_row_width} bytes");
    println!(
        "   keyed chunk capacity: {} rows",
        chunk_plan.target_chunk_capacity()
    );
    println!(
        "   registered tables: position_only={}, moving={}, keyed={}",
        position_only_table_index.value(),
        moving_table_index.value(),
        keyed_table_index.value()
    );
}

fn demonstrate_direct_chunk_queries(
    store: &mut Store,
    position_only_table_index: TableIndex,
    moving_table_index: TableIndex,
) {
    // This is the current low-level chunk-processing path. It is still useful because the
    // callback API is intentionally smaller than the query vocabulary for now.
    let integrate_query = direct_position_query();

    apply_chunk_query(store, position_only_table_index, &integrate_query);
    apply_chunk_query(store, moving_table_index, &integrate_query);

    println!("2. Direct chunk queries");
    println!("   one query shape matches multiple table schemas");
    println!(
        "   position-only rows: {:?}",
        collect_positions(
            store
                .table(position_only_table_index)
                .expect("position-only table should stay addressable")
        )
    );
    println!(
        "   position+velocity rows: {:?}",
        collect_positions(
            store
                .table(moving_table_index)
                .expect("moving table should stay addressable")
        )
    );
    println!("   `query::option(...)` lets one chunk stream stay regular across schema variants");
}

fn demonstrate_global_tables(store: &mut Store) {
    let before = *query::one::<Time>()
        .get(store)
        .expect("time global should be readable");
    query::one_mut::<Time>()
        .get_mut(store)
        .expect("time global should be writable")
        .seconds += 0.010;
    let after = *query::one::<Time>()
        .get(store)
        .expect("time global should stay readable");

    println!("3. Globals are just tables");
    println!("   time before update: {:.3}", before.seconds);
    println!("   time after update:  {:.3}", after.seconds);
    println!("   `query::one::<T>()` and `query::one_mut::<T>()` are the singleton path");
}

fn run_executor_walkthrough(store: &mut Store, keyed_table_index: TableIndex) -> ScheduleSummary {
    // The scheduled layer is where `v2` starts to differentiate itself:
    // work is declared in chunk terms, structural edits are deferred, and later functions can see
    // those edits in the same frame once resolve runs.
    //
    // The current callback surface is still lower-level than the query vocabulary. The direct
    // singleton reads above show today's ergonomic data access path; the schedule declaration
    // below shows how chunk streams, globals, command buffers, and keys compose at planning time.
    let integrate_query = direct_position_query();

    let mut builder = Builder::new(store);
    let spawn_index = builder
        .push("spawn-keyed", query::one::<Spawner>())
        .expect("spawner singleton input should be valid");
    builder
        .add_keys(spawn_index)
        .expect("key injection should succeed");
    let inserted_table_index = builder
        .add_insert(
            spawn_index,
            command::insert::<(Key, Position, Velocity, Lifetime)>(),
        )
        .expect("typed insert planning should succeed");

    let integrate_index = builder
        .push("integrate", (integrate_query, query::one::<Time>()))
        .expect("mixed stream and singleton inputs should be valid");

    let observe_index = builder.push_query(
        "observe-keyed",
        query::all(query::read::<Position>())
            .expect("observer query should be valid")
            .filter(query::has::<Key>()),
    );
    builder
        .add_keys(observe_index)
        .expect("observer key injection should succeed");

    let cull_index = builder.push_query(
        "cull-first-row",
        query::all(query::rows())
            .expect("remove query should be valid")
            .filter(query::has::<Key>()),
    );
    builder
        .add_remove(cull_index, command::remove(query::has::<Key>()))
        .expect("remove planning should succeed");

    let schedule = builder.build();
    debug_assert_eq!(inserted_table_index, keyed_table_index);

    let callbacks =
        CheatSheetCallbacks::new(spawn_index, integrate_index, observe_index, cull_index);
    let report = Executor::with_options(
        Options::default().with_worker_count(non_zero_usize(EXECUTOR_WORKER_COUNT)),
    )
    .run(&schedule, store, &callbacks);

    let reserved_keys = callbacks.reserved_keys();
    let observed_row_count = callbacks.observed_row_count();
    let integrated_row_count = callbacks.integrated_row_count();
    let removed_key = reserved_keys
        .first()
        .copied()
        .expect("spawn should reserve at least one key");
    let moved_key = *reserved_keys
        .last()
        .expect("spawn should reserve at least one key");

    println!("4. Scheduled execution and batched resolve");
    println!("   functions: {}", schedule.function_count());
    println!("   resolves:  {}", schedule.resolve_count());
    println!("   created jobs:  {}", report.created_job_count());
    println!("   injected jobs: {}", report.injected_job_count());
    println!("   observed keyed rows in the same frame: {observed_row_count}");
    println!("   integrated rows: {integrated_row_count}");
    println!("   same-frame visibility comes from deferred resolve plus happens-before edges");
    println!("   singleton inputs are declared separately from chunk streams");

    ScheduleSummary {
        reserved_keys,
        removed_key,
        moved_key,
    }
}

fn explain_why_it_matters(
    store: &Store,
    keyed_table_index: TableIndex,
    schedule_summary: &ScheduleSummary,
) {
    let keys = store
        .keys()
        .expect("key injection should initialize the Keys resource");
    let keyed_positions = query::all((
        query::read::<Key>(),
        query::read::<Position>(),
        query::option(query::read::<Velocity>()),
    ))
    .expect("keyed lookup query should be valid");

    let moved_lookup = keyed_positions
        .get(store, &keys, schedule_summary.moved_key)
        .expect("moved key should still resolve after swap_remove");
    let removed_lookup_failed = keyed_positions
        .get(store, &keys, schedule_summary.removed_key)
        .is_err();

    println!("5. Managed keys keep stable identity opt-in");
    println!("   reserved keys: {:?}", schedule_summary.reserved_keys);
    println!(
        "   removed key invalidated cleanly: {}",
        removed_lookup_failed
    );
    println!(
        "   moved key still resolves: key={:?}, position={:?}, velocity={:?}",
        moved_lookup.0, moved_lookup.1, moved_lookup.2
    );
    println!(
        "   keyed table rows after cull: {}",
        row_count_for_table(
            store
                .table(keyed_table_index)
                .expect("keyed table should stay addressable")
        )
    );
    println!("   rows stay densely packed, and stable identity only exists where you ask for it");
    println!();
    println!("Why this is interesting");
    println!("   - hot work is expressed over chunk slices instead of one row at a time");
    println!("   - structural edits are deferred and become visible later in the same frame");
    println!("   - globals are normal tables, not a separate subsystem");
    println!(
        "   - stable keys are layered on top, so tables that do not need identity stay cheaper"
    );
}

fn apply_chunk_query(store: &mut Store, table_index: TableIndex, query: &DirectPositionQuery) {
    let chunk_indices = store
        .table(table_index)
        .expect("registered table should stay addressable")
        .chunks()
        .iter()
        .filter(|chunk| chunk.count() != 0)
        .map(|chunk| chunk.chunk_index())
        .collect::<Vec<_>>();

    let table = store
        .table_mut(table_index)
        .expect("registered table should stay addressable");

    for chunk_index in chunk_indices {
        let (rows, positions, velocities) = query
            .project_chunk(table, chunk_index)
            .expect("chunk projection should succeed");

        for ((row, position), velocity) in rows.zip(positions).zip(velocities) {
            debug_assert_eq!(row.table_index(), table_index);
            match velocity {
                Some(velocity) => {
                    position.x += velocity.x;
                    position.y += velocity.y;
                }
                None => {
                    position.y += 1.0;
                }
            }
        }
    }
}

fn direct_position_query() -> DirectPositionQuery {
    query::all((
        query::rows(),
        query::write::<Position>(),
        query::option(query::read::<Velocity>()),
    ))
    .expect("direct position query should be valid")
}

fn seed_position_rows(store: &mut Store, table_index: TableIndex, rows: &[Position]) {
    let table = store
        .table_mut(table_index)
        .expect("registered table should stay addressable");
    seed_rows_across_chunks(table, rows, |table, chunk_index, row_index, row| unsafe {
        table
            .write::<Position>(chunk_index, row_index, *row)
            .expect("direct write should succeed");
    });
}

fn seed_position_velocity_rows(
    store: &mut Store,
    table_index: TableIndex,
    rows: &[(Position, Velocity)],
) {
    let table = store
        .table_mut(table_index)
        .expect("registered table should stay addressable");
    seed_rows_across_chunks(table, rows, |table, chunk_index, row_index, row| unsafe {
        table
            .write::<Position>(chunk_index, row_index, row.0)
            .expect("direct write should succeed");
        table
            .write::<Velocity>(chunk_index, row_index, row.1)
            .expect("direct write should succeed");
    });
}

fn seed_rows_across_chunks<Row, WriteRow>(table: &mut Table, rows: &[Row], mut write_row: WriteRow)
where
    WriteRow: FnMut(&mut Table, ChunkIndex, usize, &Row),
{
    let mut next_row_index = 0usize;

    while next_row_index < rows.len() {
        let chunk_index = table.push_chunk();
        let chunk_capacity = table
            .chunk(chunk_index)
            .expect("newly pushed chunk should stay addressable")
            .capacity();
        let chunk_row_count = (rows.len() - next_row_index).min(chunk_capacity);

        for row_offset in 0..chunk_row_count {
            write_row(
                table,
                chunk_index,
                row_offset,
                &rows[next_row_index + row_offset],
            );
        }

        unsafe {
            table
                .assume_initialized_prefix(chunk_index, chunk_row_count)
                .expect("initialized prefix declaration should succeed");
        }

        next_row_index += chunk_row_count;
    }
}

fn collect_positions(table: &Table) -> Vec<(f32, f32)> {
    let mut positions = Vec::new();

    for chunk in table.chunks() {
        if chunk.count() == 0 {
            continue;
        }

        positions.extend(
            table
                .slice::<Position>(chunk.chunk_index())
                .expect("position slices should stay addressable")
                .iter()
                .map(|position| (position.x, position.y)),
        );
    }

    positions
}

fn row_count_for_table(table: &Table) -> usize {
    table.chunks().iter().map(|chunk| chunk.count()).sum()
}

#[derive(Debug, Clone)]
struct ScheduleSummary {
    reserved_keys: Vec<Key>,
    removed_key: Key,
    moved_key: Key,
}

struct CheatSheetCallbacks {
    spawn_index: FunctionIndex,
    integrate_index: FunctionIndex,
    observe_index: FunctionIndex,
    cull_index: FunctionIndex,
    reserved_keys: Mutex<Vec<Key>>,
    observed_row_count: AtomicUsize,
    integrated_row_count: AtomicUsize,
}

impl CheatSheetCallbacks {
    fn new(
        spawn_index: FunctionIndex,
        integrate_index: FunctionIndex,
        observe_index: FunctionIndex,
        cull_index: FunctionIndex,
    ) -> Self {
        Self {
            spawn_index,
            integrate_index,
            observe_index,
            cull_index,
            reserved_keys: Mutex::new(Vec::new()),
            observed_row_count: AtomicUsize::new(0),
            integrated_row_count: AtomicUsize::new(0),
        }
    }

    fn reserved_keys(&self) -> Vec<Key> {
        self.reserved_keys.lock().clone()
    }

    fn observed_row_count(&self) -> usize {
        self.observed_row_count.load(Ordering::Relaxed)
    }

    fn integrated_row_count(&self) -> usize {
        self.integrated_row_count.load(Ordering::Relaxed)
    }
}

impl Callbacks for CheatSheetCallbacks {
    fn run_function(&self, mut context: FunctionContext<'_, '_>) {
        if context.function_index() == self.spawn_index {
            let reserved_keys = {
                let keys = context.keys().expect("spawn should receive Keys");
                [keys.reserve(), keys.reserve(), keys.reserve()]
            };
            let mut insert = context
                .insert::<(Key, Position, Velocity, Lifetime)>()
                .expect("spawn should expose its insert buffer");

            insert.array([
                (
                    reserved_keys[0],
                    Position { x: 100.0, y: 0.0 },
                    Velocity { x: 1.0, y: 0.5 },
                    Lifetime {
                        frames_remaining: 3,
                    },
                ),
                (
                    reserved_keys[1],
                    Position { x: 200.0, y: 0.0 },
                    Velocity { x: -1.0, y: 1.5 },
                    Lifetime {
                        frames_remaining: 2,
                    },
                ),
                (
                    reserved_keys[2],
                    Position { x: 300.0, y: 0.0 },
                    Velocity { x: 0.25, y: 0.75 },
                    Lifetime {
                        frames_remaining: 1,
                    },
                ),
            ]);

            self.reserved_keys.lock().extend(reserved_keys);
        }

        if context.function_index() == self.integrate_index {
            self.integrated_row_count
                .fetch_add(context.rows().len(), Ordering::Relaxed);
        }

        if context.function_index() == self.observe_index {
            self.observed_row_count
                .fetch_add(context.rows().len(), Ordering::Relaxed);
        }

        if context.function_index() == self.cull_index {
            if let Some(first_row) = context.rows().first() {
                context
                    .remove()
                    .expect("cull should expose its remove buffer")
                    .one(first_row);
            }
        }
    }
}

fn non_zero_usize(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("example constants must be non-zero")
}
