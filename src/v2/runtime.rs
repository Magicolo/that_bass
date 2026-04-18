//! Executor runtime for the rewrite lane.
//!
//! Task 06 consumes the reusable family-level schedule from `schedule` and turns it into a
//! frame-local runtime with:
//!
//! - per-chunk function jobs,
//! - one batched resolve job per function family,
//! - worker-local ready queues,
//! - work stealing,
//! - family-biased affinity,
//! - and resolve-driven same-frame injection of newly visible chunk jobs.
//!
//! The runtime is intentionally narrower than a full store executor for now:
//!
//! - it seeds one frame from the current `Store` snapshot,
//! - it runs generic function and resolve callbacks,
//! - and batched resolve phases mutate the store directly and report the resulting chunk states.

use crate::v2::{
    command, key,
    query::Access,
    schedule::{self, Function, FunctionIndex, Node, Resolve, ResolveIndex, Schedule},
    schema::{
        ChunkIndex, ColumnIndex, Dependency, Resource, ResourceId, RowLayout, Rows, Table,
        TableIndex,
    },
    store::Store,
};
use core::{any::TypeId, num::NonZeroUsize};
use parking_lot::{Mutex, RwLock};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    mem::take,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    thread,
    time::Instant,
};

/// Runtime configuration for one executor instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    worker_count: NonZeroUsize,
    record_trace: bool,
    injection: Injection,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            worker_count: thread::available_parallelism()
                .unwrap_or_else(|_| NonZeroUsize::new(1).expect("1 must be non-zero")),
            record_trace: false,
            injection: Injection::PreferProducer,
        }
    }
}

impl Options {
    pub const fn worker_count(self) -> NonZeroUsize {
        self.worker_count
    }

    pub const fn with_worker_count(mut self, worker_count: NonZeroUsize) -> Self {
        self.worker_count = worker_count;
        self
    }

    #[doc(hidden)]
    pub const fn record_trace(self) -> bool {
        self.record_trace
    }

    #[doc(hidden)]
    pub const fn with_record_trace(mut self, record_trace: bool) -> Self {
        self.record_trace = record_trace;
        self
    }

    #[doc(hidden)]
    pub const fn injection(self) -> Injection {
        self.injection
    }

    #[doc(hidden)]
    pub const fn with_injection(mut self, injection: Injection) -> Self {
        self.injection = injection;
        self
    }
}

/// Placement policy for newly injected ready jobs.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Injection {
    PreferProducer,
    SharedFirst,
}

/// One visible chunk reported by resolve work.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VisibleChunk {
    table_index: TableIndex,
    chunk_index: ChunkIndex,
    row_count: usize,
}

impl VisibleChunk {
    pub const fn new(table_index: TableIndex, chunk_index: ChunkIndex, row_count: usize) -> Self {
        Self {
            table_index,
            chunk_index,
            row_count,
        }
    }

    pub const fn table_index(self) -> TableIndex {
        self.table_index
    }

    pub const fn chunk_index(self) -> ChunkIndex {
        self.chunk_index
    }

    pub const fn row_count(self) -> usize {
        self.row_count
    }
}

/// The structural visibility result of one resolve callback.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Outcome {
    visible_chunks: Box<[VisibleChunk]>,
}

impl Outcome {
    pub fn none() -> Self {
        Self {
            visible_chunks: Box::new([]),
        }
    }

    pub fn visible_chunks(visible_chunks: impl IntoIterator<Item = VisibleChunk>) -> Self {
        Self {
            visible_chunks: visible_chunks.into_iter().collect(),
        }
    }

    pub fn chunks(&self) -> &[VisibleChunk] {
        &self.visible_chunks
    }
}

/// The runnable callback surface consumed by the executor runtime.
pub trait Callbacks: Send + Sync + 'static {
    fn run_function(&self, context: FunctionContext<'_, '_>);

    fn run_resolve(&self, _context: ResolveContext<'_>) {}
}

/// Context for one per-chunk function job.
pub struct FunctionContext<'schedule, 'job> {
    function: &'schedule Function,
    table_index: TableIndex,
    chunk_index: ChunkIndex,
    worker_index: usize,
    rows: Rows<'job>,
    keys: Option<key::Keys>,
    command_buffer: &'job mut command::Buffer,
}

impl<'schedule, 'job> FunctionContext<'schedule, 'job> {
    pub const fn function(&self) -> &'schedule Function {
        self.function
    }

    pub const fn function_index(&self) -> FunctionIndex {
        self.function.index()
    }

    pub const fn table_index(&self) -> TableIndex {
        self.table_index
    }

    pub const fn chunk_index(&self) -> ChunkIndex {
        self.chunk_index
    }

    pub const fn worker_index(&self) -> usize {
        self.worker_index
    }

    pub const fn rows(&self) -> Rows<'job> {
        self.rows
    }

    pub fn keys(&self) -> Option<&key::Keys> {
        self.keys.as_ref()
    }

    pub fn insert<T>(&mut self) -> Option<command::InsertRows<'_, T>>
    where
        T: command::Columns,
    {
        self.command_buffer.insert::<T>()
    }

    pub fn remove(&mut self) -> Option<&mut command::RemoveRows> {
        self.command_buffer.remove()
    }
}

/// Context for one batched resolve job.
#[derive(Debug, Clone, Copy)]
pub struct ResolveContext<'schedule> {
    resolve: &'schedule Resolve,
    worker_index: usize,
}

impl<'schedule> ResolveContext<'schedule> {
    pub const fn resolve(&self) -> &'schedule Resolve {
        self.resolve
    }

    pub const fn resolve_index(&self) -> ResolveIndex {
        self.resolve.index()
    }

    pub const fn worker_index(&self) -> usize {
        self.worker_index
    }
}

/// One executor trace entry.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Trace {
    worker_index: usize,
    kind: TraceKind,
}

impl Trace {
    pub const fn worker_index(self) -> usize {
        self.worker_index
    }

    pub const fn kind(self) -> TraceKind {
        self.kind
    }
}

/// The kind of runtime work described in one trace entry.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceKind {
    Function {
        function_index: FunctionIndex,
        table_index: TableIndex,
        chunk_index: ChunkIndex,
    },
    Resolve {
        resolve_index: ResolveIndex,
    },
}

/// One execution report for a frame-local runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    worker_count: usize,
    created_job_count: usize,
    function_job_count: usize,
    resolve_job_count: usize,
    injected_job_count: usize,
    completed_job_count: usize,
    steal_count: usize,
    dependency_edge_count: usize,
    max_ready_job_count: usize,
    worker_execution_counts: Box<[usize]>,
    trace: Box<[Trace]>,
}

impl Report {
    pub const fn worker_count(&self) -> usize {
        self.worker_count
    }

    pub const fn created_job_count(&self) -> usize {
        self.created_job_count
    }

    pub const fn function_job_count(&self) -> usize {
        self.function_job_count
    }

    pub const fn resolve_job_count(&self) -> usize {
        self.resolve_job_count
    }

    pub const fn injected_job_count(&self) -> usize {
        self.injected_job_count
    }

    pub const fn completed_job_count(&self) -> usize {
        self.completed_job_count
    }

    pub const fn steal_count(&self) -> usize {
        self.steal_count
    }

    pub const fn dependency_edge_count(&self) -> usize {
        self.dependency_edge_count
    }

    pub const fn max_ready_job_count(&self) -> usize {
        self.max_ready_job_count
    }

    #[doc(hidden)]
    pub fn worker_execution_counts(&self) -> &[usize] {
        &self.worker_execution_counts
    }

    #[doc(hidden)]
    pub fn trace(&self) -> &[Trace] {
        &self.trace
    }
}

/// The runtime executor for one reusable schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Executor {
    options: Options,
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

impl Executor {
    pub fn new() -> Self {
        Self::with_options(Options::default())
    }

    pub const fn with_options(options: Options) -> Self {
        Self { options }
    }

    pub const fn options(self) -> Options {
        self.options
    }

    pub fn run<C>(&self, schedule: &Schedule, store: &mut Store, callbacks: &C) -> Report
    where
        C: Callbacks,
    {
        let seed = Seed::from_store(store);
        self.run_seeded(schedule, &seed, store, callbacks)
    }

    #[doc(hidden)]
    pub fn run_seeded<C>(
        &self,
        schedule: &Schedule,
        seed: &Seed,
        store: &mut Store,
        callbacks: &C,
    ) -> Report
    where
        C: Callbacks,
    {
        let worker_count = self.options.worker_count.get();
        let queues = Queues::new(worker_count);
        let shared = Arc::new(Shared::new(schedule, seed, store, self.options));
        let build = Build::new(schedule, seed, worker_count).seed_ready_jobs(&queues, &shared);

        thread::scope(|scope| {
            let handles = (0..worker_count)
                .map(|worker_index| {
                    let shared = Arc::clone(&shared);
                    let queues = &queues;

                    scope.spawn(move || {
                        worker_loop(worker_index, queues, &shared, callbacks);
                    })
                })
                .collect::<Vec<_>>();

            for handle in handles {
                handle
                    .join()
                    .expect("runtime worker thread should not panic");
            }
        });

        shared.finish_report(worker_count, build.resolve_job_count)
    }
}

/// A frame-local topology snapshot used to seed the runtime executor.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Seed {
    tables_by_index: BTreeMap<TableIndex, SeedTable>,
}

impl Seed {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_store(store: &Store) -> Self {
        Self::from_tables(store.tables())
    }

    pub fn from_tables(tables: &[Table]) -> Self {
        let tables_by_index = tables
            .iter()
            .map(|table| {
                let column_indices_by_identifier = table
                    .metas()
                    .iter()
                    .enumerate()
                    .map(|(column_offset, meta)| {
                        (
                            meta.identifier(),
                            ColumnIndex::new(
                                u16::try_from(column_offset)
                                    .expect("column count should fit in u16"),
                            ),
                        )
                    })
                    .collect::<BTreeMap<_, _>>();
                let chunk_indices = table
                    .chunks()
                    .iter()
                    .map(|chunk| chunk.chunk_index())
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                let chunk_row_counts = table
                    .chunks()
                    .iter()
                    .map(|chunk| (chunk.chunk_index(), chunk.count()))
                    .collect::<BTreeMap<_, _>>();

                (
                    table.index(),
                    SeedTable {
                        table_index: table.index(),
                        chunk_indices,
                        chunk_row_counts,
                        column_indices_by_identifier,
                        row_layout: table.row_layout(),
                    },
                )
            })
            .collect();

        Self { tables_by_index }
    }

    pub fn table_count(&self) -> usize {
        self.tables_by_index.len()
    }

    fn table(&self, table_index: TableIndex) -> Option<&SeedTable> {
        self.tables_by_index.get(&table_index)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SeedTable {
    table_index: TableIndex,
    chunk_indices: Box<[ChunkIndex]>,
    chunk_row_counts: BTreeMap<ChunkIndex, usize>,
    column_indices_by_identifier: BTreeMap<TypeId, ColumnIndex>,
    row_layout: RowLayout,
}

impl SeedTable {
    fn chunk_indices(&self) -> &[ChunkIndex] {
        &self.chunk_indices
    }

    fn dependency_for_identifier(
        &self,
        root_identifier: ResourceId,
        identifier: TypeId,
        access: Access,
        chunk_index: ChunkIndex,
    ) -> Option<Dependency> {
        let column_index = *self.column_indices_by_identifier.get(&identifier)?;

        Some(Dependency::new(
            access,
            [
                Resource::store(Some(root_identifier)),
                Resource::table(Some(self.table_index.into())),
                Resource::chunk(Some(chunk_index.into())),
                Resource::column_by_identifier(identifier, Some(column_index.into())),
            ],
        ))
    }

    fn row_layout(&self) -> RowLayout {
        self.row_layout
    }

    fn chunk_row_count(&self, chunk_index: ChunkIndex) -> Option<usize> {
        self.chunk_row_counts.get(&chunk_index).copied()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct JobIndex(usize);

impl JobIndex {
    const fn new(value: usize) -> Self {
        Self(value)
    }

    const fn value(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct FunctionChunkKey {
    function_index: FunctionIndex,
    table_index: TableIndex,
    chunk_index: ChunkIndex,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum JobKind {
    Function {
        function_index: FunctionIndex,
        table_index: TableIndex,
        chunk_index: ChunkIndex,
    },
    Resolve {
        resolve_index: ResolveIndex,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Job {
    kind: JobKind,
    dependencies: Box<[Dependency]>,
    affinity_worker_index: usize,
}

#[derive(Debug)]
struct JobRecord {
    job: Job,
    command_buffer: Mutex<command::Buffer>,
    pending_predecessor_count: AtomicUsize,
    successor_indices: Mutex<Vec<JobIndex>>,
    completed: AtomicBool,
}

impl JobRecord {
    fn new(job: Job, command_buffer: command::Buffer) -> Self {
        Self {
            job,
            command_buffer: Mutex::new(command_buffer),
            pending_predecessor_count: AtomicUsize::new(0),
            successor_indices: Mutex::new(Vec::new()),
            completed: AtomicBool::new(false),
        }
    }
}

struct Queues {
    local_queues: Box<[Mutex<VecDeque<JobIndex>>]>,
    shared_queue: Mutex<VecDeque<JobIndex>>,
}

impl Queues {
    fn new(worker_count: usize) -> Self {
        Self {
            local_queues: (0..worker_count)
                .map(|_| Mutex::new(VecDeque::new()))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            shared_queue: Mutex::new(VecDeque::new()),
        }
    }

    fn local_queue(&self, worker_index: usize) -> &Mutex<VecDeque<JobIndex>> {
        &self.local_queues[worker_index]
    }
}

struct Shared<'schedule, 'seed, 'store> {
    schedule: &'schedule Schedule,
    seed: &'seed Seed,
    store: Mutex<&'store mut Store>,
    keys: Option<key::Keys>,
    options: Options,
    jobs: RwLock<Vec<Arc<JobRecord>>>,
    function_chunk_keys: Mutex<BTreeSet<FunctionChunkKey>>,
    function_job_indices_by_function: RwLock<Vec<Vec<JobIndex>>>,
    chunk_row_counts: Mutex<BTreeMap<(TableIndex, ChunkIndex), usize>>,
    resolve_job_indices: RwLock<Vec<Option<JobIndex>>>,
    ready_job_count: AtomicUsize,
    max_ready_job_count: AtomicUsize,
    remaining_job_count: AtomicUsize,
    created_job_count: AtomicUsize,
    function_job_count: AtomicUsize,
    injected_job_count: AtomicUsize,
    completed_job_count: AtomicUsize,
    steal_count: AtomicUsize,
    dependency_edge_count: AtomicUsize,
    worker_execution_counts: Box<[AtomicUsize]>,
    trace: Mutex<Vec<Trace>>,
}

impl<'schedule, 'seed, 'store> Shared<'schedule, 'seed, 'store> {
    fn new(
        schedule: &'schedule Schedule,
        seed: &'seed Seed,
        store: &'store mut Store,
        options: Options,
    ) -> Self {
        let resolve_job_indices = (0..schedule.resolve_count())
            .map(|_| None)
            .collect::<Vec<_>>();
        let function_job_indices_by_function = vec![Vec::new(); schedule.function_count()];
        let worker_execution_counts = (0..options.worker_count.get())
            .map(|_| AtomicUsize::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let chunk_row_counts = seed
            .tables_by_index
            .iter()
            .flat_map(|(table_index, seed_table)| {
                seed_table
                    .chunk_row_counts
                    .iter()
                    .map(|(chunk_index, row_count)| ((*table_index, *chunk_index), *row_count))
                    .collect::<Vec<_>>()
            })
            .collect::<BTreeMap<_, _>>();

        Self {
            schedule,
            seed,
            keys: store.keys(),
            store: Mutex::new(store),
            options,
            jobs: RwLock::new(Vec::new()),
            function_chunk_keys: Mutex::new(BTreeSet::new()),
            function_job_indices_by_function: RwLock::new(function_job_indices_by_function),
            chunk_row_counts: Mutex::new(chunk_row_counts),
            resolve_job_indices: RwLock::new(resolve_job_indices),
            ready_job_count: AtomicUsize::new(0),
            max_ready_job_count: AtomicUsize::new(0),
            remaining_job_count: AtomicUsize::new(0),
            created_job_count: AtomicUsize::new(0),
            function_job_count: AtomicUsize::new(0),
            injected_job_count: AtomicUsize::new(0),
            completed_job_count: AtomicUsize::new(0),
            steal_count: AtomicUsize::new(0),
            dependency_edge_count: AtomicUsize::new(0),
            worker_execution_counts,
            trace: Mutex::new(Vec::new()),
        }
    }

    fn set_function_job_indices(&self, function_job_indices_by_function: Vec<Vec<JobIndex>>) {
        *self.function_job_indices_by_function.write() = function_job_indices_by_function;
    }

    fn append_job(&self, job: Job, command_buffer: command::Buffer) -> JobIndex {
        let mut jobs = self.jobs.write();
        let job_index = JobIndex::new(jobs.len());
        jobs.push(Arc::new(JobRecord::new(job, command_buffer)));
        self.created_job_count.fetch_add(1, Ordering::Relaxed);
        self.remaining_job_count.fetch_add(1, Ordering::Relaxed);

        job_index
    }

    fn job(&self, job_index: JobIndex) -> Arc<JobRecord> {
        self.jobs
            .read()
            .get(job_index.value())
            .cloned()
            .expect("runtime job index should stay addressable")
    }

    fn resolve_job_index(&self, resolve_index: ResolveIndex) -> JobIndex {
        self.resolve_job_indices.read()[resolve_index.value()]
            .expect("resolve job indices should be initialized before runtime edges are read")
    }

    fn set_resolve_job_index(&self, resolve_index: ResolveIndex, job_index: JobIndex) {
        self.resolve_job_indices.write()[resolve_index.value()] = Some(job_index);
    }

    fn rows_for_chunk<'job>(
        &self,
        table_index: TableIndex,
        chunk_index: ChunkIndex,
    ) -> Option<Rows<'job>> {
        let seed_table = self.seed.table(table_index)?;
        let row_count = self
            .chunk_row_counts
            .lock()
            .get(&(table_index, chunk_index))
            .copied()
            .or_else(|| seed_table.chunk_row_count(chunk_index))?;

        Some(Rows::generated(
            table_index,
            chunk_index,
            seed_table.row_layout(),
            row_count,
        ))
    }

    fn push_ready_local(&self, local_queue: &Mutex<VecDeque<JobIndex>>, job_index: JobIndex) {
        local_queue.lock().push_back(job_index);
        self.note_ready_push();
    }

    fn push_ready_shared(&self, shared_queue: &Mutex<VecDeque<JobIndex>>, job_index: JobIndex) {
        shared_queue.lock().push_back(job_index);
        self.note_ready_push();
    }

    fn note_ready_push(&self) {
        let ready_job_count = self.ready_job_count.fetch_add(1, Ordering::Relaxed) + 1;
        update_max(&self.max_ready_job_count, ready_job_count);
    }

    fn note_ready_pop(&self) {
        self.ready_job_count.fetch_sub(1, Ordering::Relaxed);
    }

    fn add_edge(&self, predecessor_index: JobIndex, successor_index: JobIndex) {
        let predecessor_record = self.job(predecessor_index);
        let mut successor_indices = predecessor_record.successor_indices.lock();
        if successor_indices.contains(&successor_index) {
            return;
        }

        successor_indices.push(successor_index);
        drop(successor_indices);
        self.job(successor_index)
            .pending_predecessor_count
            .fetch_add(1, Ordering::Relaxed);
        self.dependency_edge_count.fetch_add(1, Ordering::Relaxed);
    }

    fn mark_completed(
        &self,
        current_job_index: JobIndex,
        current_worker_index: usize,
        queues: &Queues,
    ) {
        let current_record = self.job(current_job_index);
        current_record.completed.store(true, Ordering::Release);
        self.worker_execution_counts[current_worker_index].fetch_add(1, Ordering::Relaxed);
        self.completed_job_count.fetch_add(1, Ordering::Relaxed);

        let successor_indices = current_record.successor_indices.lock().clone();
        for successor_index in successor_indices {
            let successor_record = self.job(successor_index);
            if successor_record
                .pending_predecessor_count
                .fetch_sub(1, Ordering::AcqRel)
                == 1
            {
                self.enqueue_ready_job(successor_index, current_worker_index, queues);
            }
        }

        self.remaining_job_count.fetch_sub(1, Ordering::AcqRel);
    }

    fn enqueue_ready_job(&self, job_index: JobIndex, current_worker_index: usize, queues: &Queues) {
        let job_record = self.job(job_index);
        match self.options.injection {
            Injection::PreferProducer
                if job_record.job.affinity_worker_index == current_worker_index =>
            {
                self.push_ready_local(queues.local_queue(current_worker_index), job_index);
            }
            Injection::PreferProducer => self.push_ready_shared(&queues.shared_queue, job_index),
            Injection::SharedFirst => self.push_ready_shared(&queues.shared_queue, job_index),
        }
    }

    fn record_trace(&self, trace: Trace) {
        if self.options.record_trace {
            self.trace.lock().push(trace);
        }
    }

    fn resolve_commands(
        &self,
        resolve_index: ResolveIndex,
    ) -> Result<Outcome, command::ResolveError> {
        let resolve = self
            .schedule
            .resolve(resolve_index)
            .expect("runtime resolve should exist");
        let function_job_indices = self.function_job_indices_by_function.read();
        let function_job_indices = function_job_indices
            .get(resolve.function_index().value())
            .expect("resolve function index should stay addressable");
        let mut batch = command::Batch::new(resolve.command_plans());

        for function_job_index in function_job_indices.iter().copied() {
            let job_record = self.job(function_job_index);
            let command_buffer = take(&mut *job_record.command_buffer.lock());
            batch.merge(command_buffer);
        }

        if batch.is_empty() {
            return Ok(Outcome::none());
        }

        let changed_chunks = {
            let mut store = self.store.lock();
            batch.resolve_on(*store)?
        };
        {
            let mut chunk_row_counts = self.chunk_row_counts.lock();
            for chunk_state in changed_chunks.iter().copied() {
                chunk_row_counts.insert(
                    (chunk_state.table_index(), chunk_state.chunk_index()),
                    chunk_state.row_count(),
                );
            }
        }

        Ok(Outcome::visible_chunks(
            changed_chunks.into_vec().into_iter().map(|chunk_state| {
                VisibleChunk::new(
                    chunk_state.table_index(),
                    chunk_state.chunk_index(),
                    chunk_state.row_count(),
                )
            }),
        ))
    }

    fn inject_from_resolve(
        &self,
        resolve_index: ResolveIndex,
        outcome: Outcome,
        current_worker_index: usize,
        queues: &Queues,
    ) {
        if outcome.chunks().is_empty() {
            return;
        }

        let resolve = self
            .schedule
            .resolve(resolve_index)
            .expect("runtime resolve should exist");
        let later_function_start = resolve.function_index().value() + 1;

        for visible_chunk in outcome.chunks().iter().copied() {
            let Some(seed_table) = self.seed.table(visible_chunk.table_index()) else {
                continue;
            };

            for function in &self.schedule.functions()[later_function_start..] {
                if !function
                    .known_tables()
                    .contains(&visible_chunk.table_index())
                {
                    continue;
                }

                let function_chunk_key = FunctionChunkKey {
                    function_index: function.index(),
                    table_index: visible_chunk.table_index(),
                    chunk_index: visible_chunk.chunk_index(),
                };
                let mut function_chunk_keys = self.function_chunk_keys.lock();
                if function_chunk_keys.contains(&function_chunk_key) {
                    continue;
                }

                let dependencies = function_job_dependencies(
                    self.schedule,
                    seed_table,
                    function.index(),
                    visible_chunk.chunk_index(),
                );
                if dependencies.is_empty()
                    && function
                        .query_analysis()
                        .is_some_and(|analysis| !analysis.declared_accesses().is_empty())
                {
                    continue;
                }

                function_chunk_keys.insert(function_chunk_key);
                drop(function_chunk_keys);

                let affinity_worker_index =
                    function.index().value() % self.options.worker_count.get();
                let resolve = self
                    .schedule
                    .resolve_for_function(function.index())
                    .expect("every scheduled function should have a paired resolve family");
                let job_index = self.append_job(
                    Job {
                        kind: JobKind::Function {
                            function_index: function.index(),
                            table_index: visible_chunk.table_index(),
                            chunk_index: visible_chunk.chunk_index(),
                        },
                        dependencies,
                        affinity_worker_index,
                    },
                    command::Buffer::new(resolve.command_plans()),
                );
                self.function_job_count.fetch_add(1, Ordering::Relaxed);
                self.injected_job_count.fetch_add(1, Ordering::Relaxed);
                self.function_job_indices_by_function.write()[function.index().value()]
                    .push(job_index);

                let paired_resolve_index = function.resolve_index();
                self.add_edge(job_index, self.resolve_job_index(paired_resolve_index));

                for edge in self.schedule.edges() {
                    let Node::Resolve(predecessor_resolve_index) = edge.from() else {
                        continue;
                    };
                    let Node::Function(successor_function_index) = edge.to() else {
                        continue;
                    };
                    if successor_function_index != function.index() {
                        continue;
                    }

                    let predecessor_job_index = self.resolve_job_index(predecessor_resolve_index);
                    let predecessor_record = self.job(predecessor_job_index);
                    if predecessor_record.completed.load(Ordering::Acquire) {
                        continue;
                    }

                    if schedule::conflicts_any(
                        predecessor_record.job.dependencies.as_ref(),
                        self.job(job_index).job.dependencies.as_ref(),
                    ) {
                        self.add_edge(predecessor_job_index, job_index);
                    }
                }

                let injected_record = self.job(job_index);
                if injected_record
                    .pending_predecessor_count
                    .load(Ordering::Acquire)
                    == 0
                {
                    self.enqueue_ready_job(job_index, current_worker_index, queues);
                }
            }
        }
    }

    fn finish_report(&self, worker_count: usize, resolve_job_count: usize) -> Report {
        Report {
            worker_count,
            created_job_count: self.created_job_count.load(Ordering::Relaxed),
            function_job_count: self.function_job_count.load(Ordering::Relaxed),
            resolve_job_count,
            injected_job_count: self.injected_job_count.load(Ordering::Relaxed),
            completed_job_count: self.completed_job_count.load(Ordering::Relaxed),
            steal_count: self.steal_count.load(Ordering::Relaxed),
            dependency_edge_count: self.dependency_edge_count.load(Ordering::Relaxed),
            max_ready_job_count: self.max_ready_job_count.load(Ordering::Relaxed),
            worker_execution_counts: self
                .worker_execution_counts
                .iter()
                .map(|count| count.load(Ordering::Relaxed))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            trace: self.trace.lock().clone().into_boxed_slice(),
        }
    }
}

struct Build<'schedule, 'seed, 'store, 'shared> {
    schedule: &'schedule Schedule,
    seed: &'seed Seed,
    worker_count: usize,
    shared: Option<&'shared Arc<Shared<'schedule, 'seed, 'store>>>,
    ready_job_indices: Vec<JobIndex>,
    function_job_indices_by_function: Vec<Vec<JobIndex>>,
    resolve_job_count: usize,
}

impl<'schedule, 'seed, 'store, 'shared> Build<'schedule, 'seed, 'store, 'shared> {
    fn new(schedule: &'schedule Schedule, seed: &'seed Seed, worker_count: usize) -> Self {
        Self {
            schedule,
            seed,
            worker_count,
            shared: None,
            ready_job_indices: Vec::new(),
            function_job_indices_by_function: vec![Vec::new(); schedule.function_count()],
            resolve_job_count: schedule.resolve_count(),
        }
    }

    fn seed_ready_jobs(
        mut self,
        queues: &Queues,
        shared: &'shared Arc<Shared<'schedule, 'seed, 'store>>,
    ) -> Self {
        self.shared = Some(shared);

        for function in self.schedule.functions() {
            self.populate_function_jobs(function);
        }

        for resolve in self.schedule.resolves() {
            self.populate_resolve_job(resolve);
        }

        self.add_initial_edges();
        self.shared()
            .set_function_job_indices(self.function_job_indices_by_function.clone());

        for job_index in self.ready_job_indices.iter().copied() {
            let job_record = shared.job(job_index);
            let affinity_worker_index = job_record.job.affinity_worker_index % self.worker_count;
            shared.push_ready_local(queues.local_queue(affinity_worker_index), job_index);
        }
        self
    }

    fn populate_function_jobs(&mut self, function: &Function) {
        for table_index in function.known_tables().iter().copied() {
            let Some(seed_table) = self.seed.table(table_index) else {
                continue;
            };

            for chunk_index in seed_table.chunk_indices().iter().copied() {
                let dependencies = function_job_dependencies(
                    self.schedule,
                    seed_table,
                    function.index(),
                    chunk_index,
                );
                if dependencies.is_empty()
                    && function
                        .query_analysis()
                        .is_some_and(|analysis| !analysis.declared_accesses().is_empty())
                {
                    continue;
                }

                let affinity_worker_index = function.index().value() % self.worker_count;
                let resolve = self
                    .schedule
                    .resolve_for_function(function.index())
                    .expect("every scheduled function should have a paired resolve family");
                let job_index = self.shared().append_job(
                    Job {
                        kind: JobKind::Function {
                            function_index: function.index(),
                            table_index,
                            chunk_index,
                        },
                        dependencies,
                        affinity_worker_index,
                    },
                    command::Buffer::new(resolve.command_plans()),
                );

                self.shared()
                    .function_chunk_keys
                    .lock()
                    .insert(FunctionChunkKey {
                        function_index: function.index(),
                        table_index,
                        chunk_index,
                    });
                self.shared()
                    .function_job_count
                    .fetch_add(1, Ordering::Relaxed);
                self.function_job_indices_by_function[function.index().value()].push(job_index);
            }
        }
    }

    fn populate_resolve_job(&mut self, resolve: &Resolve) {
        let job_index = self.shared().append_job(
            Job {
                kind: JobKind::Resolve {
                    resolve_index: resolve.index(),
                },
                dependencies: resolve.dependencies().to_vec().into_boxed_slice(),
                affinity_worker_index: resolve.function_index().value() % self.worker_count,
            },
            command::Buffer::default(),
        );

        self.shared()
            .set_resolve_job_index(resolve.index(), job_index);
    }

    fn add_initial_edges(&mut self) {
        for function in self.schedule.functions() {
            let resolve_job_index = self.shared().resolve_job_index(function.resolve_index());
            for function_job_index in
                &self.function_job_indices_by_function[function.index().value()]
            {
                self.shared()
                    .add_edge(*function_job_index, resolve_job_index);
            }
        }

        for edge in self.schedule.edges() {
            match (edge.from(), edge.to()) {
                (
                    Node::Resolve(predecessor_resolve_index),
                    Node::Function(successor_function_index),
                ) => {
                    let predecessor_job_index =
                        self.shared().resolve_job_index(predecessor_resolve_index);
                    let successor_resolve_job_index = self.shared().resolve_job_index(
                        self.schedule
                            .function(successor_function_index)
                            .expect("scheduled successor function should exist")
                            .resolve_index(),
                    );
                    let predecessor_record = self.shared().job(predecessor_job_index);

                    // A predecessor resolve may expose new chunks for the successor function. Even
                    // when the successor currently has no seeded jobs, its paired resolve must stay
                    // blocked until that predecessor has had a chance to inject them.
                    self.shared()
                        .add_edge(predecessor_job_index, successor_resolve_job_index);

                    for function_job_index in
                        &self.function_job_indices_by_function[successor_function_index.value()]
                    {
                        let function_job_record = self.shared().job(*function_job_index);
                        if schedule::conflicts_any(
                            predecessor_record.job.dependencies.as_ref(),
                            function_job_record.job.dependencies.as_ref(),
                        ) {
                            self.shared()
                                .add_edge(predecessor_job_index, *function_job_index);
                        }
                    }
                }
                (
                    Node::Resolve(predecessor_resolve_index),
                    Node::Resolve(successor_resolve_index),
                ) => {
                    let predecessor_job_index =
                        self.shared().resolve_job_index(predecessor_resolve_index);
                    let successor_job_index =
                        self.shared().resolve_job_index(successor_resolve_index);
                    let predecessor_record = self.shared().job(predecessor_job_index);
                    let successor_record = self.shared().job(successor_job_index);

                    if schedule::conflicts_any(
                        predecessor_record.job.dependencies.as_ref(),
                        successor_record.job.dependencies.as_ref(),
                    ) {
                        self.shared()
                            .add_edge(predecessor_job_index, successor_job_index);
                    }
                }
                _ => {}
            }
        }

        let job_count = self.shared().jobs.read().len();
        for job_offset in 0..job_count {
            let job_index = JobIndex::new(job_offset);
            let job_record = self.shared().job(job_index);
            if job_record.pending_predecessor_count.load(Ordering::Acquire) == 0 {
                self.ready_job_indices.push(job_index);
            }
        }
    }

    fn shared(&self) -> &Arc<Shared<'schedule, 'seed, 'store>> {
        self.shared
            .as_ref()
            .expect("build should have a shared runtime before populating jobs")
    }
}

fn worker_loop<'schedule, 'seed, 'store, C>(
    worker_index: usize,
    queues: &Queues,
    shared: &Arc<Shared<'schedule, 'seed, 'store>>,
    callbacks: &C,
) where
    C: Callbacks,
{
    loop {
        let Some(job_index) = pop_ready_job(worker_index, queues, shared) else {
            if shared.remaining_job_count.load(Ordering::Acquire) == 0 {
                return;
            }

            thread::yield_now();
            continue;
        };

        let job_record = shared.job(job_index);
        match &job_record.job.kind {
            JobKind::Function {
                function_index,
                table_index,
                chunk_index,
            } => {
                let function = shared
                    .schedule
                    .function(*function_index)
                    .expect("runtime function should exist");
                let rows = shared
                    .rows_for_chunk(*table_index, *chunk_index)
                    .expect("runtime function chunk should have a visible row state");
                let mut command_buffer = job_record.command_buffer.lock();
                callbacks.run_function(FunctionContext {
                    function,
                    table_index: *table_index,
                    chunk_index: *chunk_index,
                    worker_index,
                    rows,
                    keys: function.uses_keys().then(|| {
                        shared
                            .keys
                            .clone()
                            .expect("functions that use keys should initialize the Keys resource")
                    }),
                    command_buffer: &mut command_buffer,
                });
                drop(command_buffer);
                shared.record_trace(Trace {
                    worker_index,
                    kind: TraceKind::Function {
                        function_index: *function_index,
                        table_index: *table_index,
                        chunk_index: *chunk_index,
                    },
                });
            }
            JobKind::Resolve { resolve_index } => {
                let resolve = shared
                    .schedule
                    .resolve(*resolve_index)
                    .expect("runtime resolve should exist");
                let started_at = Instant::now();
                let outcome = shared
                    .resolve_commands(*resolve_index)
                    .expect("resolve command application should succeed");
                callbacks.run_resolve(ResolveContext {
                    resolve,
                    worker_index,
                });
                let _elapsed = started_at.elapsed();
                shared.record_trace(Trace {
                    worker_index,
                    kind: TraceKind::Resolve {
                        resolve_index: *resolve_index,
                    },
                });
                shared.inject_from_resolve(*resolve_index, outcome, worker_index, queues);
            }
        }

        shared.mark_completed(job_index, worker_index, queues);
    }
}

fn pop_ready_job<'schedule, 'seed, 'store>(
    worker_index: usize,
    queues: &Queues,
    shared: &Shared<'schedule, 'seed, 'store>,
) -> Option<JobIndex> {
    if let Some(job_index) = queues.local_queue(worker_index).lock().pop_back() {
        shared.note_ready_pop();
        return Some(job_index);
    }

    if let Some(job_index) = queues.shared_queue.lock().pop_front() {
        shared.note_ready_pop();
        return Some(job_index);
    }

    for victim_index in 0..queues.local_queues.len() {
        if victim_index == worker_index {
            continue;
        }

        if let Some(job_index) = queues.local_queue(victim_index).lock().pop_front() {
            shared.note_ready_pop();
            shared.steal_count.fetch_add(1, Ordering::Relaxed);
            return Some(job_index);
        }
    }

    None
}

fn function_job_dependencies(
    schedule: &Schedule,
    seed_table: &SeedTable,
    function_index: FunctionIndex,
    chunk_index: ChunkIndex,
) -> Box<[Dependency]> {
    let function = schedule
        .function(function_index)
        .expect("function dependency projection should address an existing function");
    let mut dependencies = function.static_job_dependencies().to_vec();

    if let Some(query_analysis) = function.query_analysis() {
        for declared_access in query_analysis.declared_accesses() {
            let Some(dependency) = seed_table.dependency_for_identifier(
                schedule.root_identifier(),
                declared_access.identifier(),
                declared_access.access(),
                chunk_index,
            ) else {
                continue;
            };
            push_dependency_if_missing(&mut dependencies, dependency);
        }
    }

    dependencies.into_boxed_slice()
}

fn push_dependency_if_missing(dependencies: &mut Vec<Dependency>, dependency: Dependency) {
    if !dependencies.contains(&dependency) {
        dependencies.push(dependency);
    }
}

fn update_max(maximum: &AtomicUsize, value: usize) {
    let mut current = maximum.load(Ordering::Relaxed);
    while value > current {
        match maximum.compare_exchange(current, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}
