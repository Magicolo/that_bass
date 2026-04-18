//! Schedule building and happens-before planning for the rewrite lane.
//!
//! Task 05 turns the scheduler from a vocabulary placeholder into the first reusable plan type.
//! This module now owns:
//!
//! - function and resolve family descriptors,
//! - family-level happens-before edges,
//! - hierarchical conflict and coverage checks for monotone dependency paths,
//! - storage-aware planning helpers for chunk streams and singleton-table side inputs,
//! - and per-job static dependencies that survive runtime chunk expansion while keeping the
//!   reusable schedule core generic over dependency paths.

pub use crate::v2::query::Access;
use crate::v2::{
    command::{self, Initialize},
    key,
    query::{self, All, Analysis, Filter, QueryTuple},
    schema::{DefinitionError, Dependency, Resource, ResourceId, Table, TableIndex},
    store::Store,
};
use core::any::{type_name, TypeId};

/// The ordering source that establishes one happens-before edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ordering {
    /// The default ordering induced by function declaration order when accesses conflict.
    ImplicitDeclarationOrder,
    /// A selective dependency declared by the user.
    ExplicitDependency,
    /// A user-declared barrier that forces later work to observe all earlier work.
    ExplicitBarrier,
}

/// A stable index of one function family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FunctionIndex(usize);

impl FunctionIndex {
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    pub const fn value(self) -> usize {
        self.0
    }
}

/// A stable index of one resolve family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ResolveIndex(usize);

impl ResolveIndex {
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    pub const fn value(self) -> usize {
        self.0
    }
}

/// One scheduled function family in the reusable schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    index: FunctionIndex,
    label: Box<str>,
    dependencies: Box<[Dependency]>,
    static_job_dependencies: Box<[Dependency]>,
    known_tables: Box<[TableIndex]>,
    query_analysis: Option<Analysis>,
    globals: Box<[Global]>,
    uses_keys: bool,
    command_kinds: Box<[command::Kind]>,
    resolve_index: ResolveIndex,
}

impl Function {
    pub const fn index(&self) -> FunctionIndex {
        self.index
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn dependencies(&self) -> &[Dependency] {
        &self.dependencies
    }

    pub fn static_job_dependencies(&self) -> &[Dependency] {
        &self.static_job_dependencies
    }

    pub fn known_tables(&self) -> &[TableIndex] {
        &self.known_tables
    }

    pub fn query_analysis(&self) -> Option<&Analysis> {
        self.query_analysis.as_ref()
    }

    pub fn globals(&self) -> &[Global] {
        &self.globals
    }

    pub const fn uses_keys(&self) -> bool {
        self.uses_keys
    }

    pub fn command_kinds(&self) -> &[command::Kind] {
        &self.command_kinds
    }

    pub const fn resolve_index(&self) -> ResolveIndex {
        self.resolve_index
    }
}

/// One planned singleton-table side input attached to a function family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Global {
    table_index: TableIndex,
    identifier: TypeId,
    type_name: &'static str,
    access: Access,
}

impl Global {
    pub const fn new(
        table_index: TableIndex,
        identifier: TypeId,
        type_name: &'static str,
        access: Access,
    ) -> Self {
        Self {
            table_index,
            identifier,
            type_name,
            access,
        }
    }

    pub const fn table_index(self) -> TableIndex {
        self.table_index
    }

    pub const fn identifier(self) -> TypeId {
        self.identifier
    }

    pub const fn type_name(self) -> &'static str {
        self.type_name
    }

    pub const fn access(self) -> Access {
        self.access
    }
}

/// One resolve family paired with one scheduled function family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolve {
    index: ResolveIndex,
    function_index: FunctionIndex,
    label: Box<str>,
    strategy: command::Strategy,
    dependencies: Box<[Dependency]>,
    command_plans: Box<[command::Plan]>,
    command_kinds: Box<[command::Kind]>,
}

impl Resolve {
    pub const fn index(&self) -> ResolveIndex {
        self.index
    }

    pub const fn function_index(&self) -> FunctionIndex {
        self.function_index
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub const fn strategy(&self) -> command::Strategy {
        self.strategy
    }

    pub fn dependencies(&self) -> &[Dependency] {
        &self.dependencies
    }

    pub fn command_plans(&self) -> &[command::Plan] {
        &self.command_plans
    }

    pub fn command_kinds(&self) -> &[command::Kind] {
        &self.command_kinds
    }
}

/// One node in the family-level schedule graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Node {
    Function(FunctionIndex),
    Resolve(ResolveIndex),
}

/// Why one schedule edge exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reason {
    Completion,
    Conflict,
}

/// One family-level happens-before edge in the reusable schedule graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Edge {
    from: Node,
    to: Node,
    ordering: Ordering,
    reason: Reason,
}

impl Edge {
    pub const fn new(from: Node, to: Node, ordering: Ordering, reason: Reason) -> Self {
        Self {
            from,
            to,
            ordering,
            reason,
        }
    }

    pub const fn from(self) -> Node {
        self.from
    }

    pub const fn to(self) -> Node {
        self.to
    }

    pub const fn ordering(self) -> Ordering {
        self.ordering
    }

    pub const fn reason(self) -> Reason {
        self.reason
    }
}

/// Schedule-builder failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    MissingFunction { function_index: FunctionIndex },
    MultipleStreamInputs,
    MissingFunctionInput,
    Query(query::Error),
    Definition(DefinitionError),
}

impl From<DefinitionError> for Error {
    fn from(error: DefinitionError) -> Self {
        Self::Definition(error)
    }
}

impl From<query::Error> for Error {
    fn from(error: query::Error) -> Self {
        Self::Query(error)
    }
}

/// A reusable family-level schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schedule {
    root_identifier: ResourceId,
    functions: Box<[Function]>,
    resolves: Box<[Resolve]>,
    edges: Box<[Edge]>,
}

impl Schedule {
    pub const fn root_identifier(&self) -> ResourceId {
        self.root_identifier
    }

    pub fn functions(&self) -> &[Function] {
        &self.functions
    }

    pub fn resolves(&self) -> &[Resolve] {
        &self.resolves
    }

    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    pub fn function_count(&self) -> usize {
        self.functions.len()
    }

    pub fn resolve_count(&self) -> usize {
        self.resolves.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn function(&self, function_index: FunctionIndex) -> Option<&Function> {
        self.functions.get(function_index.value())
    }

    pub fn resolve(&self, resolve_index: ResolveIndex) -> Option<&Resolve> {
        self.resolves.get(resolve_index.value())
    }

    pub fn resolve_for_function(&self, function_index: FunctionIndex) -> Option<&Resolve> {
        let function = self.function(function_index)?;
        self.resolve(function.resolve_index())
    }
}

enum DriverPlan {
    Stream {
        known_tables: Vec<TableIndex>,
        query_analysis: Analysis,
        matches_table: Box<dyn Fn(&Table) -> bool>,
    },
    Singleton {
        table_index: TableIndex,
    },
}

#[doc(hidden)]
pub struct FunctionInputs {
    driver: DriverPlan,
    dependencies: Vec<Dependency>,
    static_job_dependencies: Vec<Dependency>,
    globals: Vec<Global>,
}

#[doc(hidden)]
pub trait Inputs {
    fn into_function_inputs(
        self,
        store: &Store,
        root_identifier: ResourceId,
    ) -> Result<FunctionInputs, Error>;
}

trait StreamInput {
    fn into_stream_inputs(
        self,
        store: &Store,
        root_identifier: ResourceId,
    ) -> Result<FunctionInputs, Error>;
}

/// A builder for one reusable schedule.
pub struct Builder<'table> {
    root_identifier: ResourceId,
    store: &'table mut Store,
    functions: Vec<PendingFunction>,
}

impl<'table> Builder<'table> {
    pub fn new(store: &'table mut Store) -> Self {
        Self::with_root_identifier(store, ResourceId::new(0))
    }

    pub fn with_root_identifier(store: &'table mut Store, root_identifier: ResourceId) -> Self {
        Self {
            root_identifier,
            store,
            functions: Vec::new(),
        }
    }

    pub fn push<I>(&mut self, label: impl Into<Box<str>>, inputs: I) -> Result<FunctionIndex, Error>
    where
        I: Inputs,
    {
        let planned_inputs = inputs.into_function_inputs(self.store, self.root_identifier)?;
        Ok(self.push_planned_function(label.into(), planned_inputs))
    }

    pub fn push_query<Q, F>(
        &mut self,
        label: impl Into<Box<str>>,
        query: All<Q, F>,
    ) -> FunctionIndex
    where
        Q: QueryTuple + 'static,
        F: Filter + 'static,
    {
        let known_tables = self
            .store
            .tables()
            .iter()
            .filter(|table| query.matches_table(table))
            .map(Table::index)
            .collect::<Vec<_>>();
        let query_analysis = query.analysis().clone();
        let dependencies = collect_query_dependencies(
            self.store.tables(),
            self.root_identifier,
            &known_tables,
            &query_analysis,
        );

        self.push_planned_function(
            label.into(),
            FunctionInputs {
                driver: DriverPlan::Stream {
                    known_tables,
                    query_analysis,
                    matches_table: Box::new(move |table| query.matches_table(table)),
                },
                dependencies,
                static_job_dependencies: Vec::new(),
                globals: Vec::new(),
            },
        )
    }

    pub fn add_insert<T>(
        &mut self,
        function_index: FunctionIndex,
        insert: command::Insert<T>,
    ) -> Result<TableIndex, Error>
    where
        T: command::Columns,
    {
        if function_index.value() >= self.functions.len() {
            return Err(Error::MissingFunction { function_index });
        }

        let command_plan = insert.initialize(self.store)?;
        let PlanLike::Insert(target_table_index) = plan_like(&command_plan) else {
            unreachable!("insert initialization should always yield an insert plan");
        };
        let dependency = Dependency::write([
            Resource::store(Some(self.root_identifier)),
            Resource::table(Some(target_table_index.into())),
        ]);
        let target_table_uses_keys = self
            .store
            .table(target_table_index)
            .is_some_and(table_uses_keys);

        self.refresh_pending_queries_for_table(target_table_index);

        let function = self
            .functions
            .get_mut(function_index.value())
            .expect("validated function index should stay addressable");
        function.command_plans.push(command_plan);
        push_dependency_if_missing(&mut function.command_dependencies, dependency);
        if target_table_uses_keys {
            push_dependency_if_missing(
                &mut function.command_dependencies,
                key::Keys::write_dependency(self.root_identifier),
            );
        }

        Ok(target_table_index)
    }

    pub fn add_keys(&mut self, function_index: FunctionIndex) -> Result<(), Error> {
        if function_index.value() >= self.functions.len() {
            return Err(Error::MissingFunction { function_index });
        }

        self.store.initialize_keys();
        let function = self
            .functions
            .get_mut(function_index.value())
            .expect("validated function index should stay addressable");
        function.uses_keys = true;
        push_dependency_if_missing(
            &mut function.dependencies,
            key::Keys::read_dependency(self.root_identifier),
        );
        push_dependency_if_missing(
            &mut function.static_job_dependencies,
            key::Keys::read_dependency(self.root_identifier),
        );

        Ok(())
    }

    pub fn add_remove<F>(
        &mut self,
        function_index: FunctionIndex,
        remove: command::Remove<F>,
    ) -> Result<(), Error>
    where
        F: Filter,
    {
        if function_index.value() >= self.functions.len() {
            return Err(Error::MissingFunction { function_index });
        }

        let command_plan = remove.initialize(self.store)?;
        let PlanLike::Remove(allowed_table_indices) = plan_like(&command_plan) else {
            unreachable!("remove initialization should always yield a remove plan");
        };
        let allowed_table_indices = allowed_table_indices.to_vec();
        let keyed_table_indices = allowed_table_indices
            .iter()
            .copied()
            .filter(|table_index| self.store.table(*table_index).is_some_and(table_uses_keys))
            .collect::<Vec<_>>();
        let function = self
            .functions
            .get_mut(function_index.value())
            .expect("validated function index should stay addressable");
        function.command_plans.push(command_plan);
        for table_index in allowed_table_indices {
            push_dependency_if_missing(
                &mut function.command_dependencies,
                Dependency::write([
                    Resource::store(Some(self.root_identifier)),
                    Resource::table(Some(table_index.into())),
                ]),
            );
        }
        if !keyed_table_indices.is_empty() {
            push_dependency_if_missing(
                &mut function.command_dependencies,
                key::Keys::write_dependency(self.root_identifier),
            );
        }

        Ok(())
    }

    pub fn build(self) -> Schedule {
        let mut functions = Vec::with_capacity(self.functions.len());
        let mut resolves = Vec::with_capacity(self.functions.len());
        let mut edges = Vec::new();

        for (function_position, pending_function) in self.functions.into_iter().enumerate() {
            let function_index = FunctionIndex::new(function_position);
            let resolve_index = ResolveIndex::new(function_position);
            let resolve_dependencies = merge_dependencies(
                &pending_function.dependencies,
                &pending_function.command_dependencies,
            );
            let resolve_label = format!("{} resolve", pending_function.label).into_boxed_str();

            functions.push(Function {
                index: function_index,
                label: pending_function.label,
                dependencies: pending_function.dependencies.into_boxed_slice(),
                static_job_dependencies: pending_function
                    .static_job_dependencies
                    .into_boxed_slice(),
                known_tables: pending_function.known_tables.into_boxed_slice(),
                query_analysis: pending_function.query_analysis,
                globals: pending_function.globals.into_boxed_slice(),
                uses_keys: pending_function.uses_keys,
                command_kinds: pending_function
                    .command_plans
                    .iter()
                    .map(|plan| plan.kind())
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                resolve_index,
            });
            resolves.push(Resolve {
                index: resolve_index,
                function_index,
                label: resolve_label,
                strategy: command::Strategy::FunctionLevelBatch,
                dependencies: resolve_dependencies.into_boxed_slice(),
                command_plans: pending_function.command_plans.clone().into_boxed_slice(),
                command_kinds: pending_function
                    .command_plans
                    .into_iter()
                    .map(|plan| plan.kind())
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            });
            push_edge_if_missing(
                &mut edges,
                Edge::new(
                    Node::Function(function_index),
                    Node::Resolve(resolve_index),
                    Ordering::ImplicitDeclarationOrder,
                    Reason::Completion,
                ),
            );
        }

        for earlier_function_offset in 0..functions.len() {
            let earlier_resolve = &resolves[earlier_function_offset];

            for later_function_offset in (earlier_function_offset + 1)..functions.len() {
                let later_function = &functions[later_function_offset];
                let later_resolve = &resolves[later_function_offset];

                if conflicts_any(
                    earlier_resolve.dependencies(),
                    later_function.dependencies(),
                ) {
                    push_edge_if_missing(
                        &mut edges,
                        Edge::new(
                            Node::Resolve(earlier_resolve.index()),
                            Node::Function(later_function.index()),
                            Ordering::ImplicitDeclarationOrder,
                            Reason::Conflict,
                        ),
                    );
                }

                if conflicts_any(earlier_resolve.dependencies(), later_resolve.dependencies()) {
                    push_edge_if_missing(
                        &mut edges,
                        Edge::new(
                            Node::Resolve(earlier_resolve.index()),
                            Node::Resolve(later_resolve.index()),
                            Ordering::ImplicitDeclarationOrder,
                            Reason::Conflict,
                        ),
                    );
                }
            }
        }

        Schedule {
            root_identifier: self.root_identifier,
            functions: functions.into_boxed_slice(),
            resolves: resolves.into_boxed_slice(),
            edges: edges.into_boxed_slice(),
        }
    }

    fn push_planned_function(
        &mut self,
        label: Box<str>,
        planned_inputs: FunctionInputs,
    ) -> FunctionIndex {
        let (known_tables, query_analysis, matches_table) = match planned_inputs.driver {
            DriverPlan::Stream {
                known_tables,
                query_analysis,
                matches_table,
            } => (known_tables, Some(query_analysis), matches_table),
            DriverPlan::Singleton { table_index } => (
                vec![table_index],
                None,
                Box::new(never_matches_table) as Box<dyn Fn(&Table) -> bool>,
            ),
        };
        let function_index = FunctionIndex::new(self.functions.len());
        self.functions.push(PendingFunction {
            label,
            dependencies: planned_inputs.dependencies,
            static_job_dependencies: planned_inputs.static_job_dependencies,
            known_tables,
            query_analysis,
            globals: planned_inputs.globals,
            uses_keys: false,
            command_plans: Vec::new(),
            command_dependencies: Vec::new(),
            matches_table,
        });

        function_index
    }

    fn refresh_pending_queries_for_table(&mut self, table_index: TableIndex) {
        let target_table = self
            .store
            .table(table_index)
            .expect("refreshed target table should stay addressable");

        for pending_function in &mut self.functions {
            if pending_function.matches_table(target_table)
                && !pending_function.known_tables.contains(&table_index)
            {
                pending_function.known_tables.push(table_index);
                if let Some(query_analysis) = pending_function.query_analysis.as_ref() {
                    for declared_access in query_analysis.declared_accesses() {
                        let Some(column_access) = target_table.map_access_for_identifier(
                            declared_access.identifier(),
                            declared_access.access(),
                        ) else {
                            continue;
                        };

                        push_dependency_if_missing(
                            &mut pending_function.dependencies,
                            column_access.dependency_with_wildcard_chunk(self.root_identifier),
                        );
                    }
                }
            }
        }
    }
}

impl<Q, F> StreamInput for All<Q, F>
where
    Q: QueryTuple + 'static,
    F: Filter + 'static,
{
    fn into_stream_inputs(
        self,
        store: &Store,
        root_identifier: ResourceId,
    ) -> Result<FunctionInputs, Error> {
        let known_tables = store
            .tables()
            .iter()
            .filter(|table| self.matches_table(table))
            .map(Table::index)
            .collect::<Vec<_>>();
        let query_analysis = self.analysis().clone();
        let dependencies = collect_query_dependencies(
            store.tables(),
            root_identifier,
            &known_tables,
            &query_analysis,
        );

        Ok(FunctionInputs {
            driver: DriverPlan::Stream {
                known_tables,
                query_analysis,
                matches_table: Box::new(move |table| self.matches_table(table)),
            },
            dependencies,
            static_job_dependencies: Vec::new(),
            globals: Vec::new(),
        })
    }
}

impl<Q, F> Inputs for All<Q, F>
where
    Q: QueryTuple + 'static,
    F: Filter + 'static,
{
    fn into_function_inputs(
        self,
        store: &Store,
        root_identifier: ResourceId,
    ) -> Result<FunctionInputs, Error> {
        self.into_stream_inputs(store, root_identifier)
    }
}

impl<Q> StreamInput for Q
where
    Q: QueryTuple + 'static,
{
    fn into_stream_inputs(
        self,
        store: &Store,
        root_identifier: ResourceId,
    ) -> Result<FunctionInputs, Error> {
        query::all(self)
            .map_err(Error::from)?
            .into_stream_inputs(store, root_identifier)
    }
}

impl<Q> Inputs for Q
where
    Q: QueryTuple + 'static,
{
    fn into_function_inputs(
        self,
        store: &Store,
        root_identifier: ResourceId,
    ) -> Result<FunctionInputs, Error> {
        self.into_stream_inputs(store, root_identifier)
    }
}

impl<T: 'static> Inputs for query::One<T> {
    fn into_function_inputs(
        self,
        store: &Store,
        root_identifier: ResourceId,
    ) -> Result<FunctionInputs, Error> {
        let table_index = self.table_index(store).map_err(Error::from)?;
        let table = store
            .table(table_index)
            .expect("validated singleton table should stay addressable");
        let column_access = table
            .map_access::<T>(self.access())
            .expect("resolved global table should expose the requested inline column");
        let dependency = column_access.dependency_with_wildcard_chunk(root_identifier);
        let global = Global::new(
            table_index,
            TypeId::of::<T>(),
            type_name::<T>(),
            self.access(),
        );

        Ok(FunctionInputs {
            driver: DriverPlan::Singleton { table_index },
            dependencies: vec![dependency.clone()],
            static_job_dependencies: vec![dependency],
            globals: vec![global],
        })
    }
}

macro_rules! impl_stream_global_inputs {
    ($(($( $global_type_name:ident : $global_value_name:ident ),+)),+ $(,)?) => {
        $(
            impl<S, $($global_type_name),+> Inputs for (S, $(query::One<$global_type_name>,)+)
            where
                S: StreamInput,
                $($global_type_name: 'static,)+
            {
                fn into_function_inputs(
                    self,
                    store: &Store,
                    root_identifier: ResourceId,
                ) -> Result<FunctionInputs, Error> {
                    let (stream_input, $($global_value_name,)+) = self;
                    let mut planned_inputs = stream_input.into_stream_inputs(store, root_identifier)?;
                    $(
                        planned_inputs = combine_planned_inputs(
                            planned_inputs,
                            $global_value_name.into_function_inputs(store, root_identifier)?,
                        )?;
                    )+

                    Ok(planned_inputs)
                }
            }
        )+
    };
}

macro_rules! impl_global_only_inputs {
    ($(($( $global_type_name:ident : $global_value_name:ident ),+)),+ $(,)?) => {
        $(
            impl<$($global_type_name),+> Inputs for ($(query::One<$global_type_name>,)+)
            where
                $($global_type_name: 'static,)+
            {
                fn into_function_inputs(
                    self,
                    store: &Store,
                    root_identifier: ResourceId,
                ) -> Result<FunctionInputs, Error> {
                    let ($($global_value_name,)+) = self;
                    let mut planned_inputs = None;

                    $(
                        let next_inputs =
                            $global_value_name.into_function_inputs(store, root_identifier)?;
                        planned_inputs = Some(match planned_inputs {
                            Some(current_inputs) => combine_planned_inputs(current_inputs, next_inputs)?,
                            None => next_inputs,
                        });
                    )+

                    planned_inputs.ok_or(Error::MissingFunctionInput)
                }
            }
        )+
    };
}

impl_stream_global_inputs!(
    (G0: g0),
    (G0: g0, G1: g1),
    (G0: g0, G1: g1, G2: g2)
);

impl_global_only_inputs!(
    (G0: g0, G1: g1),
    (G0: g0, G1: g1, G2: g2),
    (G0: g0, G1: g1, G2: g2, G3: g3)
);

fn combine_planned_inputs(
    left_inputs: FunctionInputs,
    right_inputs: FunctionInputs,
) -> Result<FunctionInputs, Error> {
    let driver = match (left_inputs.driver, right_inputs.driver) {
        (DriverPlan::Stream { .. }, DriverPlan::Stream { .. }) => {
            return Err(Error::MultipleStreamInputs);
        }
        (driver @ DriverPlan::Stream { .. }, DriverPlan::Singleton { .. }) => driver,
        (DriverPlan::Singleton { .. }, driver @ DriverPlan::Stream { .. }) => driver,
        (driver @ DriverPlan::Singleton { .. }, DriverPlan::Singleton { .. }) => driver,
    };

    let mut dependencies = left_inputs.dependencies;
    for dependency in right_inputs.dependencies {
        push_dependency_if_missing(&mut dependencies, dependency);
    }
    let mut static_job_dependencies = left_inputs.static_job_dependencies;
    for dependency in right_inputs.static_job_dependencies {
        push_dependency_if_missing(&mut static_job_dependencies, dependency);
    }
    let mut globals = left_inputs.globals;
    globals.extend(right_inputs.globals);

    Ok(FunctionInputs {
        driver,
        dependencies,
        static_job_dependencies,
        globals,
    })
}

fn never_matches_table(_table: &Table) -> bool {
    false
}

enum PlanLike<'plan> {
    Insert(TableIndex),
    Remove(&'plan [TableIndex]),
    Set,
}

fn plan_like(plan: &command::Plan) -> PlanLike<'_> {
    match plan {
        command::Plan::Insert(plan) => PlanLike::Insert(plan.table_index()),
        command::Plan::Remove(plan) => PlanLike::Remove(plan.allowed_table_indices()),
        command::Plan::Set => PlanLike::Set,
    }
}

/// Returns whether two dependency paths conflict.
pub fn conflict(left_dependency: &Dependency, right_dependency: &Dependency) -> bool {
    if !accesses_conflict(left_dependency.access(), right_dependency.access()) {
        return false;
    }

    let shared_depth = left_dependency
        .path()
        .len()
        .min(right_dependency.path().len());
    for depth in 0..shared_depth {
        if !resources_may_alias(
            left_dependency.path()[depth],
            right_dependency.path()[depth],
        ) {
            return false;
        }
    }

    true
}

/// Returns whether one broader dependency covers a more specific dependency.
pub fn covers(general_dependency: &Dependency, specific_dependency: &Dependency) -> bool {
    if !access_covers(general_dependency.access(), specific_dependency.access()) {
        return false;
    }

    if general_dependency.path().len() > specific_dependency.path().len() {
        return false;
    }

    general_dependency
        .path()
        .iter()
        .zip(specific_dependency.path())
        .all(|(general_resource, specific_resource)| {
            general_resource.kind() == specific_resource.kind()
                && match (
                    general_resource.identifier(),
                    specific_resource.identifier(),
                ) {
                    (Some(general_identifier), Some(specific_identifier)) => {
                        general_identifier == specific_identifier
                    }
                    (None, _) => true,
                    (Some(_), None) => false,
                }
        })
}

pub fn conflicts_any(left_dependencies: &[Dependency], right_dependencies: &[Dependency]) -> bool {
    left_dependencies.iter().any(|left_dependency| {
        right_dependencies
            .iter()
            .any(|right_dependency| conflict(left_dependency, right_dependency))
    })
}

fn accesses_conflict(left_access: Access, right_access: Access) -> bool {
    matches!(
        (left_access, right_access),
        (Access::Write, Access::Read | Access::Write) | (Access::Read, Access::Write)
    )
}

fn access_covers(general_access: Access, specific_access: Access) -> bool {
    matches!(general_access, Access::Write) || matches!(specific_access, Access::Read)
}

fn resources_may_alias(left_resource: Resource, right_resource: Resource) -> bool {
    left_resource.kind() == right_resource.kind()
        && match (left_resource.identifier(), right_resource.identifier()) {
            (Some(left_identifier), Some(right_identifier)) => left_identifier == right_identifier,
            (None, _) | (_, None) => true,
        }
}

fn collect_query_dependencies(
    tables: &[Table],
    root_identifier: ResourceId,
    known_tables: &[TableIndex],
    analysis: &Analysis,
) -> Vec<Dependency> {
    let mut dependencies = Vec::new();

    for known_table_index in known_tables {
        let Some(table) = tables
            .iter()
            .find(|table| table.index() == *known_table_index)
        else {
            continue;
        };

        for declared_access in analysis.declared_accesses() {
            let Some(column_access) = table
                .map_access_for_identifier(declared_access.identifier(), declared_access.access())
            else {
                continue;
            };

            push_dependency_if_missing(
                &mut dependencies,
                column_access.dependency_with_wildcard_chunk(root_identifier),
            );
        }
    }

    dependencies
}

fn push_dependency_if_missing(dependencies: &mut Vec<Dependency>, dependency: Dependency) {
    if !dependencies.contains(&dependency) {
        dependencies.push(dependency);
    }
}

fn merge_dependencies(
    left_dependencies: &[Dependency],
    right_dependencies: &[Dependency],
) -> Vec<Dependency> {
    let mut merged_dependencies = left_dependencies.to_vec();

    for dependency in right_dependencies.iter().cloned() {
        push_dependency_if_missing(&mut merged_dependencies, dependency);
    }

    merged_dependencies
}

fn push_edge_if_missing(edges: &mut Vec<Edge>, edge: Edge) {
    if !edges.contains(&edge) {
        edges.push(edge);
    }
}

struct PendingFunction {
    label: Box<str>,
    dependencies: Vec<Dependency>,
    static_job_dependencies: Vec<Dependency>,
    known_tables: Vec<TableIndex>,
    query_analysis: Option<Analysis>,
    globals: Vec<Global>,
    uses_keys: bool,
    command_plans: Vec<command::Plan>,
    command_dependencies: Vec<Dependency>,
    matches_table: Box<dyn Fn(&Table) -> bool>,
}

impl PendingFunction {
    fn matches_table(&self, table: &Table) -> bool {
        (self.matches_table)(table)
    }
}

fn table_uses_keys(table: &Table) -> bool {
    table.inline_meta_for::<key::Key>().is_some()
}
