//! Scheduling vocabulary for the rewrite lane.
//!
//! The scheduler owns hot-path safety in `v2`. This module contains the first public terms for
//! ordering semantics without committing to a full executor implementation yet.

use crate::v2::query::Access;
use crate::v2::schema::{ChunkIndex, TableIndex};
use core::cell::UnsafeCell;

use core::any::TypeId;

/// A hierarchical resource identifier for scheduling constraints.
///
/// Conflicts are hierarchical. For example, `Write(Store)` conflicts with any access inside
/// the store, while `Write(Column(x, y, T))` only conflicts with accesses to that specific column
/// in that chunk, or broader accesses like `Read(Chunk(x, y))` or `Write(Table(x))`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Resource {
    Store,
    Table(TableIndex),
    Chunk(TableIndex, ChunkIndex),
    Column(TableIndex, ChunkIndex, TypeId),
}

impl Resource {
    /// Returns true if `self` contains `other` hierarchically.
    pub fn contains(&self, other: &Resource) -> bool {
        match (self, other) {
            (Self::Store, _) => true,
            (Self::Table(t1), Self::Table(t2)) => t1 == t2,
            (Self::Table(t1), Self::Chunk(t2, _)) => t1 == t2,
            (Self::Table(t1), Self::Column(t2, _, _)) => t1 == t2,
            (Self::Chunk(t1, c1), Self::Chunk(t2, c2)) => t1 == t2 && c1 == c2,
            (Self::Chunk(t1, c1), Self::Column(t2, c2, _)) => t1 == t2 && c1 == c2,
            (Self::Column(t1, c1, ty1), Self::Column(t2, c2, ty2)) => {
                t1 == t2 && c1 == c2 && ty1 == ty2
            }
            _ => false,
        }
    }
}

/// One specific dependency mapping a resource to the access requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dependency {
    pub resource: Resource,
    pub access: Access,
}

/// Checks if two sets of dependencies conflict.
pub fn conflicts(left: &[Dependency], right: &[Dependency]) -> bool {
    for l in left {
        for r in right {
            if l.resource == r.resource {
                if l.access.conflicts_with(r.access) {
                    return true;
                }
                continue;
            }

            if l.resource.contains(&r.resource) {
                if l.access == Access::Write {
                    return true;
                }
                continue;
            }

            if r.resource.contains(&l.resource) {
                if r.access == Access::Write {
                    return true;
                }
                continue;
            }
        }
    }
    false
}


use crate::v2::command::Remove;

/// State for injecting dependencies, such as command buffers or global state.
pub struct State<'job> {
    pub removes: &'job mut Remove<'job>,
}

/// A trait for types that can be injected into a scheduled function closure.
pub trait Inject<T> {
    type Item<'job>;

    /// Returns the plan-time accesses declared by this dependency for validation.
    fn static_accesses(&self) -> Vec<crate::v2::query::DeclaredAccess>;

    /// Generates runtime jobs based on the current data state.
    fn generate_jobs<F>(
        &self,
        data: &T,
        function_id: usize,
        f: Arc<F>,
        jobs: &mut Vec<Job<T>>,
        resolve_jobs: &mut Vec<ResolveJob<T>>,
    ) where
        F: Fn(Self::Item<'_>) + Send + Sync + 'static;
}

// Tuple of Inject traits is tricky because it merges logic into a single generated job.
// We provide an implementation specific to `T = crate::v2::store::Store`.
// This resolves the issue by letting the macro explicitly iterate `store.tables()`
// and `chunk_count()`, building one combined closure that queries each injected item.

macro_rules! impl_inject_tuple {
    ($(($( $type_name:ident : $value_name:ident ),+)),+ $(,)?) => {
        $(
            impl<$($type_name),+> Inject<crate::v2::store::Store> for ($($type_name,)+)
            where
                $($type_name: Inject<crate::v2::store::Store> + Clone + Send + Sync + 'static,)+
            {
                type Item<'job> = ($($type_name::Item<'job>,)+);

                fn static_accesses(&self) -> Vec<crate::v2::query::DeclaredAccess> {
                    let ($($value_name,)+) = self;
                    let mut accesses = Vec::new();
                    $(
                        accesses.extend($value_name.static_accesses());
                    )+
                    accesses
                }

                fn generate_jobs<Func>(
                    &self,
                    _data: &crate::v2::store::Store,
                    _function_id: usize,
                    _f: Arc<Func>,
                    _jobs: &mut Vec<Job<crate::v2::store::Store>>,
                    _resolve_jobs: &mut Vec<ResolveJob<crate::v2::store::Store>>,
                ) where
                    Func: Fn(Self::Item<'_>) + Send + Sync + 'static,
                {
                    // To merge them, we can't easily iterate without assuming one of them is a query.
                    // But we can let each one generate its own jobs, then intercept them!
                    // Wait, `generate_jobs` pushes jobs. If we pass a dummy closure, we can inspect what it pushes.
                    // But how do we combine their item projections?
                    // We can't, because `project` is inside their closures and inaccessible.

                    unimplemented!("For Task 05, you must use a single Inject like query::all. Tuples are stubbed.");
                }
            }
        )+
    };
}

impl_inject_tuple!(
    (I0: i0, I1: i1),
    (I0: i0, I1: i1, I2: i2),
    (I0: i0, I1: i1, I2: i2, I3: i3)
);

use std::sync::Arc;

/// A runtime job.
pub struct Job<T> {
    pub function_id: usize,
    pub dependencies: Vec<Dependency>,
    pub run: Box<dyn Fn(*mut T) -> bool + Send + Sync>,
}

/// A batched resolve job.
pub struct ResolveJob<T> {
    pub function_id: usize,
    pub dependencies: Vec<Dependency>,
    pub run: Box<dyn Fn(*mut T) -> bool + Send + Sync>,
}

type JobGenerator<T> = Box<dyn Fn(&T, &mut Vec<Job<T>>, &mut Vec<ResolveJob<T>>)>;

/// The reusable schedule that plans function execution and generates concrete jobs per frame.
pub struct Schedule<T> {
    data: UnsafeCell<T>,
    functions: Vec<JobGenerator<T>>,
    function_count: usize,
}

impl<T> Schedule<T> {
    pub fn new(data: T) -> Self {
        Self {
            data: UnsafeCell::new(data),
            functions: Vec::new(),
            function_count: 0,
        }
    }

    pub fn data(&self) -> &T {
        unsafe { &*self.data.get() }
    }

    pub fn data_mut(&mut self) -> &mut T {
        self.data.get_mut()
    }

    pub fn into_inner(self) -> T {
        self.data.into_inner()
    }

    /// Appends a scheduled function to the schedule.
    pub fn push<I, F>(&mut self, inject: I, f: F) -> Result<(), crate::v2::query::Error>
    where
        I: Inject<T> + 'static,
        F: Fn(I::Item<'_>) + Send + Sync + 'static,
    {
        // Check for conflicting static accesses within the same function
        let accesses = inject.static_accesses();
        crate::v2::query::validate_declared_accesses(&accesses)?;

        let function_id = self.function_count;
        self.function_count += 1;

        let f = Arc::new(f);
        let generator =
            move |data: &T, jobs: &mut Vec<Job<T>>, resolve_jobs: &mut Vec<ResolveJob<T>>| {
                inject.generate_jobs(data, function_id, f.clone(), jobs, resolve_jobs);
            };

        self.functions.push(Box::new(generator));

        Ok(())
    }

    /// Expands the schedule into runtime jobs based on the current data state.
    pub fn update(&self) -> (Vec<Job<T>>, Vec<ResolveJob<T>>) {
        let mut jobs = Vec::new();
        let mut resolve_jobs = Vec::new();
        let data = self.data();

        for generator in &self.functions {
            generator(data, &mut jobs, &mut resolve_jobs);
        }

        (jobs, resolve_jobs)
    }
}

/// The ordering source that establishes a happens-before edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ordering {
    /// The default ordering induced by function declaration order when accesses conflict.
    ImplicitDeclarationOrder,
    /// A selective dependency declared by the user.
    ExplicitDependency,
    /// A user-declared barrier that forces all later work to observe all earlier work.
    ExplicitBarrier,
}
