<<<<<<< SEARCH
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
// We provide a stub here so that signature bounds are met if needed, though most
// users will use `(query::All, Commands)`.

macro_rules! impl_inject_tuple {
    ($(($( $type_name:ident : $value_name:ident ),+)),+ $(,)?) => {
        $(
            impl<T, $($type_name),+> Inject<T> for ($($type_name,)+)
            where
                $($type_name: Inject<T>,)+
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
                    _data: &T,
                    _function_id: usize,
                    _f: Arc<Func>,
                    _jobs: &mut Vec<Job<T>>,
                    _resolve_jobs: &mut Vec<ResolveJob<T>>,
                ) where
                    Func: Fn(Self::Item<'_>) + Send + Sync + 'static,
                {
                    // This requires a zip-like executor builder that evaluates all items.
                    // For Task 05 MVP we just implement Inject for Query directly.
                    unimplemented!("Merging multiple Injectors into a single job is not yet fully implemented");
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
=======
/// A trait for types that can be injected into a scheduled function closure.
pub trait Inject<T> {
    type Item<'job>;

    /// Returns the plan-time accesses declared by this dependency for validation.
    fn static_accesses(&self) -> Vec<crate::v2::query::DeclaredAccess>;

    /// Indicates whether this injection targets specific tables.
    /// If false, it acts as a global or single-execution injection (like Commands).
    fn matches_table(&self, table: &crate::v2::schema::Table) -> bool {
        true
    }

    /// Indicates if this injection has chunk-level dependencies.
    /// If true, the scheduler will generate chunk-level jobs for it.
    fn has_chunk_execution(&self) -> bool {
        false
    }

    /// Returns the runtime dependencies for a specific chunk.
    fn dynamic_dependencies(
        &self,
        table: &crate::v2::schema::Table,
        chunk_index: crate::v2::schema::ChunkIndex,
    ) -> Vec<Dependency> {
        Vec::new()
    }

    /// Returns the dependencies for a structural batch resolve phase, if any.
    fn resolve_dependencies(&self) -> Vec<Dependency> {
        Vec::new()
    }

    /// Projects the injected view for a specific chunk or global execution.
    ///
    /// # Safety
    ///
    /// The caller must ensure that aliasing rules are respected.
    unsafe fn project<'job>(
        &self,
        data: *mut T,
        table_index: crate::v2::schema::TableIndex,
        chunk_index: crate::v2::schema::ChunkIndex,
    ) -> Result<Self::Item<'job>, crate::v2::query::Error>;
}

macro_rules! impl_inject_tuple {
    ($(($( $type_name:ident : $value_name:ident ),+)),+ $(,)?) => {
        $(
            impl<T, $($type_name),+> Inject<T> for ($($type_name,)+)
            where
                $($type_name: Inject<T>,)+
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

                fn matches_table(&self, table: &crate::v2::schema::Table) -> bool {
                    let ($($value_name,)+) = self;
                    true $(&& $value_name.matches_table(table))+
                }

                fn has_chunk_execution(&self) -> bool {
                    let ($($value_name,)+) = self;
                    false $(|| $value_name.has_chunk_execution())+
                }

                fn dynamic_dependencies(
                    &self,
                    table: &crate::v2::schema::Table,
                    chunk_index: crate::v2::schema::ChunkIndex,
                ) -> Vec<Dependency> {
                    let ($($value_name,)+) = self;
                    let mut deps = Vec::new();
                    $(
                        deps.extend($value_name.dynamic_dependencies(table, chunk_index));
                    )+

                    let mut unique_deps = Vec::new();
                    for dep in deps {
                        if !unique_deps.contains(&dep) {
                            unique_deps.push(dep);
                        }
                    }
                    unique_deps
                }

                fn resolve_dependencies(&self) -> Vec<Dependency> {
                    let ($($value_name,)+) = self;
                    let mut deps = Vec::new();
                    $(
                        deps.extend($value_name.resolve_dependencies());
                    )+

                    let mut unique_deps = Vec::new();
                    for dep in deps {
                        if !unique_deps.contains(&dep) {
                            unique_deps.push(dep);
                        }
                    }
                    unique_deps
                }

                unsafe fn project<'job>(
                    &self,
                    data: *mut T,
                    table_index: crate::v2::schema::TableIndex,
                    chunk_index: crate::v2::schema::ChunkIndex,
                ) -> Result<Self::Item<'job>, crate::v2::query::Error> {
                    let ($($value_name,)+) = self;
                    Ok((
                        $(
                            unsafe { $value_name.project(data, table_index, chunk_index)? },
                        )+
                    ))
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
>>>>>>> REPLACE
