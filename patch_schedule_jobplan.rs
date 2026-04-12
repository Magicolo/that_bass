<<<<<<< SEARCH
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
=======
/// A recipe to construct a runtime job for a specific execution unit.
pub struct JobPlan<T, Item> {
    pub dependencies: Vec<Dependency>,
    pub project: Box<dyn Fn(*mut T) -> Item + Send + Sync>,
}

/// A recipe to construct a batched resolve job.
pub struct ResolveJobPlan<T> {
    pub dependencies: Vec<Dependency>,
    pub run: Box<dyn Fn(*mut T) -> bool + Send + Sync>,
}

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
>>>>>>> REPLACE
