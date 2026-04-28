use core::{
    any::TypeId,
    hash::Hash,
    iter::{empty, from_fn},
    marker::PhantomData,
    ops::{Deref, Range},
    ptr::addr_of,
};

pub struct Store {
    tables: RawVec<Table>,
}
pub struct Column {
    meta: &'static Meta,
    data: *mut u8,
}
pub struct Table {
    columns: RawSlice<Column>,
    count: usize,
    capacity: usize,
    // TODO: Perhaps use a `Range<u32>`
    free: RawVec<Range<usize>>,
}
pub struct Key {}

#[derive(Debug)]
pub struct Meta {
    identify: fn() -> TypeId,
}

impl Meta {
    pub fn identifier(&self) -> TypeId {
        (self.identify)()
    }
}

impl PartialEq for Meta {
    fn eq(&self, other: &Self) -> bool {
        self.identifier() == other.identifier()
    }
}

impl Eq for Meta {}

impl PartialOrd for Meta {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.identifier().partial_cmp(&other.identifier())
    }
}

impl Ord for Meta {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.identifier().cmp(&other.identifier())
    }
}

impl Hash for Meta {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.identifier().hash(state);
    }
}

pub trait Module: Sized {
    type State;
    type Item<'a>;

    fn initialize(self, store: &mut Store) -> Option<Self::State>;
    fn update(state: &mut Self::State, store: &mut Store) -> bool;
}

pub trait Filter {
    fn filter(&self, table: &Table) -> bool;
}

impl Filter for () {
    fn filter(&self, _: &Table) -> bool {
        true
    }
}

struct RawVec<T> {
    ptr: *mut T,
    len: usize,
    cap: usize,
    _marker: PhantomData<T>,
}

struct RawSlice<T> {
    ptr: *mut [T],
    _marker: PhantomData<T>,
}

impl Store {
    pub fn new() -> Self {
        Self {
            tables: RawVec::from_iter(empty()),
        }
    }

    pub const fn schedule(&mut self) -> schedule::Schedule<'_, ()> {
        schedule::Schedule::new(self)
    }
}

impl<T> RawVec<T> {
    pub const unsafe fn get_raw_unchecked(this: *const Self, index: usize) -> *const T {
        unsafe { (*this).ptr.add(index) }
    }

    pub const unsafe fn get_mut_raw_unchecked(this: *mut Self, index: usize) -> *mut T {
        unsafe { (*this).ptr.add(index) }
    }

    pub const unsafe fn get_raw(this: *const Self, index: usize) -> Option<*const T> {
        if index >= unsafe { (*this).len } {
            None
        } else {
            Some(unsafe { Self::get_raw_unchecked(this, index) })
        }
    }

    pub const unsafe fn get_mut_raw(this: *mut Self, index: usize) -> Option<*mut T> {
        if index >= unsafe { (*this).len } {
            None
        } else {
            Some(unsafe { Self::get_mut_raw_unchecked(this, index) })
        }
    }

    pub unsafe fn iter_raw(this: *const Self) -> impl ExactSizeIterator<Item = *const T> {
        (0..unsafe { (*this).len })
            .map(move |index| unsafe { Self::get_raw_unchecked(this, index) })
    }

    pub unsafe fn iter_mut_raw(this: *mut Self) -> impl ExactSizeIterator<Item = *mut T> {
        (0..unsafe { (*this).len })
            .map(move |index| unsafe { Self::get_mut_raw_unchecked(this, index) })
    }

    pub const unsafe fn get_unchecked(&self, index: usize) -> &T {
        unsafe { &*Self::get_raw_unchecked(self, index) }
    }

    pub const unsafe fn get_mut_unchecked(&mut self, index: usize) -> &mut T {
        unsafe { &mut *Self::get_mut_raw_unchecked(self, index) }
    }

    pub const fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            Some(unsafe { self.get_unchecked(index) })
        } else {
            None
        }
    }

    pub const fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            Some(unsafe { self.get_mut_unchecked(index) })
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &T> {
        unsafe { Self::iter_raw(self).map(|item| &*item) }
    }

    pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = &mut T> {
        unsafe { Self::iter_mut_raw(self).map(|item| &mut *item) }
    }

    pub unsafe fn push(&mut self, item: T) {
        let mut vec = unsafe { Vec::from_raw_parts(self.ptr, self.len, self.cap) };
        vec.push(item);
        let (ptr, len, cap) = vec.into_raw_parts();
        self.ptr = ptr;
        self.len = len;
        self.cap = cap;
    }
}

impl<T> RawSlice<T> {
    pub const unsafe fn get_raw_unchecked(this: *const Self, index: usize) -> *const T {
        unsafe { (*this).ptr.cast::<T>().add(index) }
    }

    pub const unsafe fn get_mut_raw_unchecked(this: *mut Self, index: usize) -> *mut T {
        unsafe { (*this).ptr.cast::<T>().add(index) }
    }

    pub const unsafe fn get_raw(this: *const Self, index: usize) -> Option<*const T> {
        if index >= unsafe { (*this).ptr.len() } {
            None
        } else {
            Some(unsafe { Self::get_raw_unchecked(this, index) })
        }
    }

    pub const unsafe fn get_mut_raw(this: *mut Self, index: usize) -> Option<*mut T> {
        if index >= unsafe { (*this).ptr.len() } {
            None
        } else {
            Some(unsafe { Self::get_mut_raw_unchecked(this, index) })
        }
    }

    pub unsafe fn iter_raw(this: *const Self) -> impl ExactSizeIterator<Item = *const T> {
        (0..unsafe { (*this).ptr.len() })
            .map(move |index| unsafe { Self::get_raw_unchecked(this, index) })
    }

    pub unsafe fn iter_mut_raw(this: *mut Self) -> impl ExactSizeIterator<Item = *mut T> {
        (0..unsafe { (*this).ptr.len() })
            .map(move |index| unsafe { Self::get_mut_raw_unchecked(this, index) })
    }

    pub const unsafe fn get_unchecked(&self, index: usize) -> &T {
        unsafe { &*Self::get_raw_unchecked(self, index) }
    }

    pub const unsafe fn get_mut_unchecked(&mut self, index: usize) -> &mut T {
        unsafe { &mut *Self::get_mut_raw_unchecked(self, index) }
    }

    pub const fn get(&self, index: usize) -> Option<&T> {
        if index < self.ptr.len() {
            Some(unsafe { Self::get_unchecked(self, index) })
        } else {
            None
        }
    }

    pub const fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.ptr.len() {
            Some(unsafe { Self::get_mut_unchecked(self, index) })
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &T> {
        unsafe { Self::iter_raw(self).map(|item| &*item) }
    }

    pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = &mut T> {
        unsafe { Self::iter_mut_raw(self).map(|item| &mut *item) }
    }
}

impl<T> FromIterator<T> for RawVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let (ptr, len, cap) = Vec::into_raw_parts(iter.into_iter().collect::<Vec<_>>());
        Self {
            ptr,
            len,
            cap,
            _marker: PhantomData,
        }
    }
}

impl<T> FromIterator<T> for RawSlice<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let ptr = Box::into_raw(iter.into_iter().collect::<Box<[_]>>());
        Self {
            ptr,
            _marker: PhantomData,
        }
    }
}

impl Table {
    pub fn fragments(&self) -> impl Iterator<Item = Range<usize>> + use<'_> {
        fragments_iter(
            self.free.iter().map(|range| (range.start, range.end)),
            self.count,
        )
    }

    unsafe fn fragments_raw(this: *const Self) -> impl Iterator<Item = Range<usize>> {
        let free = unsafe { RawVec::iter_raw(addr_of!((*this).free)) };
        fragments_iter(
            free.map(|range| (unsafe { (*range).start }, unsafe { (*range).end })),
            unsafe { (*this).count },
        )
    }
}

impl Meta {
    pub const fn of<T: 'static>() -> &'static Self {
        &Self {
            identify: TypeId::of::<T>,
        }
    }
}

fn fragments_iter(
    free: impl IntoIterator<Item = (usize, usize)>,
    count: usize,
) -> impl Iterator<Item = Range<usize>> {
    let mut last = 0;
    let mut free = free.into_iter();
    from_fn(move || match free.next() {
        Some((start, end)) => {
            let fragment = last..start;
            last = end.saturating_add(1);
            Some(fragment)
        }
        None if last < count => {
            let fragment = last..count;
            last = count;
            Some(fragment)
        }
        None => None,
    })
}

pub mod command {
    use super::*;

    pub trait Template {}
    pub struct Insert<T: Template>(T);
    pub struct Remove<F: Filter = ()>(F);
}

pub mod declare {
    use crate::v3::Meta;

    pub struct Context<'a> {
        nodes: &'a mut Vec<Node>,
    }

    pub(crate) enum Branch {
        Store,
        Tables,
        Table,
        Columns,
        Column,
        Fragments,
        Not,
    }

    pub(crate) enum Leaf {
        Read(&'static Meta),
        Write(&'static Meta),
        Has(&'static Meta),
    }

    pub(crate) enum Node {
        Branch(Branch, Vec<Node>),
        Leaf(Leaf),
    }

    pub trait Declare {
        fn declare<'a>(&self, context: Context<'a>);
    }

    impl Context<'_> {
        #[must_use]
        pub fn own(&mut self) -> Context<'_> {
            Context { nodes: self.nodes }
        }

        #[must_use]
        pub fn store(&mut self) -> Context<'_> {
            self.branch(Branch::Store)
        }

        #[must_use]
        pub fn tables(&mut self) -> Context<'_> {
            self.branch(Branch::Tables)
        }

        #[must_use]
        pub fn columns(&mut self) -> Context<'_> {
            self.branch(Branch::Columns)
        }

        #[must_use]
        pub fn not(&mut self) -> Context<'_> {
            self.branch(Branch::Not)
        }

        pub fn has<T: 'static>(self) {
            self.leaf(Leaf::Has(Meta::of::<T>()));
        }

        pub fn read<T: 'static>(self) {
            self.leaf(Leaf::Read(Meta::of::<T>()));
        }

        pub fn write<T: 'static>(self) {
            self.leaf(Leaf::Write(Meta::of::<T>()));
        }

        fn branch(&mut self, branch: Branch) -> Context<'_> {
            self.nodes.push(Node::Branch(branch, Vec::new()));
            match self.nodes.last_mut() {
                Some(Node::Branch(_, nodes)) => Context { nodes },
                _ => unreachable!(),
            }
        }

        fn leaf(self, leaf: Leaf) {
            self.nodes.push(Node::Leaf(leaf));
        }
    }

    impl Declare for () {
        fn declare<'a>(&self, _: Context<'a>) {}
    }

    impl<T: Declare> Declare for (T,) {
        fn declare<'a>(&self, context: Context<'a>) {
            self.0.declare(context);
        }
    }

    impl<L: Declare, R: Declare> Declare for (L, R) {
        fn declare<'a>(&self, mut context: Context<'a>) {
            self.0.declare(context.own());
            self.1.declare(context);
        }
    }

    pub(crate) fn declare<T: Declare>(declare: &T) -> Node {
        let mut nodes = Vec::new();
        declare.declare(Context { nodes: &mut nodes });
        Node::Branch(Branch::Store, nodes)
    }
}

pub mod prepare {
    pub struct Context<'a> {
        boba: &'a mut (),
    }

    pub trait Prepare {
        type Item<'a>;
        fn prepare<'a>(&self, context: Context<'a>) -> Self::Item<'a>;
    }

    impl<'a> Context<'a> {
        pub fn and(&self) -> Self {
            todo!()
        }

        pub fn table(&self) -> Self {
            todo!()
        }

        pub const fn read<T: 'static>(self) -> &'a [T] {
            todo!()
        }

        pub const fn write<T: 'static>(self) -> &'a mut [T] {
            todo!()
        }
    }

    impl Prepare for () {
        type Item<'a> = ();

        fn prepare<'a>(&self, _: Context<'a>) -> Self::Item<'a> {}
    }

    impl<T: Prepare> Prepare for (T,) {
        type Item<'a> = T::Item<'a>;

        fn prepare<'a>(&self, context: Context<'a>) -> Self::Item<'a> {
            self.0.prepare(context)
        }
    }

    impl<L: Prepare, R: Prepare> Prepare for (L, R) {
        type Item<'a> = (L::Item<'a>, R::Item<'a>);

        fn prepare<'a>(&self, context: Context<'a>) -> Self::Item<'a> {
            (self.0.prepare(context.and()), self.1.prepare(context))
        }
    }
}

pub mod task {
    use crate::v3::{
        declare::{self, Declare},
        prepare::{self, Prepare},
    };

    pub struct Context<'a> {
        boba: &'a (),
    }

    pub trait Task: Declare + Prepare {
        unsafe fn run<'a>(&self, context: Context<'a>);
    }

    pub struct Function<I, F>(I, F);

    impl<'a> Context<'a> {
        pub fn next(&mut self) -> Option<Context<'a>> {
            todo!()
        }

        pub fn prepare(&mut self) -> prepare::Context<'a> {
            todo!()
        }
    }

    impl Task for () {
        #[inline]
        unsafe fn run(&self, _: Context<'_>) {
            unreachable!()
        }
    }

    impl<L: Task, R: Task> Task for (L, R) {
        #[inline]
        unsafe fn run<'a>(&self, mut context: Context<'a>) {
            match context.next() {
                Some(context) => unsafe { self.0.run(context) },
                None => unsafe { self.1.run(context) },
            }
        }
    }

    impl<T: Declare, F> Declare for Function<T, F> {
        fn declare<'a>(&self, context: declare::Context<'a>) {
            self.0.declare(context)
        }
    }

    impl<T: Prepare, F> Prepare for Function<T, F> {
        type Item<'a> = T::Item<'a>;

        fn prepare<'a>(&self, context: prepare::Context<'a>) -> Self::Item<'a> {
            self.0.prepare(context)
        }
    }

    impl<T: Declare + Prepare, F: Fn(T::Item<'_>) + Send + Sync> Task for Function<T, F> {
        #[inline]
        unsafe fn run<'a>(&self, mut context: Context<'a>) {
            self.1(self.0.prepare(context.prepare()));
        }
    }

    pub const fn function<I: Prepare, F: Fn(I::Item<'_>) + Send + Sync>(
        input: I,
        run: F,
    ) -> Function<I, F> {
        Function(input, run)
    }
}

pub mod schedule {
    use super::*;
    use crate::v3::{
        declare::Declare,
        prepare::Prepare,
        query::Query,
        task::{Function, Task, function},
    };
    use core::error::Error;
    use std::collections::HashMap;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Access {
        Read,
        Write,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Order {
        Weak,
        Strong,
    }

    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    pub enum Part {
        Store,
        Tables,
        Table(usize),
        Columns,
        Column(usize),
        Fragments,
        Type(TypeId),
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Resource {
        identifier: usize,
        path: Vec<Part>,
        access: Access,
        order: Order,
    }

    struct State {
        resources: Vec<Resource>,
    }

    pub struct Schedule<'a, T: Task = ()> {
        store: &'a mut Store,
        task: T,
        states: Vec<State>,
        graph: Graph,
    }

    struct Graph {}

    impl<'a> Schedule<'a> {
        pub const fn new(store: &'a mut Store) -> Self {
            Self {
                store,
                task: (),
                graph: Graph {},
            }
        }
    }

    impl<'a, T: Task> Schedule<'a, T> {
        pub fn push<U: Task>(mut self, task: U) -> Result<Schedule<'a, (T, U)>, Box<dyn Error>> {
            let resources = Vec::new();
            // TODO: Declare access to the `Store`, validate and convert into a efficient
            // format for scheduling...
            use declare::{Branch, Leaf, Node};

            let node = declare::declare(&task);
            fn descend(
                node: &Node,
                path: impl Iterator<Item = Part> + Clone,
                resources: &mut Vec<Resource>,
                identifiers: &mut HashMap<Vec<Part>, usize>,
                not: bool,
            ) -> Result<(), Box<dyn Error>> {
                match node {
                    Node::Branch(branch, nodes) => {
                        let (key, not) = match branch {
                            Branch::Store => (Some(Part::Store), not),
                            Branch::Tables => (Some(Part::Tables), not),
                            // Branch::Table => todo!(),
                            Branch::Columns => (Some(Part::Columns), not),
                            // Branch::Column => todo!(),
                            Branch::Fragments => (Some(Part::Fragments), not),
                            Branch::Not => (None, !not),
                            _ => todo!(),
                        };
                        for node in nodes {
                            descend(node, path.clone().chain(key), resources, identifiers, not)?;
                        }
                        Ok(())
                    }
                    Node::Leaf(leaf) => match leaf {
                        Leaf::Read(meta) => todo!(),
                        Leaf::Write(meta) => todo!(),
                        // Leaf::Has(meta) => todo!(),
                        _ => todo!(),
                    },
                }
            }

            fn access(node: &Node) -> Access {
                match node {
                    Node::Branch(branch, nodes) => todo!(),
                    Node::Leaf(Leaf::Read(_)) => Access::Read,
                    Node::Leaf(Leaf::Write(_)) => Access::Write(true),
                }
            }

            Ok(Schedule {
                store: self.store,
                graph: self.graph,
                task: (self.task, task),
                resource: self.resource,
                states: self.states,
            })
        }

        pub fn function<I: Declare + Prepare, F: Fn(I::Item<'_>) + Send + Sync>(
            self,
            input: I,
            run: F,
        ) -> Result<Schedule<'a, (T, Function<I, F>)>, Box<dyn Error>> {
            self.push(function(input, run))
        }

        pub fn run(&mut self) -> Result<(), Box<dyn Error>> {
            // TODO: How to know to reschedule?
            // if self.task.update(self.store) {
            // TODO: Reschedule.
            // }
            // TODO: Run according to schedule.
            Ok(())
        }
    }

    fn boba() -> Result<(), Box<dyn Error>> {
        let mut store = Store::new();
        let mut schedule = store
            .schedule()
            .function(Query::new().read::<()>(), |a| {})?
            .function(Query::new().read::<()>().write::<()>().has::<()>(), |a| {})?;
        for _ in 0..1_000 {
            schedule.run()?;
        }
        Ok(())
    }
}

pub mod query {
    use super::*;
    use crate::v3::{
        declare::{self, Declare},
        prepare::{self, Prepare},
    };
    use core::marker::PhantomData;

    pub struct Query<Q, F>(Q, F);

    pub struct Has<T>(PhantomData<T>);
    pub struct Not<F>(F);
    pub struct Read<T>(PhantomData<T>);
    pub struct Write<T>(PhantomData<T>);

    pub enum Dynamic {
        Read(TypeId),
        Write(TypeId),
        All(Box<[Dynamic]>),
    }

    impl Query<(), ()> {
        pub const fn new() -> Self {
            Self((), ())
        }
    }

    impl<Q: Stack, F> Query<Q, F> {
        pub fn read<T: 'static>(self) -> Query<Q::Push<Read<T>>, F> {
            Query(self.0.push(Read(PhantomData)), self.1)
        }

        pub fn write<T: 'static>(self) -> Query<Q::Push<Write<T>>, F> {
            Query(self.0.push(Write(PhantomData)), self.1)
        }
    }

    impl<Q, F: Stack> Query<Q, F> {
        pub fn has<T: 'static>(self) -> Query<Q, F::Push<Has<T>>> {
            Query(self.0, self.1.push(Has(PhantomData)))
        }
    }

    impl<Q: Declare, F: Declare> Declare for Query<Q, F> {
        fn declare<'a>(&self, context: declare::Context<'a>) {
            let mut context = context.tables();
            self.0.declare(context.own());
            self.1.declare(context)
        }
    }

    impl<Q: Prepare, F: Declare> Prepare for Query<Q, F> {
        type Item<'a> = Q::Item<'a>;

        fn prepare<'a>(&self, context: prepare::Context<'a>) -> Self::Item<'a> {
            self.0.prepare(context.table())
        }
    }

    impl<T: 'static> Declare for Read<T> {
        fn declare<'a>(&self, context: declare::Context<'a>) {
            context.read::<T>()
        }
    }

    impl<T: 'static> Prepare for Read<T> {
        type Item<'a> = &'a [T];

        fn prepare<'a>(&self, context: prepare::Context<'a>) -> Self::Item<'a> {
            context.read()
        }
    }

    impl<T: 'static> Declare for Write<T> {
        fn declare<'a>(&self, context: declare::Context<'a>) {
            context.write::<T>()
        }
    }

    impl<T: 'static> Prepare for Write<T> {
        type Item<'a> = &'a mut [T];

        fn prepare<'a>(&self, context: prepare::Context<'a>) -> Self::Item<'a> {
            context.write()
        }
    }

    impl<T: 'static> Declare for Has<T> {
        fn declare<'a>(&self, context: declare::Context<'a>) {
            context.has::<T>()
        }
    }

    impl<F: Declare> Declare for Not<F> {
        fn declare<'a>(&self, context: declare::Context<'a>) {
            self.0.declare(context.not())
        }
    }
}

trait Stack {
    type Push<P>;
    type Pop;

    fn push<P>(self, item: P) -> Self::Push<P>;
    fn pop(self) -> Self::Pop;
}

impl Stack for () {
    type Pop = ();
    type Push<P> = (P,);

    fn push<P>(self, item: P) -> Self::Push<P> {
        (item,)
    }

    fn pop(self) -> Self::Pop {}
}

impl<T> Stack for (T,) {
    type Pop = ();
    type Push<P> = (T, P);

    fn push<P>(self, item: P) -> Self::Push<P> {
        (self.0, item)
    }

    fn pop(self) -> Self::Pop {}
}

impl<T0, T1> Stack for (T0, T1) {
    type Pop = (T0,);
    type Push<P> = (T0, T1, P);

    fn push<P>(self, item: P) -> Self::Push<P> {
        (self.0, self.1, item)
    }

    fn pop(self) -> Self::Pop {
        (self.0,)
    }
}

impl<T0, T1, T2> Stack for (T0, T1, T2) {
    type Pop = (T0, T1);
    type Push<P> = (T0, T1, T2, P);

    fn push<P>(self, item: P) -> Self::Push<P> {
        (self.0, self.1, self.2, item)
    }

    fn pop(self) -> Self::Pop {
        (self.0, self.1)
    }
}

impl<T0, T1, T2, T3> Stack for (T0, T1, T2, T3) {
    type Pop = (T0, T1, T2);
    type Push<P> = (T0, T1, T2, T3, P);

    fn push<P>(self, item: P) -> Self::Push<P> {
        (self.0, self.1, self.2, self.3, item)
    }

    fn pop(self) -> Self::Pop {
        (self.0, self.1, self.2)
    }
}

impl<T0, T1, T2, T3, T4> Stack for (T0, T1, T2, T3, T4) {
    type Pop = (T0, T1, T2, T3);
    type Push<P> = (T0, T1, T2, T3, T4, P);

    fn push<P>(self, item: P) -> Self::Push<P> {
        (self.0, self.1, self.2, self.3, self.4, item)
    }

    fn pop(self) -> Self::Pop {
        (self.0, self.1, self.2, self.3)
    }
}

impl<T0, T1, T2, T3, T4, T5> Stack for (T0, T1, T2, T3, T4, T5) {
    type Pop = (T0, T1, T2, T3, T4);
    type Push<P> = (T0, T1, T2, T3, T4, T5, P);

    fn push<P>(self, item: P) -> Self::Push<P> {
        (self.0, self.1, self.2, self.3, self.4, self.5, item)
    }

    fn pop(self) -> Self::Pop {
        (self.0, self.1, self.2, self.3, self.4)
    }
}

fn push<S: Stack, T>(stack: S, item: T) -> S::Push<T> {
    stack.push(item)
}

fn pop<S: Stack>(stack: S) -> S::Pop {
    todo!()
}

fn karl() {
    let a = ();
    let a = push(a, 'a');
    let a = push(a, 'a');
    let a = push(a, 'a');
    let a = push(a, 'a');
    let a = push(a, 'a');
    let a = push(a, 'a');
    let a = pop(a);
    let a = pop(a);
    let a = pop(a);
    let a = pop(a);
    let a = pop(a);
    let a = pop(a);
    let a = pop(a);
    let a = pop(a);
    let a = pop(a);
}
