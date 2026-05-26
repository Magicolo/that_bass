pub mod v1;
pub mod v2;
pub mod v3;

pub mod boba {
    use core::{
        any::TypeId,
        cell::UnsafeCell,
        iter::empty,
        ptr::null_mut,
        sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering},
    };
    use orn::{Or1, Or2, Or3};
    use std::array::from_fn;

    #[derive(Default)]
    struct Boba {
        a: usize,
        b: bool,
        c: Vec<Fett>,
    }

    #[derive(Default)]
    struct Fett([char; 2]);

    macro_rules! path {
        (Field) => {
            fn parts(&self) -> impl Iterator<Item = Part>
            where
                Self: Sized,
            {
                [Part::Field(TypeId::of::<Self>())].into_iter()
            }
        };
    }

    impl Boba {
        const A: boba::A = boba::A;
        const B: boba::B = boba::B;
        const C: boba::C = boba::C;
    }

    mod boba {
        use super::*;
        use core::{
            ptr::{addr_of, addr_of_mut},
            str::FromStr,
        };

        pub struct A;
        pub struct B;
        pub struct C;
        pub enum Key {
            A,
            B,
            C,
        }

        impl From<A> for Key {
            fn from(_: A) -> Self {
                Key::A
            }
        }

        impl From<B> for Key {
            fn from(_: B) -> Self {
                Key::B
            }
        }

        impl From<C> for Key {
            fn from(_: C) -> Self {
                Key::C
            }
        }

        impl FromStr for Key {
            type Err = ();

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    "a" => Ok(Key::A),
                    "b" => Ok(Key::B),
                    "c" => Ok(Key::C),
                    _ => Err(()),
                }
            }
        }

        impl TryFrom<&str> for Key {
            type Error = ();

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                value.parse()
            }
        }

        impl TryFrom<usize> for Key {
            type Error = ();

            fn try_from(value: usize) -> Result<Self, Self::Error> {
                match value {
                    0 => Ok(Key::A),
                    1 => Ok(Key::B),
                    2 => Ok(Key::C),
                    _ => Err(()),
                }
            }
        }

        impl super::Key<Boba> for A {
            type Value = usize;

            path!(Field);

            fn get(&self, instance: Boba) -> Self::Value {
                instance.a
            }
        }

        impl<'a> super::Key<&'a Boba> for A {
            type Value = &'a usize;

            path!(Field);

            fn get(&self, instance: &'a Boba) -> Self::Value {
                &instance.a
            }
        }

        impl<'a> super::Key<&'a mut Boba> for A {
            type Value = &'a usize;

            path!(Field);

            fn get(&self, instance: &'a mut Boba) -> Self::Value {
                &mut instance.a
            }
        }

        impl super::Key<*const Boba> for A {
            type Value = *const usize;

            path!(Field);

            fn get(&self, instance: *const Boba) -> Self::Value {
                unsafe { addr_of!((*instance).a) }
            }
        }

        impl super::Key<*mut Boba> for A {
            type Value = *mut usize;

            path!(Field);

            fn get(&self, instance: *mut Boba) -> Self::Value {
                unsafe { addr_of_mut!((*instance).a) }
            }
        }

        impl super::Key<Boba> for B {
            type Value = bool;

            path!(Field);

            fn get(&self, instance: Boba) -> Self::Value {
                instance.b
            }
        }

        impl<'a> super::Key<&'a Boba> for B {
            type Value = &'a bool;

            path!(Field);

            fn get(&self, instance: &'a Boba) -> Self::Value {
                &instance.b
            }
        }

        impl<'a> super::Key<&'a mut Boba> for B {
            type Value = &'a bool;

            path!(Field);

            fn get(&self, instance: &'a mut Boba) -> Self::Value {
                &mut instance.b
            }
        }

        impl super::Key<*const Boba> for B {
            type Value = *const bool;

            path!(Field);

            fn get(&self, instance: *const Boba) -> Self::Value {
                unsafe { addr_of!((*instance).b) }
            }
        }

        impl super::Key<*mut Boba> for B {
            type Value = *mut bool;

            path!(Field);

            fn get(&self, instance: *mut Boba) -> Self::Value {
                unsafe { addr_of_mut!((*instance).b) }
            }
        }

        impl super::Key<Boba> for C {
            type Value = Vec<Fett>;

            path!(Field);

            fn get(&self, instance: Boba) -> Self::Value {
                instance.c
            }
        }

        impl<'a> super::Key<&'a Boba> for C {
            type Value = &'a Vec<Fett>;

            path!(Field);

            fn get(&self, instance: &'a Boba) -> Self::Value {
                &instance.c
            }
        }

        impl<'a> super::Key<&'a mut Boba> for C {
            type Value = &'a Vec<Fett>;

            path!(Field);

            fn get(&self, instance: &'a mut Boba) -> Self::Value {
                &mut instance.c
            }
        }

        impl super::Key<*const Boba> for C {
            type Value = *const Vec<Fett>;

            path!(Field);

            fn get(&self, instance: *const Boba) -> Self::Value {
                unsafe { addr_of!((*instance).c) }
            }
        }

        impl super::Key<*mut Boba> for C {
            type Value = *mut Vec<Fett>;

            const REFRESH_ON_DROP: bool = true;

            path!(Field);

            fn get(&self, instance: *mut Boba) -> Self::Value {
                unsafe { addr_of_mut!((*instance).c) }
            }
        }

        impl super::Key<Boba> for Key {
            type Value = Or3<
                <A as super::Key<Boba>>::Value,
                <B as super::Key<Boba>>::Value,
                <C as super::Key<Boba>>::Value,
            >;

            fn get(&self, instance: Boba) -> Self::Value {
                match self {
                    Key::A => Or3::T0(A.get(instance)),
                    Key::B => Or3::T1(B.get(instance)),
                    Key::C => Or3::T2(C.get(instance)),
                }
            }

            fn parts(&self) -> impl Iterator<Item = Part>
            where
                Self: Sized,
            {
                match self {
                    Key::A => Or3::T0(super::Key::<Boba>::parts(&A)),
                    Key::B => Or3::T1(super::Key::<Boba>::parts(&B)),
                    Key::C => Or3::T2(super::Key::<Boba>::parts(&C)),
                }
                .into_iter()
                .map(Or3::into)
            }
        }
    }

    mod fett {
        use super::*;

        pub struct _0;
        pub enum Key {
            _0,
        }

        impl From<_0> for Key {
            fn from(_: _0) -> Self {
                Self::_0
            }
        }

        impl TryFrom<usize> for Key {
            type Error = ();

            fn try_from(value: usize) -> Result<Self, Self::Error> {
                match value {
                    0 => Ok(Self::_0),
                    _ => Err(()),
                }
            }
        }

        impl super::Key<Fett> for Key {
            type Value = Or1<<_0 as super::Key<Fett>>::Value>;

            fn get(&self, instance: Fett) -> Self::Value {
                match self {
                    Self::_0 => Or1::T0(_0.get(instance)),
                }
            }

            fn parts(&self) -> impl Iterator<Item = Part>
            where
                Self: Sized,
            {
                match self {
                    Self::_0 => _0.parts(),
                }
            }
        }

        impl super::Key<Fett> for _0 {
            type Value = [char; 2];

            path!(Field);

            fn get(&self, instance: Fett) -> Self::Value {
                instance.0
            }
        }
    }

    impl Boba {
        const A_INDEX: Index<Usize<0>> = Index::VALUE;
        const A_NAME: Name<(Char<'a'>,)> = Name::VALUE;
        const B_INDEX: Index<Usize<1>> = Index::VALUE;
        const B_NAME: Name<(Char<'b'>,)> = Name::VALUE;
        const C_INDEX: Index<Usize<2>> = Index::VALUE;
        const C_NAME: Name<(Char<'c'>,)> = Name::VALUE;
    }

    struct Char<const C: char>(());
    struct Usize<const N: usize>(());
    struct Name<N>(N);
    struct Index<I>(I);
    trait Constant {
        const VALUE: Self;
    }

    impl<N: Constant> Constant for Name<N> {
        const VALUE: Self = Self(N::VALUE);
    }

    impl<I: Constant> Constant for Index<I> {
        const VALUE: Self = Self(I::VALUE);
    }

    impl<const C: char> Constant for Char<C> {
        const VALUE: Self = Self(());
    }

    impl<const N: usize> Constant for Usize<N> {
        const VALUE: Self = Self(());
    }

    impl<C0: Constant> Constant for (C0,) {
        const VALUE: Self = (C0::VALUE,);
    }

    trait Getz<K> {
        type Value<'a>
        where
            Self: 'a;
        type ValueMut<'a>
        where
            Self: 'a;
        type Raw;
        type RawMut;

        fn getz(&self, key: K) -> Self::Value<'_>;
        fn getz_mut(&mut self, key: K) -> Self::ValueMut<'_>;
        fn getz_raw(this: *const Self, key: K) -> Self::Raw;
        fn getz_raw_mut(this: *mut Self, key: K) -> Self::RawMut;
    }

    trait Keyz {}
    struct Same;
    struct May<K>(K);

    impl<T> Getz<Same> for T {
        type Raw = *const Self;
        type RawMut = *mut Self;
        type Value<'a>
            = &'a Self
        where
            Self: 'a;
        type ValueMut<'a>
            = &'a mut Self
        where
            Self: 'a;

        fn getz(&self, _: Same) -> Self::Value<'_> {
            self
        }

        fn getz_mut(&mut self, _: Same) -> Self::ValueMut<'_> {
            self
        }

        fn getz_raw(this: *const Self, _: Same) -> Self::Raw {
            this
        }

        fn getz_raw_mut(this: *mut Self, _: Same) -> Self::RawMut {
            this
        }
    }

    impl<K, G: Getz<K>> Getz<May<K>> for Option<G> {
        type Raw = Option<G::Raw>;
        type RawMut = Option<G::RawMut>;
        type Value<'a>
            = Option<G::Value<'a>>
        where
            Self: 'a;
        type ValueMut<'a>
            = Option<G::ValueMut<'a>>
        where
            Self: 'a;

        fn getz(&self, May(key): May<K>) -> Self::Value<'_> {
            Option::Some(self.as_ref()?.getz(key))
        }

        fn getz_mut(&mut self, May(key): May<K>) -> Self::ValueMut<'_> {
            Option::Some(self.as_mut()?.getz_mut(key))
        }

        fn getz_raw(this: *const Self, May(key): May<K>) -> Self::Raw {
            todo!()
        }

        fn getz_raw_mut(this: *mut Self, May(key): May<K>) -> Self::RawMut {
            todo!()
        }
    }

    pub trait Base: Sized {
        fn then<K>(self, key: K) -> Then<(Self, K)> {
            Then((self, key))
        }

        fn at<const N: usize>(self) -> Then<(Self, At<N>)> {
            self.then(At::<N>)
        }

        fn some(self) -> Some<Self> {
            Some(self)
        }

        fn flat(self) -> Then<(Self, Flat)> {
            self.then(Flat)
        }
    }

    pub trait Key<T> {
        type Value;
        const REFRESH_ON_DROP: bool = false;

        fn get(&self, instance: T) -> Self::Value;
        fn parts(&self) -> impl Iterator<Item = Part>
        where
            Self: Sized;
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub enum Part {
        Field(TypeId),
        At(usize),
    }

    pub struct Then<T>(T);
    pub struct Some<T>(T);
    pub struct Flat;
    pub struct At<const N: usize>;

    impl<T> Base for T {}

    impl<'a, T> Key<&'a [T]> for usize {
        type Value = Option<&'a T>;

        fn get(&self, instance: &'a [T]) -> Self::Value {
            instance.get(*self)
        }

        fn parts(&self) -> impl Iterator<Item = Part>
        where
            Self: Sized,
        {
            [Part::At(*self)].into_iter()
        }
    }

    impl<T> Key<Vec<T>> for usize {
        type Value = Option<T>;

        fn get(&self, instance: Vec<T>) -> Self::Value {
            instance.into_iter().nth(*self)
        }

        fn parts(&self) -> impl Iterator<Item = Part>
        where
            Self: Sized,
        {
            [Part::At(*self)].into_iter()
        }
    }

    impl<'a, T> Key<&'a Vec<T>> for usize {
        type Value = Option<&'a T>;

        fn get(&self, instance: &'a Vec<T>) -> Self::Value {
            instance.get(*self)
        }

        fn parts(&self) -> impl Iterator<Item = Part>
        where
            Self: Sized,
        {
            [Part::At(*self)].into_iter()
        }
    }

    impl<T> Key<*const Vec<T>> for usize {
        type Value = Option<*const T>;

        #[allow(clippy::not_unsafe_ptr_arg_deref)]
        fn get(&self, instance: *const Vec<T>) -> Self::Value {
            let instance = unsafe { &*instance };
            if *self < instance.len() {
                core::option::Option::Some(unsafe { instance.as_ptr().add(*self) })
            } else {
                None
            }
        }

        fn parts(&self) -> impl Iterator<Item = Part>
        where
            Self: Sized,
        {
            [Part::At(*self)].into_iter()
        }
    }

    impl<T> Key<*mut Vec<T>> for usize {
        type Value = Option<*mut T>;

        #[allow(clippy::not_unsafe_ptr_arg_deref)]
        fn get(&self, instance: *mut Vec<T>) -> Self::Value {
            let instance = unsafe { &mut *instance };
            if *self < instance.len() {
                core::option::Option::Some(unsafe { instance.as_mut_ptr().add(*self) })
            } else {
                None
            }
        }

        fn parts(&self) -> impl Iterator<Item = Part>
        where
            Self: Sized,
        {
            [Part::At(*self)].into_iter()
        }
    }

    impl<T, const N: usize> Key<[T; N]> for usize {
        type Value = Option<T>;

        fn get(&self, instance: [T; N]) -> Self::Value {
            instance.into_iter().nth(*self)
        }

        fn parts(&self) -> impl Iterator<Item = Part>
        where
            Self: Sized,
        {
            [Part::At(*self)].into_iter()
        }
    }

    impl<T> Key<T> for () {
        type Value = T;

        const REFRESH_ON_DROP: bool = true;

        fn get(&self, instance: T) -> Self::Value {
            instance
        }

        fn parts(&self) -> impl Iterator<Item = Part>
        where
            Self: Sized,
        {
            empty()
        }
    }

    impl<T, K0: Key<T>, K1: Key<K0::Value>> Key<T> for Then<(K0, K1)> {
        type Value = K1::Value;

        const REFRESH_ON_DROP: bool = K1::REFRESH_ON_DROP;

        fn get(&self, instance: T) -> Self::Value {
            self.0.1.get(self.0.0.get(instance))
        }

        fn parts(&self) -> impl Iterator<Item = Part>
        where
            Self: Sized,
        {
            self.0.0.parts().chain(self.0.1.parts())
        }
    }

    impl<T, K: Key<T>> Key<Option<T>> for Some<K> {
        type Value = Option<K::Value>;

        const REFRESH_ON_DROP: bool = K::REFRESH_ON_DROP;

        fn get(&self, instance: Option<T>) -> Self::Value {
            Option::Some(self.0.get(instance?))
        }

        fn parts(&self) -> impl Iterator<Item = Part>
        where
            Self: Sized,
        {
            self.0.parts()
        }
    }

    impl<T, K: Key<T>> Key<Or1<T>> for Some<K> {
        type Value = Or1<K::Value>;

        const REFRESH_ON_DROP: bool = K::REFRESH_ON_DROP;

        fn get(&self, instance: Or1<T>) -> Self::Value {
            instance.map_t0(|instance| self.0.get(instance))
        }

        fn parts(&self) -> impl Iterator<Item = Part>
        where
            Self: Sized,
        {
            self.0.parts()
        }
    }

    impl<T0, T1, K: Key<T0>> Key<Or2<T0, T1>> for Some<K> {
        type Value = Or2<K::Value, T1>;

        const REFRESH_ON_DROP: bool = K::REFRESH_ON_DROP;

        fn get(&self, instance: Or2<T0, T1>) -> Self::Value {
            instance.map_t0(|instance| self.0.get(instance))
        }

        fn parts(&self) -> impl Iterator<Item = Part>
        where
            Self: Sized,
        {
            self.0.parts()
        }
    }

    impl<T0, T1, T2, K: Key<T0>> Key<Or3<T0, T1, T2>> for Some<K> {
        type Value = Or3<K::Value, T1, T2>;

        const REFRESH_ON_DROP: bool = K::REFRESH_ON_DROP;

        fn get(&self, instance: Or3<T0, T1, T2>) -> Self::Value {
            instance.map_t0(|instance| self.0.get(instance))
        }

        fn parts(&self) -> impl Iterator<Item = Part>
        where
            Self: Sized,
        {
            self.0.parts()
        }
    }

    impl<T> Key<Option<Option<T>>> for Flat {
        type Value = Option<T>;

        fn get(&self, instance: Option<Option<T>>) -> Self::Value {
            instance?
        }

        fn parts(&self) -> impl Iterator<Item = Part>
        where
            Self: Sized,
        {
            empty()
        }
    }

    impl<T0, T1, T2> Key<Or3<T0, T1, T2>> for At<2> {
        type Value = Option<T2>;

        fn get(&self, instance: Or3<T0, T1, T2>) -> Self::Value {
            instance.t2()
        }

        fn parts(&self) -> impl Iterator<Item = Part>
        where
            Self: Sized,
        {
            empty()
        }
    }

    fn jango() {
        let b = Boba::default();
        let a = key().get(b);
    }

    fn key() -> impl Key<Boba, Value = Option<char>> {
        boba::Key::C
            .at::<2>()
            .then(1.some())
            .flat()
            .then(fett::_0.some())
            .then(1.some())
            .flat()
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum LockError {
        TooManyLocks,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Bits {
        Static(u32),
        Dynamic(u32),
    }

    #[derive(Clone, Copy)]
    struct Plan {
        set: u32,
        conflict: u32,
        version: u64,
        dynamic: bool,
    }

    pub struct Allocator(AtomicU32);

    impl Allocator {
        fn new() -> Self {
            Self(AtomicU32::new(0))
        }

        fn allocate(&self) -> Result<u32, LockError> {
            let mut allocated = 0;
            self.0
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                    if used == u32::MAX {
                        return None;
                    }

                    let bit = 1_u32 << (!used).trailing_zeros();
                    allocated = bit;
                    core::option::Option::Some(used | bit)
                })
                .map(|_| allocated)
                .map_err(|_| LockError::TooManyLocks)
        }
    }

    struct StaticChild {
        part: Part,
        node: Box<Node>,
    }

    struct Dynamic {
        length: AtomicUsize,
        children: [AtomicPtr<Node>; 32],
        build: fn(&Allocator) -> Result<Box<Node>, LockError>,
        refresh: unsafe fn(&Node, *mut u8, &Skeleton) -> Result<(), LockError>,
    }

    impl Drop for Dynamic {
        fn drop(&mut self) {
            for child in self.children.iter_mut() {
                let child = child.load(Ordering::Relaxed);
                if !child.is_null() && child != initializing_node() {
                    unsafe { drop(Box::from_raw(child)) };
                }
            }
        }
    }

    impl Dynamic {
        fn new<T: Shape>() -> Self {
            Self {
                length: AtomicUsize::new(0),
                children: from_fn(|_| AtomicPtr::new(null_mut())),
                build: |allocator| Ok(Box::new(T::build(allocator)?)),
                refresh: |node, value, skeleton| unsafe {
                    T::refresh_shape(node, value.cast::<T>(), skeleton)
                },
            }
        }

        fn child(&self, index: usize) -> Option<&Node> {
            let child = self.children.get(index)?.load(Ordering::Acquire);
            if child.is_null() || child == initializing_node() {
                return None;
            }

            core::option::Option::Some(unsafe { &*child })
        }

        fn ensure_child(
            &self,
            index: usize,
            skeleton: &Skeleton,
        ) -> Result<Option<&Node>, LockError> {
            let core::option::Option::Some(slot) = self.children.get(index) else {
                return Err(LockError::TooManyLocks);
            };

            loop {
                let child = slot.load(Ordering::Acquire);
                if child.is_null() {
                    if slot
                        .compare_exchange(
                            null_mut(),
                            initializing_node(),
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        match (self.build)(&skeleton.allocator) {
                            Ok(child) => {
                                let child = Box::into_raw(child);
                                slot.store(child, Ordering::Release);
                                skeleton.version.fetch_add(1, Ordering::AcqRel);
                                return Ok(core::option::Option::Some(unsafe { &*child }));
                            }
                            Err(error) => {
                                slot.store(null_mut(), Ordering::Release);
                                return Err(error);
                            }
                        }
                    }
                } else if child == initializing_node() {
                    core::hint::spin_loop();
                } else {
                    return Ok(core::option::Option::Some(unsafe { &*child }));
                }
            }
        }
    }

    pub struct Node {
        bit: u32,
        static_children: Box<[StaticChild]>,
        dynamic: Option<Dynamic>,
    }

    impl Node {
        fn new(
            allocator: &Allocator,
            static_children: Box<[StaticChild]>,
            dynamic: Option<Dynamic>,
        ) -> Result<Self, LockError> {
            Ok(Self {
                bit: allocator.allocate()?,
                static_children,
                dynamic,
            })
        }

        fn leaf(allocator: &Allocator) -> Result<Self, LockError> {
            Self::new(allocator, Box::new([]), None)
        }

        fn child(&self, part: Part) -> Option<&Node> {
            self.static_children
                .iter()
                .find(|child| child.part == part)
                .map(|child| child.node.as_ref())
                .or_else(|| match (&self.dynamic, part) {
                    (core::option::Option::Some(dynamic), Part::At(index)) => dynamic.child(index),
                    _ => None,
                })
        }

        fn descendants(&self) -> u32 {
            let mut bits = 0;
            for child in self.static_children.iter() {
                bits |= child.node.bit | child.node.descendants();
            }

            if let core::option::Option::Some(dynamic) = &self.dynamic {
                for child in dynamic.children.iter() {
                    let child = child.load(Ordering::Acquire);
                    if child.is_null() || child == initializing_node() {
                        continue;
                    }

                    let child = unsafe { &*child };
                    bits |= child.bit | child.descendants();
                }
            }

            bits
        }

        fn has_missing_dynamic_children(&self) -> bool {
            let core::option::Option::Some(dynamic) = &self.dynamic else {
                return false;
            };

            let length = dynamic
                .length
                .load(Ordering::Acquire)
                .min(dynamic.children.len());
            (0..length).any(|index| dynamic.child(index).is_none())
        }
    }

    fn initializing_node() -> *mut Node {
        core::ptr::dangling_mut::<Node>()
    }

    pub struct Skeleton {
        root: Box<Node>,
        allocator: Allocator,
        version: AtomicU64,
    }

    impl Skeleton {
        fn new<T: Shape>() -> Result<Self, LockError> {
            let allocator = Allocator::new();
            Ok(Self {
                root: Box::new(T::build(&allocator)?),
                allocator,
                version: AtomicU64::new(0),
            })
        }

        fn plan(&self, parts: impl IntoIterator<Item = Part>) -> Result<Plan, LockError> {
            let version = self.version.load(Ordering::Acquire);
            let mut ancestor_bits = 0;
            let mut node = self.root.as_ref();

            for part in parts {
                match node.child(part) {
                    core::option::Option::Some(child) => {
                        ancestor_bits |= node.bit;
                        node = child;
                    }
                    None => {
                        let dynamic = match (&node.dynamic, part) {
                            (core::option::Option::Some(dynamic), Part::At(index)) => {
                                let length = dynamic.length.load(Ordering::Acquire);
                                if index >= dynamic.children.len() && index < length {
                                    return Err(LockError::TooManyLocks);
                                }
                                index < length
                            }
                            _ => false,
                        };
                        let conflict = ancestor_bits | node.bit | node.descendants();
                        return Ok(Plan {
                            set: node.bit,
                            conflict,
                            version,
                            dynamic,
                        });
                    }
                }
            }

            let descendants = node.descendants();
            Ok(Plan {
                set: node.bit,
                conflict: ancestor_bits | node.bit | descendants,
                version,
                dynamic: node.has_missing_dynamic_children(),
            })
        }

        unsafe fn refresh<T: Shape>(&self, value: *mut T) -> Result<(), LockError> {
            unsafe { T::refresh_shape(&self.root, value, self) }
        }
    }

    /// Describes how to build and refresh the lock skeleton for a value.
    ///
    /// # Safety
    ///
    /// Implementations must ensure every skeleton edge maps to the same memory
    /// reached by the matching `Key` parts. A wrong mapping can let `Lock` hand
    /// out overlapping mutable projections.
    pub unsafe trait Shape: Sized {
        fn build(allocator: &Allocator) -> Result<Node, LockError> {
            Node::leaf(allocator)
        }

        /// Refreshes dynamic skeleton metadata for `value` at `node`.
        ///
        /// # Safety
        ///
        /// `value` must point to a valid value whose skeleton is rooted at
        /// `node`, and the caller must hold the structural gate bit that makes
        /// reading any dynamic shape metadata safe.
        unsafe fn refresh_shape(
            node: &Node,
            value: *mut Self,
            skeleton: &Skeleton,
        ) -> Result<(), LockError> {
            let _ = (node, value, skeleton);
            Ok(())
        }
    }

    unsafe impl Shape for () {}
    unsafe impl Shape for usize {}
    unsafe impl Shape for bool {}
    unsafe impl Shape for char {}

    unsafe impl<T: Shape> Shape for Vec<T> {
        fn build(allocator: &Allocator) -> Result<Node, LockError> {
            Node::new(
                allocator,
                Box::new([]),
                core::option::Option::Some(Dynamic::new::<T>()),
            )
        }

        unsafe fn refresh_shape(
            node: &Node,
            value: *mut Self,
            skeleton: &Skeleton,
        ) -> Result<(), LockError> {
            let core::option::Option::Some(dynamic) = &node.dynamic else {
                return Ok(());
            };

            let value = unsafe { &mut *value };
            let length = value.len();
            dynamic.length.store(length, Ordering::Release);

            let base = value.as_mut_ptr();
            for index in 0..length.min(dynamic.children.len()) {
                if let core::option::Option::Some(child) = dynamic.ensure_child(index, skeleton)? {
                    unsafe { (dynamic.refresh)(child, base.add(index).cast::<u8>(), skeleton)? };
                }
            }

            Ok(())
        }
    }

    unsafe impl<T: Shape, const N: usize> Shape for [T; N] {
        fn build(allocator: &Allocator) -> Result<Node, LockError> {
            let mut children = Vec::with_capacity(N);
            for index in 0..N {
                children.push(StaticChild {
                    part: Part::At(index),
                    node: Box::new(T::build(allocator)?),
                });
            }

            Node::new(allocator, children.into_boxed_slice(), None)
        }

        unsafe fn refresh_shape(
            node: &Node,
            value: *mut Self,
            skeleton: &Skeleton,
        ) -> Result<(), LockError> {
            let base = value.cast::<T>();
            for child in node.static_children.iter() {
                let Part::At(index) = child.part else {
                    continue;
                };
                unsafe { T::refresh_shape(&child.node, base.add(index), skeleton)? };
            }
            Ok(())
        }
    }

    unsafe impl Shape for Boba {
        fn build(allocator: &Allocator) -> Result<Node, LockError> {
            Node::new(
                allocator,
                vec![
                    StaticChild {
                        part: Part::Field(TypeId::of::<boba::A>()),
                        node: Box::new(usize::build(allocator)?),
                    },
                    StaticChild {
                        part: Part::Field(TypeId::of::<boba::B>()),
                        node: Box::new(bool::build(allocator)?),
                    },
                    StaticChild {
                        part: Part::Field(TypeId::of::<boba::C>()),
                        node: Box::new(Vec::<Fett>::build(allocator)?),
                    },
                ]
                .into_boxed_slice(),
                None,
            )
        }

        unsafe fn refresh_shape(
            node: &Node,
            value: *mut Self,
            skeleton: &Skeleton,
        ) -> Result<(), LockError> {
            for child in node.static_children.iter() {
                if child.part == Part::Field(TypeId::of::<boba::C>()) {
                    unsafe {
                        Vec::<Fett>::refresh_shape(
                            &child.node,
                            core::ptr::addr_of_mut!((*value).c),
                            skeleton,
                        )?
                    };
                }
            }
            Ok(())
        }
    }

    unsafe impl Shape for Fett {
        fn build(allocator: &Allocator) -> Result<Node, LockError> {
            Node::new(
                allocator,
                vec![StaticChild {
                    part: Part::Field(TypeId::of::<fett::_0>()),
                    node: Box::new(<[char; 2]>::build(allocator)?),
                }]
                .into_boxed_slice(),
                None,
            )
        }

        unsafe fn refresh_shape(
            node: &Node,
            value: *mut Self,
            skeleton: &Skeleton,
        ) -> Result<(), LockError> {
            for child in node.static_children.iter() {
                if child.part == Part::Field(TypeId::of::<fett::_0>()) {
                    unsafe {
                        <[char; 2]>::refresh_shape(
                            &child.node,
                            core::ptr::addr_of_mut!((*value).0),
                            skeleton,
                        )?
                    };
                }
            }
            Ok(())
        }
    }

    pub struct Lock<T>(AtomicU32, UnsafeCell<T>, Skeleton);

    unsafe impl<T: Send> Send for Lock<T> {}
    unsafe impl<T: Send> Sync for Lock<T> {}

    pub struct Guard<'a, T: Shape, V> {
        lock: &'a Lock<T>,
        bits: u32,
        value: V,
        refresh: bool,
    }

    impl<T: Shape> Lock<T> {
        pub fn new(value: T) -> Result<Self, LockError> {
            let lock = Self(
                AtomicU32::new(0),
                UnsafeCell::new(value),
                Skeleton::new::<T>()?,
            );
            unsafe { lock.2.refresh(lock.1.get())? };
            Ok(lock)
        }

        pub fn lock<K: Key<*mut T>>(&self, key: &K) -> Result<Guard<'_, T, K::Value>, LockError> {
            loop {
                match self.try_lock_or_wait(key)? {
                    Ok(guard) => break Ok(guard),
                    Err(locks) => atomic_wait::wait(&self.0, locks),
                }
            }
        }

        pub fn try_lock<K: Key<*mut T>>(
            &self,
            key: &K,
        ) -> Result<Option<Guard<'_, T, K::Value>>, LockError> {
            match self.try_lock_or_wait(key)? {
                Ok(guard) => Ok(core::option::Option::Some(guard)),
                Err(_) => Ok(None),
            }
        }

        fn try_lock_or_wait<K: Key<*mut T>>(
            &self,
            key: &K,
        ) -> Result<Result<Guard<'_, T, K::Value>, u32>, LockError> {
            loop {
                let plan = self.plan(key)?;
                let mut old = self.0.load(Ordering::Acquire);
                loop {
                    if old & plan.conflict == 0 {
                        match self.0.compare_exchange_weak(
                            old,
                            old | plan.set,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        ) {
                            Ok(_) => break,
                            Err(new) => old = new,
                        }
                    } else {
                        return Ok(Err(old));
                    }
                }

                if plan.version != self.2.version.load(Ordering::Acquire) {
                    self.release(plan.set);
                    continue;
                }

                if plan.dynamic {
                    unsafe { self.2.refresh(self.1.get())? };
                    self.release(plan.set);
                    continue;
                }

                return Ok(Ok(Guard {
                    lock: self,
                    bits: plan.set,
                    value: key.get(self.1.get()),
                    refresh: K::REFRESH_ON_DROP,
                }));
            }
        }

        fn release(&self, bits: u32) {
            self.0.fetch_and(!bits, Ordering::AcqRel);
            atomic_wait::wake_all(&self.0);
        }

        fn plan<K: Key<*mut T>>(&self, key: &K) -> Result<Plan, LockError> {
            self.2.plan(key.parts())
        }

        fn bits<K: Key<*mut T>>(&self, key: &K) -> Result<Bits, LockError> {
            let plan = self.plan(key)?;
            if plan.dynamic {
                Ok(Bits::Dynamic(plan.conflict))
            } else {
                Ok(Bits::Static(plan.conflict))
            }
        }
    }

    impl<T: Shape, V> Drop for Guard<'_, T, V> {
        fn drop(&mut self) {
            if self.refresh {
                unsafe {
                    self.lock
                        .2
                        .refresh(self.lock.1.get())
                        .expect("lock skeleton outgrew one AtomicU32 during guard drop")
                };
            }
            self.lock.release(self.bits);
        }
    }

    pub mod path {
        #[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
        pub enum Path {
            Store,
            Tables,
            Table(usize),
            Columns(usize),
            Column(usize, usize),
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum Error {
            InvalidPath,
        }

        pub struct Tables;
        pub struct Columns;
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn lock_bits_use_child_set_bits_and_parent_conflict_bits() {
            let lock = Lock::<Boba>::new(Boba::default()).unwrap();
            let a = lock.plan(&boba::A).unwrap();
            let b = lock.plan(&boba::B).unwrap();
            let c = lock.plan(&boba::C).unwrap();
            let root = lock.plan(&()).unwrap();

            assert_eq!(a.set.count_ones(), 1);
            assert_eq!(b.set.count_ones(), 1);
            assert_eq!(c.set.count_ones(), 1);
            assert_eq!(root.set.count_ones(), 1);
            assert_eq!(root.conflict, root.set | a.set | b.set | c.set);
            assert_eq!(a.conflict & b.set, 0);
            assert_eq!(b.conflict & a.set, 0);
        }

        #[test]
        fn lock_rejects_more_than_thirty_two_dynamic_masks() {
            let result = Lock::<Vec<()>>::new((0..32).map(|_| ()).collect());

            assert!(matches!(result, Err(LockError::TooManyLocks)));
        }

        #[test]
        fn held_parent_blocks_future_child_allocation() {
            let lock = Lock::<Boba>::new(Boba::default()).unwrap();
            let root = lock.try_lock(&()).unwrap().unwrap();

            assert!(lock.try_lock(&boba::A).unwrap().is_none());

            drop(root);
            assert!(lock.try_lock(&boba::A).unwrap().is_some());
        }

        #[test]
        fn vector_field_refreshes_after_mutation() {
            let lock = Lock::<Boba>::new(Boba::default()).unwrap();
            let c = lock.try_lock(&boba::C).unwrap().unwrap();
            unsafe { (*c.value).push(Fett::default()) };

            assert!(lock.try_lock(&boba::C.then(0usize)).unwrap().is_none());

            drop(c);
            assert!(lock.try_lock(&boba::C.then(0usize)).unwrap().is_some());
        }

        #[test]
        fn sibling_fields_can_lock_concurrently() {
            let lock = Lock::<Boba>::new(Boba::default()).unwrap();
            let a = lock.try_lock(&boba::A).unwrap().unwrap();

            assert!(lock.try_lock(&boba::B).unwrap().is_some());

            drop(a);
        }
    }
}

pub mod v4;
pub use v4 as store;
