pub mod v1;
pub mod v2;
pub mod v3;

pub mod boba {
    use core::{
        any::TypeId,
        cell::UnsafeCell,
        hash::Hash,
        sync::atomic::{AtomicU32, Ordering},
    };
    use orn::{Or1, Or2, Or3};
    use std::collections::HashMap;

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
        fn get(&self, instance: T) -> Self::Value;
        fn parts(&self) -> impl Iterator<Item = Part>
        where
            Self: Sized;
    }

    pub trait Path: Sized {
        type Error;
        fn try_from_parts(parts: impl Iterator<Item = Part>) -> Result<Self, Self::Error>;
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

        fn get(&self, instance: T) -> Self::Value {
            instance
        }

        fn parts(&self) -> impl Iterator<Item = Part>
        where
            Self: Sized,
        {
            [].into_iter()
        }
    }

    impl<T, K0: Key<T>, K1: Key<K0::Value>> Key<T> for Then<(K0, K1)> {
        type Value = K1::Value;

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

        fn get(&self, instance: Option<T>) -> Self::Value {
            instance.map(|instance| self.0.get(instance))
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
            [].into_iter()
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
            [].into_iter()
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

    pub struct Lock<T, P = Vec<usize>>(AtomicU32, UnsafeCell<T>, HashMap<P, u8>);
    pub struct Guard<'a, T>(&'a AtomicU32, u32, T);

    impl<T, P: Path + Hash + Eq> Lock<T, P> {
        pub fn lock<K: Key<*mut T>>(&self, key: &K) -> Result<Guard<'_, K::Value>, P::Error> {
            let bits = self.bits(key)?;
            loop {
                match self.try_lock_once(key, bits) {
                    Ok(guard) => break Ok(guard),
                    Err(locks) => atomic_wait::wait(&self.0, locks),
                }
            }
        }

        pub fn try_lock<K: Key<*mut T>>(
            &self,
            key: &K,
        ) -> Result<Option<Guard<'_, K::Value>>, P::Error> {
            Ok(self.try_lock_once(key, self.bits(key)?).ok())
        }

        fn try_lock_once<K: Key<*mut T>>(
            &self,
            key: &K,
            bits: u32,
        ) -> Result<Guard<'_, K::Value>, u32> {
            let mut old = self.0.load(Ordering::Acquire);
            loop {
                if old & bits == old {
                    match self.0.compare_exchange_weak(
                        old,
                        old | bits,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => return Ok(Guard(&self.0, bits, key.get(self.1.get()))),
                        Err(new) => old = new,
                    }
                } else {
                    break Err(old);
                }
            }
        }

        fn bits<K: Key<*mut T>>(&self, key: &K) -> Result<u32, P::Error> {
            // TODO: Map the key to bits using a path.
            let path = P::try_from_parts(key.parts())?;
            todo!()
        }
    }

    impl<T> Drop for Guard<'_, T> {
        fn drop(&mut self) {
            self.0.fetch_nand(self.1, Ordering::AcqRel);
            atomic_wait::wake_all(self.0);
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

        pub struct Tables;
        pub struct Columns;
    }

    impl FromIterator<Part> for path::Path {
        fn from_iter<I: IntoIterator<Item = Part>>(iter: I) -> Self {
            let mut path = Self::Store;
            for at in iter {
                path = match (path, at) {
                    (Self::Store, Part::Field(field)) if field == TypeId::of::<path::Tables>() => {
                        Self::Tables
                    }
                    (Self::Tables, Part::At(table)) => Self::Table(table),
                    (Self::Table(table), Part::Field(field))
                        if field == TypeId::of::<path::Columns>() =>
                    {
                        Self::Columns(table)
                    }
                    (Self::Columns(table), Part::At(column)) => Self::Column(table, column),
                    _ => panic!("invalid path"),
                };
            }
            path
        }
    }
}
