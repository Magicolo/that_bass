use std::{
    mem::transmute,
    ops::{Deref, DerefMut},
};

pub trait IntoRef {
    type Ref;
    fn into_ref(self) -> Self::Ref;
}

pub struct Ref<B, T>(B, T);
pub struct Mut<B, T>(B, T);

impl<T> IntoRef for &T {
    type Ref = Self;
    #[inline]
    fn into_ref(self) -> Self::Ref {
        self
    }
}

impl<'a, T> IntoRef for &'a mut T {
    type Ref = &'a T;
    #[inline]
    fn into_ref(self) -> Self::Ref {
        self
    }
}

impl<B, T> IntoRef for Ref<B, T> {
    type Ref = Self;
    #[inline]
    fn into_ref(self) -> Self::Ref {
        self
    }
}

impl<B: IntoRef, T> IntoRef for Mut<B, T> {
    type Ref = Ref<B::Ref, T>;
    #[inline]
    fn into_ref(self) -> Self::Ref {
        Ref(self.0.into_ref(), self.1)
    }
}

impl<B, T> Mut<B, T> {
    #[inline]
    pub fn new<'a>(mut value: T, borrow: impl FnOnce(&'a mut T) -> B) -> Self
    where
        T: 'a,
    {
        Self(borrow(unsafe { transmute(&mut value) }), value)
    }

    #[inline]
    pub fn try_new<'a>(mut value: T, borrow: impl FnOnce(&'a mut T) -> Option<B>) -> Result<Self, T>
    where
        T: 'a,
    {
        match borrow(unsafe { transmute(&mut value) }) {
            Some(borrow) => Ok(Self(borrow, value)),
            None => Err(value),
        }
    }

    #[inline]
    pub fn map<'a, C: 'a>(self, map: impl FnOnce(B) -> C) -> Mut<C, T>
    where
        T: 'a,
    {
        Mut::new(self.1, |_| map(self.0))
    }

    /// SAFETY:
    /// - All borrows included in `C` must be still valid when `T` is converted to `U`.
    /// - If `C` holds a borrow in `T`, `U` must include `T`.
    /// - `B` must not alias `T`.
    #[inline]
    pub unsafe fn map_both<C, U>(self, map: impl FnOnce(B, T) -> (C, U)) -> Mut<C, U> {
        let (borrow, value) = map(self.0, self.1);
        Mut(borrow, value)
    }

    #[inline]
    pub fn push<'a, C: 'a>(self, borrow: impl FnOnce(&'a mut B) -> C) -> Mut<C, Self>
    where
        B: 'a,
        T: 'a,
    {
        Mut::new(self, |value| borrow(value.deref_mut()))
    }

    #[inline]
    pub fn pop(self) -> T {
        self.1
    }

    #[inline]
    pub fn pop_with<U>(self, with: impl FnOnce(B) -> U) -> (U, T) {
        (with(self.0), self.1)
    }
}

impl<B, T> Ref<B, T> {
    #[inline]
    pub fn new<'a>(value: T, borrow: impl FnOnce(&'a T) -> B) -> Self
    where
        T: 'a,
    {
        Self(borrow(unsafe { transmute(&value) }), value)
    }

    #[inline]
    pub fn try_new<'a>(value: T, borrow: impl FnOnce(&'a T) -> Option<B>) -> Result<Self, T>
    where
        T: 'a,
    {
        match borrow(unsafe { transmute(&value) }) {
            Some(borrow) => Ok(Self(borrow, value)),
            None => Err(value),
        }
    }

    #[inline]
    pub fn map<'a, C: 'a>(self, map: impl FnOnce(B) -> C) -> Ref<C, T>
    where
        T: 'a,
    {
        Ref::new(self.1, |_| map(self.0))
    }

    /// SAFETY: All borrows included in `C` must be still valid when `T` is converted to `U`.
    /// For example, if `C` holds a borrow in `T`, `U` must include `T`.
    #[inline]
    pub unsafe fn map_both<C, U>(self, map: impl FnOnce(B, T) -> (C, U)) -> Ref<C, U> {
        let (borrow, value) = map(self.0, self.1);
        Ref(borrow, value)
    }

    #[inline]
    pub fn push<'a, C: 'a>(self, push: impl FnOnce(&'a B) -> C) -> Ref<C, Self>
    where
        B: 'a,
        T: 'a,
    {
        Ref::new(self, |value| push(value.deref()))
    }

    #[inline]
    pub fn pop(self) -> T {
        self.1
    }

    #[inline]
    pub fn pop_with<U>(self, with: impl FnOnce(B) -> U) -> (U, T) {
        (with(self.0), self.1)
    }
}

impl<B, T> Deref for Ref<B, T> {
    type Target = B;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<B, T> Deref for Mut<B, T> {
    type Target = B;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<B, T> DerefMut for Mut<B, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use super::Mut;

    #[test]
    fn test_mut() {
        let mut a = (1u8, 2u8);
        let mut guard = Mut::new(&mut a, |(left, _)| left);
        **guard += 1;
        guard.pop().0 += 1;
        a.0 += 1;
        assert_eq!(a.0, 4);
    }

    #[test]
    fn test_alias() {
        let mut a = (0u8, 1u8);
        let b = Mut::new(&mut a, |a| &mut **a);
        let c = b.map(|a| &mut a.0);
        let mut d = None;
        c.map(|a| d = Some(a)); // TODO: `a` must not be able to escape...
                                // a.0 += 1;
                                // **c.deref_mut() += 1;
    }
}
