use std::{
    hint::unreachable_unchecked,
    ops::{Deref, DerefMut},
};
use Or::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Or<L, R> {
    Left(L),
    Right(R),
}

pub enum Iter<L, R> {
    Left(L),
    Right(R),
}

impl<L, R> Or<L, R> {
    #[inline]
    pub fn is_left(&self) -> bool {
        matches!(self, Left(_))
    }

    #[inline]
    pub fn is_right(&self) -> bool {
        matches!(self, Right(_))
    }

    #[inline]
    pub fn left(self) -> Option<L> {
        self.map_or(Some, |_| None)
    }

    #[inline]
    pub unsafe fn left_unchecked(self) -> L {
        debug_assert!(self.is_left());
        match self {
            Left(left) => left,
            // SAFETY: the safety contract must be upheld by the caller.
            Right(_) => unsafe { unreachable_unchecked() },
        }
    }

    #[inline]
    pub fn right(self) -> Option<R> {
        self.map_or(|_| None, Some)
    }

    #[inline]
    pub unsafe fn right_unchecked(self) -> R {
        debug_assert!(self.is_right());
        match self {
            // SAFETY: the safety contract must be upheld by the caller.
            Left(_) => unsafe { unreachable_unchecked() },
            Right(right) => right,
        }
    }

    #[inline]
    pub fn left_or(self, default: L) -> L {
        self.left_or_with(|_| default)
    }

    #[inline]
    pub fn right_or(self, default: R) -> R {
        self.right_or_with(|_| default)
    }

    #[inline]
    pub fn left_or_with(self, right: impl FnOnce(R) -> L) -> L {
        self.map_or(|left| left, right)
    }

    #[inline]
    pub fn right_or_with(self, left: impl FnOnce(L) -> R) -> R {
        self.map_or(left, |right| right)
    }

    #[inline]
    pub fn into_left(self) -> L
    where
        R: Into<L>,
    {
        self.left_or_with(R::into)
    }

    #[inline]
    pub fn into_right(self) -> R
    where
        L: Into<R>,
    {
        self.right_or_with(L::into)
    }

    #[inline]
    pub fn map_or<T>(self, left: impl FnOnce(L) -> T, right: impl FnOnce(R) -> T) -> T {
        match self {
            Left(value) => left(value),
            Right(value) => right(value),
        }
    }

    #[inline]
    pub fn map<T, U>(self, left: impl FnOnce(L) -> T, right: impl FnOnce(R) -> U) -> Or<T, U> {
        match self {
            Left(value) => Left(left(value)),
            Right(value) => Right(right(value)),
        }
    }

    #[inline]
    pub fn map_left<T>(self, left: impl FnOnce(L) -> T) -> Or<T, R> {
        self.map(left, |right| right)
    }

    #[inline]
    pub fn map_right<T>(self, right: impl FnOnce(R) -> T) -> Or<L, T> {
        self.map(|left| left, right)
    }

    #[inline]
    pub fn as_ref(&self) -> Or<&L, &R> {
        match self {
            Left(left) => Left(left),
            Right(right) => Right(right),
        }
    }

    #[inline]
    pub fn as_mut(&mut self) -> Or<&mut L, &mut R> {
        match self {
            Left(left) => Left(left),
            Right(right) => Right(right),
        }
    }
}

impl<L: Clone, R: Clone> Or<&L, &R> {
    #[inline]
    pub fn cloned(self) -> Or<L, R> {
        self.map(L::clone, R::clone)
    }
}

impl<L: Copy, R: Copy> Or<&L, &R> {
    #[inline]
    pub fn copied(self) -> Or<L, R> {
        self.map(L::clone, R::clone)
    }
}

impl<L, R> Into<Result<L, R>> for Or<L, R> {
    #[inline]
    fn into(self) -> Result<L, R> {
        self.map_or(Ok, Err)
    }
}

impl<L: IntoIterator, R: IntoIterator> IntoIterator for Or<L, R> {
    type Item = Or<L::Item, R::Item>;
    type IntoIter = Iter<L::IntoIter, R::IntoIter>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.map_or(
            |left| Iter::Left(left.into_iter()),
            |right| Iter::Right(right.into_iter()),
        )
    }
}

impl<L: Iterator, R: Iterator> Iterator for Iter<L, R> {
    type Item = Or<L::Item, R::Item>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        Some(match self {
            Iter::Left(left) => Left(left.next()?),
            Iter::Right(right) => Right(right.next()?),
        })
    }
}

impl<T: Deref> Deref for Or<T, T> {
    type Target = T::Target;

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Left(left) => left,
            Right(right) => right,
        }
    }
}

impl<T: DerefMut> DerefMut for Or<T, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Left(left) => left,
            Right(right) => right,
        }
    }
}
