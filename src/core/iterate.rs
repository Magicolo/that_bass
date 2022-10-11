use std::iter::FusedIterator;

pub trait FullIterator: Iterator + DoubleEndedIterator + ExactSizeIterator + FusedIterator {}
impl<I: Iterator + DoubleEndedIterator + ExactSizeIterator + FusedIterator> FullIterator for I {}

pub trait Iterate {
    type Item<'a>
    where
        Self: 'a;

    fn next(&mut self) -> Option<Self::Item<'_>>;
}
