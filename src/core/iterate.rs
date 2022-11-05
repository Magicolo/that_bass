use std::iter::FusedIterator;

pub trait FullIterator: Iterator + DoubleEndedIterator + ExactSizeIterator + FusedIterator {}
impl<I: Iterator + DoubleEndedIterator + ExactSizeIterator + FusedIterator> FullIterator for I {}
