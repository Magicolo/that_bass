use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::ops::Deref;
use core::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Row<'a> {
    row: u32,
    table: u32,
    _marker: PhantomData<&'a ()>,
}

#[derive(Clone)]
pub struct Rows<'a> {
    rows: Range<u32>,
    table: u32,
    _marker: PhantomData<&'a ()>,
}

pub struct At<'a, T: ?Sized>(pub(crate) u32, pub(crate) &'a T);

impl Row<'_> {
    pub(crate) const fn new(row: u32, table: u32) -> Self {
        Self {
            row,
            table,
            _marker: PhantomData,
        }
    }

    pub const fn row(&self) -> u32 {
        self.row
    }

    pub const fn table(&self) -> u32 {
        self.table
    }
}

impl Rows<'_> {
    pub(crate) const fn new(rows: Range<u32>, table: u32) -> Self {
        Self {
            rows,
            table,
            _marker: PhantomData,
        }
    }
}

impl<'a> Iterator for Rows<'a> {
    type Item = Row<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(Row::new(self.rows.next()?, self.table))
    }
}

impl ExactSizeIterator for Rows<'_> {
    fn len(&self) -> usize {
        self.rows.len()
    }
}

impl DoubleEndedIterator for Rows<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        Some(Row::new(self.rows.next_back()?, self.table))
    }
}

impl FusedIterator for Rows<'_> {}

impl<'a, T: ?Sized> At<'a, T> {
    pub const fn index(&self) -> u32 {
        self.0
    }

    pub const fn value(&self) -> &'a T {
        self.1
    }
}

impl<'a, T: ?Sized> Clone for At<'a, T> {
    fn clone(&self) -> Self {
        At(self.0, self.1)
    }
}

impl<'a, T: ?Sized> Copy for At<'a, T> {}

impl<T> Deref for At<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.1
    }
}
