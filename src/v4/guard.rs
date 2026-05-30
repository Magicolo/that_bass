use derive_more::{Deref, DerefMut, From};
use parking_lot::{MappedRwLockReadGuard, MappedRwLockWriteGuard};

#[derive(Deref, From)]
pub struct Read<'a, T: ?Sized>(#[deref(forward)] MappedRwLockReadGuard<'a, T>);

#[derive(Deref, DerefMut, From)]
pub struct Write<'a, T: ?Sized>(
    #[deref(forward)]
    #[deref_mut(forward)]
    MappedRwLockWriteGuard<'a, T>,
);

impl<'a, T: ?Sized> IntoIterator for &'a Read<'_, T>
where
    &'a T: IntoIterator,
{
    type IntoIter = <&'a T as IntoIterator>::IntoIter;
    type Item = <&'a T as IntoIterator>::Item;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T: ?Sized> IntoIterator for &'a mut Read<'_, T>
where
    &'a T: IntoIterator,
{
    type IntoIter = <&'a T as IntoIterator>::IntoIter;
    type Item = <&'a T as IntoIterator>::Item;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T: ?Sized> IntoIterator for &'a Write<'_, T>
where
    &'a T: IntoIterator,
{
    type IntoIter = <&'a T as IntoIterator>::IntoIter;
    type Item = <&'a T as IntoIterator>::Item;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T: ?Sized> IntoIterator for &'a mut Write<'_, T>
where
    &'a mut T: IntoIterator,
{
    type IntoIter = <&'a mut T as IntoIterator>::IntoIter;
    type Item = <&'a mut T as IntoIterator>::Item;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
