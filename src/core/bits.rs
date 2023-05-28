use super::utility::get_unchecked_mut;
use std::{
    convert::TryInto,
    hash::Hash,
    iter::{self, FromIterator},
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not},
};

pub type Bucket = usize;

#[derive(Default, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bits {
    buckets: Vec<Bucket>,
}

pub struct Iterator<'a> {
    bucket: usize,
    shift: usize,
    bits: &'a Bits,
}

impl Bits {
    pub const EMPTY: Self = Self::new();

    const SIZE: usize = Bucket::BITS as usize;

    #[inline]
    pub const fn new() -> Self {
        Self {
            buckets: Vec::new(),
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.buckets.len() * Self::SIZE
    }

    #[inline]
    pub fn buckets(&self) -> impl IntoIterator<Item = Bucket> + '_ {
        self.buckets.iter().copied()
    }

    #[inline]
    pub fn has(&self, index: usize) -> bool {
        if let Some(&bucket) = self.buckets.get(index / Self::SIZE) {
            let bit = 1 << (index % Self::SIZE);
            (bucket & bit) == bit
        } else {
            false
        }
    }

    #[inline]
    pub fn has_all(&self, bits: &Bits) -> bool {
        self.buckets.len() == bits.buckets.len()
            && self
                .buckets
                .iter()
                .zip(bits.buckets.iter())
                .all(|(&left, &right)| left & right == right)
    }

    #[inline]
    pub fn has_any(&self, bits: &Bits) -> bool {
        self.buckets
            .iter()
            .zip(bits.buckets.iter())
            .any(|(&left, &right)| left & right > 0)
    }

    #[inline]
    pub fn has_none(&self, bits: &Bits) -> bool {
        !self.has_any(bits)
    }

    #[inline]
    pub fn set(&mut self, index: usize, value: bool) -> bool {
        if value {
            self.ensure(index + 1);
            let bucket = unsafe { get_unchecked_mut(&mut self.buckets, index / Self::SIZE) };
            let bit = 1 << (index % Self::SIZE);
            Self::map(bucket, |bucket| bucket | bit)
        } else if let Some(bucket) = self.buckets.get_mut(index / Self::SIZE) {
            let bit = 1 << (index % Self::SIZE);
            let change = Self::map(bucket, |bucket| bucket & !bit);
            self.shrink();
            change
        } else {
            false
        }
    }

    #[inline]
    pub fn copy(&mut self, bits: &Self) {
        self.buckets.resize(bits.buckets.len(), 0);
        self.buckets.copy_from_slice(&bits.buckets);
    }

    #[inline]
    pub fn not(&mut self) {
        self.buckets.iter_mut().for_each(|value| *value = !*value);
        self.shrink();
    }

    #[inline]
    pub fn or(&mut self, bits: &Bits) {
        // No need to shrink since an 'or' operation cannot add more '0' bits to a bucket.
        self.binary(bits, true, false, |left, right| left | right)
    }

    #[inline]
    pub fn or_not(&mut self, bits: &Bits) {
        // No need to shrink since an 'or' operation cannot add more '0' bits to a bucket.
        self.binary(bits, true, false, |left, right| left | !right)
    }

    #[inline]
    pub fn and(&mut self, bits: &Bits) {
        self.binary(bits, false, true, |left, right| left & right)
    }

    #[inline]
    pub fn and_not(&mut self, bits: &Bits) {
        self.binary(bits, false, true, |left, right| left & !right)
    }

    #[inline]
    pub fn xor(&mut self, bits: &Bits) {
        self.binary(bits, true, true, |left, right| left ^ right)
    }

    #[inline]
    pub fn xor_not(&mut self, bits: &Bits) {
        self.binary(bits, true, true, |left, right| left ^ !right)
    }

    #[inline]
    fn ensure(&mut self, capacity: usize) {
        while self.capacity() < capacity {
            self.buckets.push(0);
        }
    }

    #[inline]
    fn shrink(&mut self) {
        while let Some(&0) = self.buckets.last() {
            self.buckets.pop();
        }
    }

    #[inline]
    fn binary(
        &mut self,
        bits: &Bits,
        ensure: bool,
        shrink: bool,
        mut merge: impl FnMut(Bucket, Bucket) -> Bucket,
    ) {
        let count = if ensure {
            self.ensure(bits.capacity());
            self.buckets.len()
        } else {
            self.buckets.len().min(bits.buckets.len())
        };

        for i in 0..count {
            self.buckets[i] = merge(self.buckets[i], bits.buckets[i]);
        }

        if shrink {
            self.shrink();
        }
    }

    fn map(bucket: &mut Bucket, map: impl FnOnce(Bucket) -> Bucket) -> bool {
        let old = *bucket;
        let new = map(old);
        *bucket = new;
        old != new
    }
}

impl<'a> IntoIterator for &'a Bits {
    type Item = usize;
    type IntoIter = Iterator<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Iterator {
            bucket: 0,
            shift: 0,
            bits: self,
        }
    }
}

impl iter::Iterator for Iterator<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(&bucket) = self.bits.buckets.get(self.bucket) {
            if bucket > 0 {
                while self.shift < Bits::SIZE {
                    let shift = self.shift;
                    let bit = 1 << shift;
                    if bit > bucket {
                        // Early break since the upper bits of the bucket are all zeroes.
                        break;
                    }

                    self.shift += 1;
                    if (bucket & bit) == bit {
                        return Some(self.bucket * Bits::SIZE + shift);
                    }
                }
            }

            self.bucket += 1;
            self.shift = 0;
        }
        None
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let upper = self.bits.capacity() - self.bucket * Bits::SIZE - self.shift;
        (0, Some(upper))
    }
}

impl BitOr<&Bits> for Bits {
    type Output = Bits;

    #[inline]
    fn bitor(mut self, rhs: &Bits) -> Self::Output {
        self.or(rhs);
        self
    }
}

impl BitOrAssign<&Bits> for Bits {
    #[inline]
    fn bitor_assign(&mut self, rhs: &Bits) {
        self.or(rhs);
    }
}

impl BitAnd<&Bits> for Bits {
    type Output = Bits;

    #[inline]
    fn bitand(mut self, rhs: &Bits) -> Self::Output {
        self.and(rhs);
        self
    }
}

impl BitAndAssign<&Bits> for Bits {
    #[inline]
    fn bitand_assign(&mut self, rhs: &Bits) {
        self.and(rhs);
    }
}

impl BitXor<&Bits> for Bits {
    type Output = Bits;

    #[inline]
    fn bitxor(mut self, rhs: &Bits) -> Self::Output {
        self.xor(rhs);
        self
    }
}

impl BitXorAssign<&Bits> for Bits {
    #[inline]
    fn bitxor_assign(&mut self, rhs: &Bits) {
        self.xor(rhs);
    }
}

impl Not for Bits {
    type Output = Bits;

    #[inline]
    fn not(mut self) -> Self::Output {
        Bits::not(&mut self);
        self
    }
}

impl<I: TryInto<usize>> FromIterator<I> for Bits {
    fn from_iter<T: IntoIterator<Item = I>>(iter: T) -> Self {
        let mut bits = Bits::new();
        for index in iter {
            if let Ok(index) = index.try_into() {
                bits.set(index, true);
            }
        }
        bits
    }
}
