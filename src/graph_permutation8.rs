use std::{iter::FusedIterator, num::NonZeroU8};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct GraphPermutation8(u16);

impl GraphPermutation8 {
    pub fn inner(&self) -> u16 {
        self.0
    }

    pub fn from_inner(value: u16) -> Self {
        debug_assert!(value < FACTORIALS[8]);
        Self(value)
    }

    pub fn swaps(self) -> impl Iterator<Item = Swap> + Clone + FusedIterator {
        SwapsIter8 {
            inner: self.0,
            index: NonZeroU8::MIN,
        }
    }

    pub fn try_from_swaps(swaps: impl Iterator<Item = Swap>) -> Option<Self> {
        let mut inner = 0u16;
        let mut multiplier = 1;
        for (index, swap) in swaps.enumerate() {
            let index = index as u16;
            let s = swap.index as u16;

            let diff = index.checked_sub(s)?;
            multiplier *= index.max(1);
            inner += diff * multiplier;
        }
        Some(Self(inner))
    }

    pub fn apply_to_slice<T>(self, arr: &mut [T]) {
        for (index, swap) in self.swaps().enumerate() {
            swap.apply(index, arr);
        }
    }

    pub fn all_for_n_elements(
        n: usize,
    ) -> impl Iterator<Item = Self> + ExactSizeIterator + DoubleEndedIterator + FusedIterator + Clone
    {
        (0..FACTORIALS[n]).into_iter().map(|x| Self(x))
    }
}

const FACTORIALS: [u16; 9] = {
    let mut arr = [1; 9];

    let mut index = 2u16;
    let mut current = 1;
    while index < arr.len() as u16 {
        current *= index;
        arr[index as usize] = current;
        index += 1;
    }
    arr
};

#[derive(Debug, Clone, PartialEq)]
struct SwapsIter8 {
    index: NonZeroU8,
    inner: u16,
}

/// A swap of two elements.
/// Swaps are delivered in a sequence. The nth swap should be swapped with element `index`
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Swap {
    pub index: u8,
}

impl Swap {
    pub fn apply<T>(self, index: usize, arr: &mut [T]) {
        arr.swap(self.index as usize, index);
    }
}

impl std::iter::FusedIterator for SwapsIter8 {}

impl std::iter::Iterator for SwapsIter8 {
    type Item = Swap;

    fn next(&mut self) -> Option<Self::Item> {
        if self.inner == 0 {
            return None;
        }

        let i = (self.inner % self.index.get() as u16) as u8;
        self.inner /= self.index.get() as u16;

        let result = Some(Swap {
            index: self.index.get() - (i + 1),
        });
        self.index = self.index.saturating_add(1);
        result
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use crate::graph_permutation8::GraphPermutation8;

    #[test]
    pub fn test_sequence_for_5_elements() {
        let mut data = String::new();
        let arr = [0, 1, 2, 3, 4];
        for perm in GraphPermutation8::all_for_n_elements(5) {
            let mut arr = arr.clone();
            perm.apply_to_slice(&mut arr);
            use std::fmt::Write;
            writeln!(data, "{}", arr.into_iter().join(",")).unwrap();
        }

        //This should permute the first four elements and leave the last four in the same order
        insta::assert_snapshot!(data);
    }

    #[test]
    pub fn test_swaps_roundtrip() {
        for perm in GraphPermutation8::all_for_n_elements(5) {
            let swaps = perm.swaps();
            let perm2 = GraphPermutation8::try_from_swaps(swaps)
                .expect("Should be able to make iter from swaps");

            if perm != perm2 {
                panic!(
                    "Swaps don't roundtrip: {perm:?} != {perm2:?} (perm swaps {})",
                    perm.swaps().map(|x| x.index.to_string()).join(", ")
                );
            }
        }
    }
}
