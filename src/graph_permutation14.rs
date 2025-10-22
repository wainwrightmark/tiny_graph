use crate::{graph_permutation8::Swap, FOURTEEN};
use const_sized_bit_set::BitSet16;
use std::{iter::FusedIterator, num::NonZeroU8};

#[cfg(any(test, feature = "serde"))]
use serde::{Deserialize, Serialize};


/// Graph permutation of up to 14 elements
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(any(test, feature = "serde"), derive(Serialize, Deserialize), serde(transparent))]
pub struct GraphPermutation14(u64);

impl Default for GraphPermutation14 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl GraphPermutation14 {
    /// The identity permutation which does nothing
    pub const IDENTITY: GraphPermutation14 = GraphPermutation14(0);

    /// Permutations made by just swapping element `0` with element `n`
    pub const SWAP_0_N: [GraphPermutation14; FOURTEEN] = {
        let mut result = [Self::IDENTITY; FOURTEEN];
        let mut index = 1; //can skip the first as it is already the identity

        while index < FOURTEEN {
            let mut swaps = Swap::array_n();
            swaps[index] = Swap { index: 0 };

            let gp = Self::try_from_swaps_arr(swaps).unwrap();
            result[index] = gp;

            index += 1;
        }

        result
    };

    //todo ROTATE_LEFT and ROTATE_RIGHT

    pub const fn is_identity(&self) -> bool {
        self.0 == Self::IDENTITY.0
    }

    pub const fn inner(&self) -> u64 {
        self.0
    }

    #[allow(dead_code)]
    pub(crate) fn from_inner(value: u64) -> Self {
        debug_assert!(value < FACTORIALS[8]);
        Self(value)
    }

    pub const fn swaps(self) -> impl Clone + FusedIterator<Item = Swap> {
        SwapsIter14::new(&self)
    }

    pub const fn swaps_array(&self) -> [Swap; FOURTEEN] {
        let mut swaps = Swap::array_n();
        let mut swaps_iter = SwapsIter14::new(&self);
        let mut i = 0;

        while let Some(swap) = swaps_iter.next_const() {
            swaps[i] = swap;
            i += 1;
        }
        swaps
    }

    pub const fn try_from_swaps_arr(swaps: [Swap; FOURTEEN]) -> Option<Self> {
        let mut inner = 0u64;
        let mut multiplier = 1u64;

        let mut index = 0;
        while index < FOURTEEN {
            let swap = swaps[index];
            let s = swap.index as u64;

            let diff = match (index as u64).checked_sub(s) {
                Some(diff) => diff,
                None => return None,
            };
            if index > 0 {
                multiplier *= index as u64;
            }
            inner += diff * multiplier;
            index += 1;
        }
        Some(Self(inner))
    }

    pub fn try_from_swaps_iter(swaps: impl Iterator<Item = Swap>) -> Option<Self> {
        let mut inner = 0u64;
        let mut multiplier = 1;
        for (index, swap) in swaps.enumerate() {
            let index = index as u64;
            let s = swap.index as u64;

            let diff = index.checked_sub(s)?;
            multiplier *= index.max(1);
            inner += diff * multiplier;
        }
        Some(Self(inner))
    }

    /// Try to calculate a permutation from an array
    /// The elements of the array must be distinct and all less than or equal to 8
    pub fn calculate(arr: &mut [u8]) -> Option<Self> {
        let mut indexes_found = BitSet16::EMPTY;

        for x in arr.iter() {
            if *x as usize >= FOURTEEN {
                return None;
            }
            if !indexes_found.insert_const(*x as u32) {
                return None;
            }
        }

        Self::calculate_unchecked(arr)
    }

    pub(crate) fn calculate_unchecked(mut arr: &mut [u8]) -> Option<Self> {
        fn position_max(arr: &[u8]) -> Option<usize> {
            let mut max = *arr.first()?;
            let mut max_index = 0;

            for (x, item) in arr.iter().copied().enumerate().skip(1) {
                if item > max {
                    max = item;
                    max_index = x;
                }
            }

            Some(max_index)
        }

        let mut swaps = Swap::array_n();

        while let Some(max_index) = position_max(arr) {
            let index = arr.len() - 1;
            swaps[index] = Swap {
                index: max_index as u8,
            };

            arr.swap(index, max_index);

            let (_, last) = arr.split_last_mut().unwrap();
            arr = last;
        }

        Self::try_from_swaps_arr(swaps)
    }

    pub fn apply_to_slice<T>(self, arr: &mut [T]) {
        for (index, swap) in self.swaps().enumerate() {
            swap.apply(index, arr);
        }
    }

    pub fn all_for_n_elements(
        n: usize,
    ) -> impl Iterator<Item = Self> + DoubleEndedIterator + FusedIterator + Clone {
        (0..FACTORIALS[n]).map(Self)
    }

    /// Get the complete array of this permutation's elements
    pub fn get_array(&self) -> [u8; 8] {
        let mut arr = [0, 1, 2, 3, 4, 5, 6, 7];
        self.apply_to_slice(&mut arr);
        arr
    }

    /// Combine this permutation with another. Producing a permutation equivalent to performing this and then the other.
    /// Note that this operation is neither commutative nor associative
    pub fn combine(&self, rhs: &Self) -> Self {
        let mut arr = self.get_array();
        rhs.apply_to_slice(&mut arr);

        Self::calculate_unchecked(&mut arr).unwrap()
    }

    /// Invert this permutation
    /// This produces the permutation that will reorder the array back to its original order
    pub fn invert(&self) -> Self {
        let arr = self.get_array();

        let mut inverse = [0, 1, 2, 3, 4, 5, 6, 7];

        for (index, element) in arr.into_iter().enumerate() {
            inverse[element as usize] = index as u8;
        }

        if inverse == arr {
            return *self;
        }
        Self::calculate_unchecked(&mut inverse).unwrap()
    }

    // /// Generate the cycle with this permutation as the operator
    // pub fn generate_cycle(self) -> impl Iterator<Item = Self> {
    //     CyclicGenerator8::from(self)
    // }
}

const FACTORIALS: [u64; 15] = {
    let mut arr = [1; 15];

    let mut index = 2u64;
    let mut current = 1;
    while index < arr.len() as u64 {
        current *= index;
        arr[index as usize] = current;
        index += 1;
    }
    arr
};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SwapsIter14 {
    index: NonZeroU8,
    inner: u64,
}

impl std::iter::FusedIterator for SwapsIter14 {}

impl std::iter::Iterator for SwapsIter14 {
    type Item = Swap;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_const()
    }
}

impl SwapsIter14 {
    pub const fn new(permutation: &GraphPermutation14) -> Self {
        Self {
            index: std::num::NonZeroU8::MIN,
            inner: permutation.0,
        }
    }

    pub const fn next_const(&mut self) -> Option<Swap> {
        if self.inner == 0 {
            return None;
        }

        let i = (self.inner % self.index.get() as u64) as u8;
        self.inner /= self.index.get() as u64;

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

    use crate::graph_permutation14::GraphPermutation14;

    #[test]
    pub fn test_sequence_for_5_elements() {
        let mut data = String::new();
        let arr = [0, 1, 2, 3, 4];
        for perm in GraphPermutation14::all_for_n_elements(5) {
            let mut arr = arr;
            perm.apply_to_slice(&mut arr);
            use std::fmt::Write;
            writeln!(data, "{}", arr.into_iter().join(",")).unwrap();
        }

        //This should permute the first four elements and leave the last four in the same order
        insta::assert_snapshot!(data);
    }

    #[test]
    pub fn test_swaps_roundtrip1() {
        for perm in GraphPermutation14::all_for_n_elements(5) {
            let swaps = perm.swaps();
            let perm2 = GraphPermutation14::try_from_swaps_iter(swaps)
                .expect("Should be able to make iter from swaps");

            if perm != perm2 {
                panic!(
                    "Swaps don't roundtrip: {perm:?} != {perm2:?} (perm swaps {})",
                    perm.swaps().map(|x| x.index.to_string()).join(", ")
                );
            }
        }
    }

    #[test]
    pub fn test_swaps_roundtrip2() {
        for perm in GraphPermutation14::all_for_n_elements(5) {
            let swaps = perm.swaps_array();
            let perm2 = GraphPermutation14::try_from_swaps_arr(swaps)
                .expect("Should be able to make iter from swaps");

            if perm != perm2 {
                panic!(
                    "Swaps don't roundtrip: {perm:?} != {perm2:?} (perm swaps {})",
                    perm.swaps().map(|x| x.index.to_string()).join(", ")
                );
            }
        }
    }

    #[test]
    pub fn test_calculate_unchecked() {
        fn test_calculate_roundtrip(arr: &[u8]) {
            let mut vec1 = Vec::from_iter(arr.iter().copied());

            let perm = GraphPermutation14::calculate_unchecked(&mut vec1)
                .expect("Should be able to calculate permutation");

            let mut slice = (0u8..8).take(arr.len()).collect::<Vec<_>>();

            perm.apply_to_slice(&mut slice);

            assert_eq!(arr, slice.as_slice())
        }

        assert_eq!(
            GraphPermutation14::calculate_unchecked(&mut [0, 1, 2, 3, 4, 5, 6, 7]),
            Some(GraphPermutation14::IDENTITY)
        );

        test_calculate_roundtrip(&[]);
        test_calculate_roundtrip(&[0]);
        test_calculate_roundtrip(&[0, 1]);
        test_calculate_roundtrip(&[0, 1, 2, 3, 4, 5]);

        test_calculate_roundtrip(&[1, 0]);

        test_calculate_roundtrip(&[0, 2, 4, 6, 1, 3, 5, 7]);
    }

    #[test]
    pub fn test_combine() {
        let perm1 = GraphPermutation14::calculate_unchecked(&mut [0, 2, 4, 6, 1, 3, 5, 7]).unwrap();
        let perm2 = GraphPermutation14::calculate_unchecked(&mut [3, 0, 1, 2, 4, 5, 6, 7]).unwrap();

        let perm1_then_2 = perm1.combine(&perm2);

        assert_eq!(perm1_then_2.get_array(), [6, 0, 2, 4, 1, 3, 5, 7]);

        let perm2_then_1 = perm2.combine(&perm1);

        assert_eq!(perm2_then_1.get_array(), [3, 1, 4, 6, 0, 2, 5, 7]);
    }

    #[test]
    pub fn test_invert() {
        for arr in GraphPermutation14::all_for_n_elements(8) {
            let inverse = arr.invert();

            assert_eq!(arr.combine(&inverse), GraphPermutation14::IDENTITY);
            assert_eq!(inverse.combine(&arr), GraphPermutation14::IDENTITY)
        }
    }

    #[test]
    pub fn test_small_swaps() {
        let mut arr = [0, 1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13];
        for perm in GraphPermutation14::SWAP_0_N.iter() {
            perm.apply_to_slice(&mut arr);
        }

        assert_eq!(arr, [13, 0, 1, 2, 3, 4, 5, 6,7,8,9,10,11,12])
    }
}
