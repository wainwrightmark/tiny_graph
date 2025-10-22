use const_sized_bit_set::BitSet8;
use crate::{util, EIGHT};
use std::{iter::FusedIterator, num::NonZeroU8};

#[cfg(any(test, feature = "serde"))]
use serde::{Deserialize, Serialize};

#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(any(test, feature = "serde"), derive(Serialize, Deserialize), serde(transparent))]
pub struct GraphPermutation8(u16);

impl Default for GraphPermutation8 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl GraphPermutation8 {
    /// The identity permutation which does nothing
    pub const IDENTITY: GraphPermutation8 = GraphPermutation8(0);

    /// Permutations made by just swapping element `0` with element `n`
    pub const SWAP_0_N: [GraphPermutation8; EIGHT] = {
        let mut result = [GraphPermutation8::IDENTITY; EIGHT];
        let mut index = 1; //can skip the first as it is already the identity

        while index < EIGHT {
            let mut swaps = Swap::array_n();
            swaps[index] = Swap { index: 0 };

            let gp = GraphPermutation8::try_from_swaps_arr(swaps).unwrap();
            result[index] = gp;

            index += 1;
        }

        result
    };

    //todo ROTATE_LEFT and ROTATE_RIGHT

    pub const fn is_identity(&self) -> bool {
        self.0 == Self::IDENTITY.0
    }

    pub const fn inner(&self) -> u16 {
        self.0
    }

    #[allow(dead_code)]
    pub(crate) fn from_inner(value: u16) -> Self {
        debug_assert!(value < FACTORIALS[8]);
        Self(value)
    }

    pub const fn swaps(self) -> impl Clone + FusedIterator<Item = Swap> {
        SwapsIter8::new(&self)
    }

    pub const fn swaps_array(&self) -> [Swap; EIGHT] {
        let mut swaps = Swap::array_n();
        let mut swaps_iter = SwapsIter8::new(&self);
        let mut i = 0;

        while let Some(swap) = swaps_iter.next_const() {
            swaps[i] = swap;
            i += 1;
        }
        swaps
    }

    pub const fn try_from_swaps_arr(swaps: [Swap; EIGHT]) -> Option<Self> {
        let mut inner = 0u16;
        let mut multiplier = 1u16;

        let mut index = 0;
        while index < EIGHT {
            let swap = swaps[index];
            let s = swap.index as u16;

            let diff = match (index as u16).checked_sub(s) {
                Some(diff) => diff,
                None => return None,
            };
            if index > 0 {
                multiplier *= index as u16;
            }
            inner += diff * multiplier;
            index += 1;
        }
        Some(Self(inner))
    }

    pub fn try_from_swaps_iter(swaps: impl Iterator<Item = Swap>) -> Option<Self> {
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

    /// Try to calculate a permutation from an array
    /// The elements of the array must be distinct and all less than or equal to 8
    pub fn calculate(arr: &mut [u8]) -> Option<Self> {
        let mut indexes_found = BitSet8::EMPTY;

        for x in arr.iter() {
            if *x as usize >= EIGHT {
                return None;
            }
            if !indexes_found.insert_const(*x as u32) {
                return None;
            }
        }

        Self::calculate_unchecked(arr)
    }

    pub(crate) fn calculate_unchecked(mut arr: &mut [u8]) -> Option<Self> {        
        let mut swaps = Swap::array_n();

        while let Some(max_index) = crate::util::position_max(arr) {
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
    ) -> impl ExactSizeIterator<Item = Self> + DoubleEndedIterator + FusedIterator + Clone {
        (0..FACTORIALS[n]).map(Self)
    }

    /// Get the complete array of this permutation's elements
    pub fn get_array(&self) -> [u8; EIGHT] {
        let mut arr = util::indexes_array();
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

        let mut inverse = util::indexes_array();

        for (index, element) in arr.into_iter().enumerate() {
            inverse[element as usize] = index as u8;
        }

        if inverse == arr {
            return *self;
        }
        Self::calculate_unchecked(&mut inverse).unwrap()
    }

    /// Generate the cycle with this permutation as the operator
    pub fn generate_cycle(self) -> impl Iterator<Item = Self> {
        CyclicGenerator8::from(self)
    }

    /// Moves an existing element to a new index
    pub fn with_element_moved(self, previous_index: usize, new_index: usize) -> Self {
        if previous_index == new_index {
            return self;
        }

        let mut arr = self.get_array();

        if previous_index < new_index {
            arr[previous_index..=new_index].rotate_left(1);
        } else {
            arr[new_index..=previous_index].rotate_right(1);
        }

        Self::calculate_unchecked(&mut arr).unwrap()
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

/// A swap of two elements.
/// Swaps are delivered in a sequence. The nth swap should be swapped with element `index`
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Swap {
    pub index: u8,
}

impl Swap {
    pub const fn apply<T>(self, index: usize, arr: &mut [T]) {
        arr.swap(self.index as usize, index);
    }

    
    // pub const ARRAY8: [Swap; 8] = Self::array_n();

    // pub const ARRAY14: [Swap; 14] = Self::array_n();

    /// Identity Array mapping each index to itself
    pub const fn array_n<const N: usize>()-> [Swap; N]{
        let mut array = [Swap{index: 0}; N];
        let mut index = 1;
        while index < N{
            array[index] = Swap { index: index as u8 };
            index+= 1;
        }

        array
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SwapsIter8 {
    index: NonZeroU8,
    inner: u16,
}

impl std::iter::FusedIterator for SwapsIter8 {}

impl std::iter::Iterator for SwapsIter8 {
    type Item = Swap;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_const()
    }
}

impl SwapsIter8 {
    pub const fn new(permutation: &GraphPermutation8) -> Self {
        Self {
            index: std::num::NonZeroU8::MIN,
            inner: permutation.0,
        }
    }

    pub const fn next_const(&mut self) -> Option<Swap> {
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

/// An iterator of the cyclic group generated by a particular permutation.
/// The length of this iterator will be the least common multiple of the lengths of the operators cycles
#[derive(Clone, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[must_use]
pub(crate) struct CyclicGenerator8 {
    operator: GraphPermutation8,
    next: Option<GraphPermutation8>,
}

impl From<GraphPermutation8> for CyclicGenerator8 {
    fn from(perm: GraphPermutation8) -> Self {
        Self {
            operator: perm,
            next: Some(perm),
        }
    }
}

impl Iterator for CyclicGenerator8 {
    type Item = GraphPermutation8;

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.next?;

        if n.is_identity() {
            self.next = None;
        } else {
            self.next = Some(n.combine(&self.operator));
        }

        Some(n)
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
        for perm in GraphPermutation8::all_for_n_elements(5) {
            let swaps = perm.swaps();
            let perm2 = GraphPermutation8::try_from_swaps_iter(swaps)
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
        for perm in GraphPermutation8::all_for_n_elements(5) {
            let swaps = perm.swaps_array();
            let perm2 = GraphPermutation8::try_from_swaps_arr(swaps)
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

            let perm = GraphPermutation8::calculate_unchecked(&mut vec1)
                .expect("Should be able to calculate permutation");

            let mut slice = (0u8..8).take(arr.len()).collect::<Vec<_>>();

            perm.apply_to_slice(&mut slice);

            assert_eq!(arr, slice.as_slice())
        }

        assert_eq!(
            GraphPermutation8::calculate_unchecked(&mut [0, 1, 2, 3, 4, 5, 6, 7]),
            Some(GraphPermutation8::IDENTITY)
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
        let perm1 = GraphPermutation8::calculate_unchecked(&mut [0, 2, 4, 6, 1, 3, 5, 7]).unwrap();
        let perm2 = GraphPermutation8::calculate_unchecked(&mut [3, 0, 1, 2, 4, 5, 6, 7]).unwrap();

        let perm1_then_2 = perm1.combine(&perm2);

        assert_eq!(perm1_then_2.get_array(), [6, 0, 2, 4, 1, 3, 5, 7]);

        let perm2_then_1 = perm2.combine(&perm1);

        assert_eq!(perm2_then_1.get_array(), [3, 1, 4, 6, 0, 2, 5, 7]);
    }

    #[test]
    pub fn test_cycle() {
        let perm = GraphPermutation8::calculate_unchecked(&mut [7, 0, 1, 2, 3, 4, 5, 6]).unwrap();

        let cycle = perm.generate_cycle().collect_vec();

        let cycle_str = cycle
            .into_iter()
            .map(|x| x.get_array().into_iter().join(","))
            .join("\n");

        assert_eq!(cycle_str, "7,0,1,2,3,4,5,6\n6,7,0,1,2,3,4,5\n5,6,7,0,1,2,3,4\n4,5,6,7,0,1,2,3\n3,4,5,6,7,0,1,2\n2,3,4,5,6,7,0,1\n1,2,3,4,5,6,7,0\n0,1,2,3,4,5,6,7")
    }

    #[test]
    pub fn test_invert() {
        for arr in GraphPermutation8::all_for_n_elements(8) {
            let inverse = arr.invert();

            assert_eq!(arr.combine(&inverse), GraphPermutation8::IDENTITY);
            assert_eq!(inverse.combine(&arr), GraphPermutation8::IDENTITY)
        }
    }

    #[test]
    pub fn test_small_swaps() {
        let mut arr = [0, 1, 2, 3, 4, 5, 6, 7];
        for perm in GraphPermutation8::SWAP_0_N.iter() {
            perm.apply_to_slice(&mut arr);
        }

        assert_eq!(arr, [7, 0, 1, 2, 3, 4, 5, 6])
    }

    #[test]
    pub fn test_move_element() {
        let mut perm = GraphPermutation8::IDENTITY;

        perm = perm.with_element_moved(0, 1);
        assert_eq!(
            perm.get_array(),
            [1, 0, 2, 3, 4, 5, 6, 7]
        );

        perm = perm.with_element_moved(5, 2);
        assert_eq!(
            perm.get_array(),
            [1, 0, 5, 2, 3, 4, 6, 7]
        );
        perm = perm.with_element_moved(4, 7);
        assert_eq!(
            perm.get_array(),
            [1, 0, 5, 2, 4, 6, 7, 3]
        );
    }
}
