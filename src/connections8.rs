use std::{
    fmt::Display,
    ops::{Deref, DerefMut},
};

use const_sized_bit_set::{bit_set_trait::BitSetTrait, BitSet32, BitSet8};

use crate::{graph8::Graph8, NodeIndex};

#[derive(Debug, PartialEq, Clone, Copy, PartialOrd, Eq, Ord, Hash, Default)]
pub struct Connections8 {
    set: BitSet32,
}

impl Deref for Connections8 {
    type Target = BitSet32;

    fn deref(&self) -> &Self::Target {
        &self.set
    }
}

impl DerefMut for Connections8 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.set
    }
}

impl Extend<ConnectionKey> for Connections8 {
    fn extend<T: IntoIterator<Item = ConnectionKey>>(&mut self, iter: T) {
        for x in iter {
            self.insert(x);
        }
    }
}

impl FromIterator<ConnectionKey> for Connections8 {
    fn from_iter<T: IntoIterator<Item = ConnectionKey>>(iter: T) -> Self {
        let mut s = Self::default();
        s.extend(iter);
        s
    }
}

impl Connections8 {
    pub const EMPTY: Self = Connections8 {
        set: BitSet32::EMPTY,
    };

    pub const ALL: Self = Connections8 {
        set: BitSet32::from_first_n_const(28),
    };

    pub const fn inner(&self) -> u32 {
        self.set.inner_const()
    }

    pub const fn from_inner_unchecked(inner: u32)-> Self{
        Self{set: BitSet32::from_inner_const(inner)}
    }

    pub fn to_graph(self) -> Graph8 {
        let mut graph = Graph8::EMPTY;
        for key in self.iter() {
            let (a, b) = key.to_indexes();
            graph.insert(a, b);
        }
        graph
    }

    pub fn insert(&mut self, key: impl Into<ConnectionKey>) {
        let key: ConnectionKey = key.into();
        debug_assert!(
            key.0 < 64,
            "Connection key inner is {} but should be < 64",
            key.0
        );
        self.set.insert_const(key.0 as u32);
    }

    pub fn from_graph(graph: &Graph8) -> Self {
        //todo const
        let mut bits_used: u32 = 0;
        let mut set = 0u32;
        for index in 1..graph.active_nodes() {
            let mut adj = graph.adjacencies[index];
            adj.intersect_with_const(&BitSet8::from_first_n_const(index as u32));

            let mut adj = adj.inner_const() as u32;

            adj <<= bits_used;
            set |= adj;
            bits_used += index as u32;
        }

        Self {
            set: BitSet32::from_inner(set),
        }
    }

    pub fn iter(
        &self,
    ) -> impl Iterator<Item = ConnectionKey>
           + DoubleEndedIterator
           + ExactSizeIterator
           + std::iter::FusedIterator
           + Clone {
        self.set.into_iter().map(|x| ConnectionKey(x as u8))
    }
}

impl Display for Connections8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for ck in self.iter() {
            let (a, b) = ck.to_indexes();
            use std::fmt::Write;
            if first {
                first = false
            } else {
                f.write_char(',')?;
            }
            write!(f, "{}{}", a.0, b.0)?;
        }

        Ok(())
    }
}

impl From<Graph8> for Connections8 {
    fn from(value: Graph8) -> Self {
        Self::from_graph(&value)
    }
}

impl From<Connections8> for Graph8 {
    fn from(value: Connections8) -> Self {
        value.to_graph()
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct ConnectionKey(u8);

impl ConnectionKey {

    const KEYS_TO_PAIRS: [(u8, u8); 28] = {
        let mut keys_to_pairs = [(0, 0); 28];
        let mut right = 1;
        let mut key = 0;
        while right < 8 {
            let mut left = 0;
            while left < right {
                keys_to_pairs[key] = (left, right);
                left += 1;
                key += 1;
            }
            right += 1;
        }
        keys_to_pairs
    };

    const PAIRS_TO_KEYS: [u8; 64] = {
        //NOTE key 31 indicates an invalid value where l == r
        let mut pairs_to_keys = [31; 64];

        let mut key = 0u8;

        while key < Self::KEYS_TO_PAIRS.len() as u8 {
            let (left, right) = Self::KEYS_TO_PAIRS[key as usize];

            let lr_pair_index = ((left << 3) + right) as usize;
            let rl_pair_index = ((right << 3) + left) as usize;
            pairs_to_keys[rl_pair_index] = key;
            pairs_to_keys[lr_pair_index] = key;
            key += 1;
        }

        pairs_to_keys
    };

    pub fn from_indexes(left: NodeIndex, right: NodeIndex) -> Self {
        let index = (left.0 << 3) + right.0;
        ConnectionKey(Self::PAIRS_TO_KEYS[index as usize])
    }

    pub fn to_indexes(self) -> (NodeIndex, NodeIndex) {
        let (a, b) = Self::KEYS_TO_PAIRS[self.0 as usize];

        (NodeIndex(a), NodeIndex(b))
    }
}

impl From<(NodeIndex, NodeIndex)> for ConnectionKey {
    fn from(value: (NodeIndex, NodeIndex)) -> Self {
        ConnectionKey::from_indexes(value.0, value.1)
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use const_sized_bit_set::bit_set_trait::BitSetTrait;

    use crate::{graph8::Graph8, NodeIndex};

    #[test]
    pub fn test_insert() {
        let graph = Graph8::from_str("01,02,23").unwrap();
        let mut cs: crate::connections8::Connections8 = graph.into();

        assert!(!cs.is_empty());
        assert_eq!(cs.to_string(), "01,02,23");

        cs.insert((NodeIndex(1), NodeIndex(2)));

        assert_eq!(cs.to_string(), "01,02,12,23");

        let graph2: Graph8 = cs.into();

        assert_eq!(graph2.to_string(), "01,02,12,23");
    }
}
