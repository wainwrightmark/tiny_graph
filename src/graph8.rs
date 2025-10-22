use std::cell::RefCell;
use std::fmt::Display;
use std::iter::FusedIterator;

use std::collections::BTreeMap;
use std::num::ParseIntError;
use std::str::FromStr;

use const_sized_bit_set::bit_set_trait::BitSetTrait;
use const_sized_bit_set::{BitSet64, BitSet8};

use crate::connections8::Connections8;
use crate::graph_path_iter::{GraphPath8, GraphPathIter};
use crate::graph_permutation8::{GraphPermutation8, Swap};
use crate::symmetries8::Symmetries8;
use crate::{NodeIndex, EIGHT};

/// A graph with up to 8 nodes
#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
#[must_use]
pub struct Graph8 {
    pub(crate) inner: BitSet64,
}

// impl Deref for Graph8 {
//     type Target = [BitSet8; EIGHT];

//     fn deref(&self) -> &Self::Target {
//         &self.adjacencies
//     }
// }

impl Graph8 {
    pub const EMPTY: Self = Self {
        inner: BitSet64::EMPTY,
    };

    pub const ALL: Self = Self {
        inner: BitSet64::from_inner_const(0x7fbfdfeff7fbfdfe),
    };

    #[inline]
    /// A graph is not valid if any of its nodes are connected to themselves
    pub const fn is_valid(&self) -> bool {
        self.inner.is_subset_const(&Self::ALL.inner)
    }

    #[inline]
    pub const fn fully_connected(n: usize) -> Self {
        const FC: [u64; 9] = [
            0x0,
            0x0, //One fully connected node still has no connections
            0x102,
            0x30506,
            0x70b0d0e,
            0xf171b1d1e,
            0x1f2f373b3d3e,
            0x3f5f6f777b7d7e,
            0x7fbfdfeff7fbfdfe,
        ];

        Self {
            inner: BitSet64::from_inner_const(FC[n]),
        }
    }

    #[inline]
    pub const fn from_inner_unchecked(inner: BitSet64) -> Self {
        Self { inner }
    }

    #[inline]
    pub const fn insert(&mut self, a: NodeIndex, b: NodeIndex) {
        self.inner.insert_const(((a.0 * 8) + b.0) as u32);
        self.inner.insert_const(((b.0 * 8) + a.0) as u32);
    }

    #[inline]
    /// The number of active nodes.
    /// A node is active if it has connections or if a node with greater index has connections
    pub const fn active_nodes(&self) -> usize {
        8 - (self.inner.inner_const().leading_zeros() as usize / 8)
    }

    #[inline]
    /// Return an iterator of permutations that map `sub` to a subgraph of `self`
    /// TODO cache results of this
    pub fn iter_subgraph_permutations(
        self,
        potential_subgraph: &Self,
    ) -> impl DoubleEndedIterator<Item = GraphPermutation8> {
        let potential_subgraph = potential_subgraph.clone();
        let active_nodes = self.active_nodes();

        GraphPermutation8::all_for_n_elements(active_nodes).filter(move |permutation| {
            let mut clone = potential_subgraph.clone();
            clone.apply_permutation(*permutation);

            self.is_super_graph(&clone)
        })
    }

    #[inline]
    ///Is this a super graph of a subgraph
    pub const fn is_super_graph(&self, subgraph: &Self) -> bool {
        self.inner.is_superset_const(&subgraph.inner)
    }

    #[inline]
    /// Whether this could be a super graph based numbers of connections
    pub fn could_be_super_graph(&self, subgraph: &Self) -> bool {
        struct Counts {
            min_connections: u32,
            max_connections: u32,
            total_connections: u32,
        }

        fn calculate_counts(sets: &[u8]) -> Counts {
            let mut min_connections: u32 = u32::MAX;
            let mut max_connections: u32 = 0;
            let mut total_connections: u32 = 0;
            for x in sets {
                let c = x.count_ones();
                total_connections += c;
                min_connections = min_connections.min(c);
                max_connections = max_connections.max(c);
            }

            Counts {
                min_connections,
                max_connections,
                total_connections,
            }
        }

        let self_counts = calculate_counts(&self.inner.inner_const().to_ne_bytes());
        let subgraph_counts = calculate_counts(&subgraph.inner.inner_const().to_ne_bytes());

        match self_counts
            .total_connections
            .cmp(&subgraph_counts.total_connections)
        {
            std::cmp::Ordering::Less => false,
            std::cmp::Ordering::Equal => {
                self_counts.min_connections == subgraph_counts.min_connections
                    && self_counts.max_connections == subgraph_counts.max_connections
            }
            std::cmp::Ordering::Greater => {
                self_counts.min_connections >= subgraph_counts.min_connections
                    && self_counts.max_connections >= subgraph_counts.max_connections
            }
        }
    }

    #[inline]
    pub const fn is_unchanged_under_permutation(&self, permutation: GraphPermutation8) -> bool {
        let mut clone = Graph8 { inner: self.inner };
        //todo is there a faster way?
        clone.apply_permutation(permutation);
        self.inner.inner_const() == clone.inner.inner_const()
    }

    #[inline]
    pub const fn apply_permutation(&mut self, permutation: GraphPermutation8) {
        let mut swaps_iter = crate::graph_permutation8::SwapsIter8::new(&permutation);

        let mut index = 0;

        while let Some(next) = swaps_iter.next_const() {
            self.swap_nodes(NodeIndex(index as u8), NodeIndex(next.index));
            index += 1;
        }
    }

    #[inline]
    pub fn get_symmetries(&self) -> Symmetries8 {
        Symmetries8::new_cached(self)
    }

    #[inline]
    pub fn find_mapping_permutation(mut self, other: &Self) -> Option<GraphPermutation8> {
        let mut swaps = [0, 1, 2, 3, 4, 5, 6, 7].map(|index| Swap { index });

        let mut stack = [BitSet8::EMPTY; EIGHT];

        let Some(last_index) = self.active_nodes().max(other.active_nodes()).checked_sub(1) else {
            return if *other == Graph8::EMPTY {
                Some(GraphPermutation8::IDENTITY)
            } else {
                None
            };
        };
        let last_adj_count = other.adjacencies(last_index).len_const();

        stack[last_index] = BitSet8::from_iter(
            self.inner
                .inner_const()
                .to_le_bytes()
                .iter()
                .map(|x| BitSet8::from_inner_const(*x))
                .enumerate()
                .filter(|(_index, set)| set.len_const() == last_adj_count)
                .map(|x| x.0 as u32),
        );

        //println!("Self: {self}. Other: {other}");

        //println!("Self: {}", self.adjacencies.map(|x|format!("{x:b}")) .join(","));
        //println!("Other: {}", other.adjacencies.map(|x|format!("{x:b}")) .join(","));

        //println!("Last Index {last_index}. Last adj count {last_adj_count}");
        let mut index: usize = last_index;

        loop {
            let top = stack.get_mut(index).unwrap();

            //println!("Index {index} Top: {top:?}");

            let Some(next_adj) = top.pop_last_const() else {
                index += 1;
                if index > last_index {
                    //println!("Giving up");
                    return None;
                }

                //println!("Reverting swap {index} and {}",swaps[index].index);

                self.swap_nodes(NodeIndex(index as u8), NodeIndex(swaps[index].index));

                continue;
            };

            //println!("Swapping {index} and {next_adj}");
            swaps[index] = Swap {
                index: next_adj as u8,
            };
            self.swap_nodes(NodeIndex(index as u8), NodeIndex(next_adj as u8));

            let Some(next_index) = index.checked_sub(1) else {
                //println!("Finished");
                return GraphPermutation8::try_from_swaps_arr(swaps);
            };

            let other_adj = other.adjacencies(next_index);
            let other_adj_count = other_adj.len_const();
            let last_n = BitSet8::from_first_n_const(index as u32).with_negated();
            let other_intersect_last_n = other_adj.with_intersect(&last_n);

            let possible_swaps = BitSet8::from_iter(
                self.inner
                    .inner_const()
                    .to_le_bytes()
                    .iter()
                    .map(|x| BitSet8::from_inner_const(*x))
                    .enumerate()
                    .take(next_index + 1)
                    .filter(|(_index, set)| {
                        set.len_const() == other_adj_count
                            && set.with_intersect(&last_n) == other_intersect_last_n
                    })
                    .map(|(index, _set)| index as u32),
            );
            //println!("Found {possible_swaps:?} with count {other_adj_count}");
            stack[next_index] = possible_swaps;

            index = next_index;
        }
    }

    #[inline]
    pub fn adjacencies(&self, index: usize) -> BitSet8 {
        BitSet8::from_inner_const(self.inner.inner_const().to_le_bytes()[index])
    }

    #[inline]
    /// Finds the minimum connections set
    /// Uses a thread local cache
    pub fn find_min_connections_set(&self) -> Connections8 {
        thread_local! {
            static CACHE: RefCell<BTreeMap<(Connections8, u32), Connections8>>  = const{RefCell::new(BTreeMap::new())}  ;
        }

        CACHE.with(|c| self.find_min_connections_set_with_cache(&mut c.borrow_mut()))
    }

    #[inline]
    fn find_min_connections_set_with_cache(
        &self,
        cache: &mut BTreeMap<(Connections8, u32), Connections8>,
    ) -> Connections8 {
        fn find_next_target_adjacencies(graph: &Graph8, unfrozen: u32) -> BitSet8 {
            let mut possibles = BitSet8::from_first_n_const(unfrozen);
            //check frozen bits to find referenced sets
            for frozen_index in (unfrozen..8).rev() {
                // The set of nodes which are not connected to this frozen node
                let ai = graph.adjacencies(frozen_index as usize).with_negated();
                let p = possibles.with_intersect(&ai);

                if !p.is_empty() {
                    possibles = p;
                }
                if possibles.len() == 1 {
                    return possibles;
                }
            }

            //find things with the fewest elements
            possibles = possibles.min_set_by_key(|x| graph.adjacencies(x as usize).len());

            possibles
        }

        fn find_min_thin_graph_inner(
            graph: &Graph8,
            cache: &mut BTreeMap<(Connections8, u32), Connections8>,
            unfrozen: u32,
        ) -> Connections8 {
            let as_thin = graph.to_connection_set();

            let min_thin = if unfrozen == 0 {
                as_thin
            } else if let Some(cached) = cache.get(&(as_thin, unfrozen)) {
                *cached
            } else {
                let possibles = find_next_target_adjacencies(graph, unfrozen);

                possibles
                    .into_iter()
                    .map(|index| {
                        let mut graph = graph.clone();
                        graph.swap_nodes(NodeIndex((unfrozen - 1) as u8), NodeIndex(index as u8));

                        find_min_thin_graph_inner(&graph, cache, unfrozen - 1)
                    })
                    .min()
                    .unwrap_or(as_thin)
            };

            cache.insert((as_thin, unfrozen), min_thin);
            min_thin

            //graph.adjacencies
        }

        find_min_thin_graph_inner(self, cache, self.active_nodes() as u32)
    }

    #[inline]
    pub const fn swap_nodes(&mut self, i: NodeIndex, j: NodeIndex) {
        let i = i.0 as u32;
        let j = j.0 as u32;
        let mut inner = self.inner.inner_const();
        let mask1: u64 = 0xff;
        let i_masked1 = (inner >> (i * 8)) & mask1;
        let j_masked1 = (inner >> (j * 8)) & mask1;
        let x1 = i_masked1 ^ j_masked1;

        inner ^= x1 << (i * 8);
        inner ^= x1 << (j * 8);

        let mask2: u64 = 0x101010101010101;
        let i_masked2 = (inner >> i) & mask2;
        let j_masked2 = (inner >> j) & mask2;
        let x2 = i_masked2 ^ j_masked2;

        inner ^= x2 << i;
        inner ^= x2 << j;

        self.inner = BitSet64::from_inner_const(inner)
    }

    #[inline]
    pub const fn to_connection_set(&self) -> Connections8 {
        Connections8::from_graph(self)
    }

    #[inline]
    /// Counts connections between nodes.
    /// Each connection is effectively counted twice - once in each direction
    pub const fn count_connections(&self) -> u32 {
        self.inner.len_const()
    }

    #[inline]
    pub const fn negate(&self) -> Self {
        let mut inner = self.inner;
        inner.negate_const();
        inner.intersect_with_const(&Self::ALL.inner);
        Self { inner }
    }

    #[inline]
    /// Remove all connections to a particular node
    pub const fn remove(&mut self, index: usize) {
        const MASKS: [BitSet64; 8] = {
            let mut masks = [BitSet64::EMPTY; 8];

            let mask_a: u64 = 0xff;
            let mask_b: u64 = 0x101010101010101;

            let mut index = 0;
            while index < EIGHT {
                let mask_a_shifted = mask_a << (index as u32 * 8);
                let mask_b_shifted = mask_b << (index as u32);
                let inner = !(mask_a_shifted ^ mask_b_shifted);
                masks[index] = BitSet64::from_inner_const(inner);
                index += 1
            }

            masks
        };

        self.inner.intersect_with_const(&MASKS[index]);
    }

    #[inline]
    /// Iterate through graph paths.
    /// Paths will be ordered like
    ///
    /// 0
    /// 0, 1
    /// 0, 1, 2
    /// 0, 1, 2, 3
    /// 0, 1, 3
    /// 0, 1, 3, 2
    /// 0, 2
    /// etc.
    pub fn iter_paths(self) -> impl FusedIterator<Item = GraphPath8> + Clone {
        GraphPathIter::new(self)
    }
}

impl Display for Graph8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;

        for x in self.inner.into_iter() {
            let a = x / 8;
            let b = x % 8;
            if a < b {
                use std::fmt::Write;
                if first {
                    first = false
                } else {
                    f.write_char(',')?;
                }
                write!(f, "{a}{b}")?;
            }
        }

        Ok(())
    }
}

impl FromStr for Graph8 {
    type Err = ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut graph = Graph8::EMPTY;

        for x in s.split(",") {
            if !x.is_empty() {
                let (a, b) = x.split_at(1);

                let a: u8 = a.parse()?;
                let b: u8 = b.parse()?;

                graph.insert(NodeIndex(a), NodeIndex(b));
            }
        }

        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use const_sized_bit_set::{bit_set_trait::BitSetTrait, BitSet64};
    use itertools::Itertools;

    use super::Graph8;
    use crate::{graph_permutation8::GraphPermutation8, NodeIndex};
    use std::str::FromStr;

    fn parse_graph(s: &str) -> Graph8 {
        Graph8::from_str(s).unwrap()
    }

    #[test]
    fn test_parse_to_string() {
        let g = parse_graph("01,02,23");

        assert_eq!(g.to_string(), "01,02,23");
    }

    #[test]
    fn test_insert() {
        let mut g = parse_graph("01,02,23");
        g.insert(NodeIndex(1), NodeIndex(3));
        assert_eq!(g.to_string(), "01,02,13,23");
    }

    #[test]
    fn test_active_nodes() {
        assert_eq!(parse_graph("").active_nodes(), 0);
        assert_eq!(parse_graph("01").active_nodes(), 2);
        assert_eq!(parse_graph("02").active_nodes(), 3);
        assert_eq!(parse_graph("01,02").active_nodes(), 3);
        assert_eq!(parse_graph("01,02,07").active_nodes(), 8);
    }

    #[test]
    fn test_subgraph_permutations() {
        let supergraph = parse_graph("01,02,12,13,23");
        let subgraph = parse_graph("01,02,13");

        let subgraph_permutations: Vec<_> =
            supergraph.iter_subgraph_permutations(&subgraph).collect();

        let mut data = String::new();

        for permutation in subgraph_permutations {
            let mut subgraph = subgraph.clone();
            subgraph.apply_permutation(permutation);
            use std::fmt::Write;
            writeln!(data, "{}", subgraph).unwrap();
        }

        insta::assert_snapshot!(data);
    }

    #[test]
    fn test_is_supergraph() {
        let supergraph = parse_graph("01,02,13,23");
        let subgraph13 = parse_graph("01,02,13");
        let subgraph23 = parse_graph("01,02,23");

        assert!(supergraph.is_super_graph(&subgraph13));
        assert!(supergraph.is_super_graph(&subgraph23));
        assert!(!subgraph13.is_super_graph(&subgraph23));
    }

    #[test]
    fn test_is_unchanged_under_permutations() {
        let g1 = parse_graph("01,03,13");
        let g2 = parse_graph("01,02,23");

        let swap_01 = GraphPermutation8::from_inner(1);

        assert!(g1.is_unchanged_under_permutation(swap_01));
        assert!(!g2.is_unchanged_under_permutation(swap_01));
    }

    #[test]
    fn test_apply_permutation() {
        let mut graph = parse_graph("01,02,23");

        let swap_01 = GraphPermutation8::from_inner(1);
        graph.apply_permutation(swap_01);
        assert_eq!(graph.to_string(), "01,12,23");
    }

    #[test]
    fn test_find_mapping_permutation_success() {
        let g1 = parse_graph("01,02,12");
        let g2 = parse_graph("01,03,13");

        let permutation = g1
            .find_mapping_permutation(&g2)
            .expect("Should be able to find swaps");

        let swaps = permutation.swaps().map(|x| x.index).join(",");
        // Swap nodes 2 and 3
        assert_eq!(swaps, "0,1,2,2")
    }

    #[test]
    fn test_find_mapping_permutation_success2() {
        let g1 = parse_graph("01,02,12,23");
        let g2 = parse_graph("01,12,03,13");

        let permutation = g1
            .find_mapping_permutation(&g2)
            .expect("Should be able to find swaps");

        let swaps = permutation.swaps().map(|x| x.index).join(",");
        assert_eq!(swaps, "0,1,1,1")
    }

    #[test]
    fn test_find_mapping_permutation_fail() {
        let g1 = parse_graph("01,02,13");
        let g2 = parse_graph("01,03,13");

        let swaps = g1.find_mapping_permutation(&g2);

        // No possible solution
        assert_eq!(swaps, None)
    }

    //todo more advanced test for mapping swaps

    #[test]
    fn test_to_connection_set() {
        let g1 = parse_graph("01,02,12");
        let connection_set = g1.to_connection_set();
        let fg = connection_set.to_graph();

        assert_eq!(g1, fg)
    }

    #[test]
    fn test_min_thin_connections_set() {
        let g1 = parse_graph("01,02,12");
        let g2 = parse_graph("01,03,13");

        let mg1 = g1.find_min_connections_set();
        let mg2 = g2.find_min_connections_set();

        assert_eq!(mg1, mg2, "{} {}", mg1, mg2);
        assert_eq!(mg1.inner(), 7);

        assert_eq!(mg2.to_graph(), g1);
    }

    #[test]
    fn test_min_thin_connections_set2() {
        let g1 = parse_graph("12,14,15");
        let g2 = parse_graph("01,02,03");

        let mg1 = g1.find_min_connections_set();
        let mg2 = g2.find_min_connections_set();

        assert_eq!(mg1, mg2, "{} {}", mg1, mg2);
        assert_eq!(mg1.inner(), 11);

        assert_eq!(mg1.to_graph(), g2);
    }

    #[test]
    fn test_swap_nodes() {
        let mut g1 = parse_graph("01,02,13");
        g1.swap_nodes(NodeIndex(1), NodeIndex(3));
        assert_eq!("02,03,13", g1.to_string());
    }

    #[test]
    fn test_connection_count() {
        let g1 = parse_graph("01,02,13");

        assert_eq!(g1.count_connections(), 6);
    }

    #[test]
    fn test_is_valid() {
        let g1 = Graph8::from_inner_unchecked(BitSet64::EMPTY.with_inserted(9));

        assert!(!g1.is_valid())
    }

    #[test]
    fn test_negate() {
        let g1 = parse_graph("01,02,13");
        let negated: Graph8 = g1.negate();

        assert!(negated.is_valid());

        assert_eq!(
            negated.to_string(),
            "03,04,05,06,07,12,14,15,16,17,23,24,25,26,27,34,35,36,37,45,46,47,56,57,67"
        );
    }

    #[test]
    fn test_fully_connected() {
        assert_eq!(Graph8::fully_connected(0), Graph8::EMPTY);
        assert_eq!(Graph8::fully_connected(1), Graph8::EMPTY);
        assert_eq!(Graph8::fully_connected(2).to_string(), "01");
        assert_eq!(Graph8::fully_connected(3).to_string(), "01,02,12");
        assert_eq!(Graph8::fully_connected(4).to_string(), "01,02,03,12,13,23");
        assert_eq!(
            Graph8::fully_connected(5).to_string(),
            "01,02,03,04,12,13,14,23,24,34"
        );
        assert_eq!(
            Graph8::fully_connected(6).to_string(),
            "01,02,03,04,05,12,13,14,15,23,24,25,34,35,45"
        );
        assert_eq!(
            Graph8::fully_connected(7).to_string(),
            "01,02,03,04,05,06,12,13,14,15,16,23,24,25,26,34,35,36,45,46,56"
        );
        assert_eq!(Graph8::fully_connected(8), Graph8::ALL);

        for x in 0..=8 {
            assert!(Graph8::fully_connected(x).is_valid());
        }
    }

    #[test]
    fn test_remove() {
        let mut set = Graph8::from_str("01,02,03,04,12,13,14,23,24,34").unwrap();
        set.remove(2);

        assert_eq!(set.to_string(), "01,03,04,13,14,34")
    }

    // #[test]
    // fn test_cs_to_layout(){
    //     let mut output = String::new();

    //     use std::fmt::Write;
    //     writeln!(output, "{}", Connections8::from_inner_unchecked(2180)).unwrap();
    //     writeln!(output, "{}", Connections8::from_inner_unchecked(21)).unwrap();
    //     writeln!(output, "{}", Connections8::from_inner_unchecked(529)).unwrap();

    //     assert_eq!(output, "abc")
    // }
}
