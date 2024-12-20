use std::cell::RefCell;
use std::fmt::Display;
use std::iter::FusedIterator;

use std::collections::BTreeMap;
use std::num::ParseIntError;
use std::ops::Deref;
use std::str::FromStr;
use std::{u32, u8};

use const_sized_bit_set::bit_set_trait::BitSetTrait;
use const_sized_bit_set::BitSet8;

use crate::connections8::Connections8;
use crate::graph_path_iter::{GraphPath8, GraphPathIter};
use crate::graph_permutation8::{GraphPermutation8, Swap};
use crate::symmetries8::Symmetries8;
use crate::{NodeIndex, EIGHT};

/// A graph with up to 8 nodes
#[derive(Debug, Clone, PartialEq)]
pub struct Graph8 {
    pub(crate) adjacencies: [BitSet8; EIGHT], //todo refactor and just use a u64
}

impl Deref for Graph8 {
    type Target = [BitSet8; EIGHT];

    fn deref(&self) -> &Self::Target {
        &self.adjacencies
    }
}

impl Graph8 {
    pub const EMPTY: Self = Self {
        adjacencies: [BitSet8::EMPTY; EIGHT],
    };

    pub const ALL: Self = {
        let mut adjacencies = [BitSet8::ALL; EIGHT];
        let mut index = 0;
        while index < EIGHT {
            adjacencies[index].remove_const(index as u32);
            index += 1;
        }
        Self { adjacencies }
    };

    /// A graph is not valid if any of its nodes are connected to themselves
    pub const fn is_valid(&self) -> bool {
        let mut index = 0usize;
        while index < EIGHT {
            if self.adjacencies[index].contains_const(index as u32) {
                return false;
            }
            index += 1;
        }

        true
    }

    pub const fn from_adjacencies_unchecked(adjacencies: [BitSet8; EIGHT]) -> Self {
        Self { adjacencies }
    }

    pub const fn insert(&mut self, a: NodeIndex, b: NodeIndex) {
        self.adjacencies[a.0 as usize].insert_const(b.0 as u32);
        self.adjacencies[b.0 as usize].insert_const(a.0 as u32);
    }

    /// The number of active nodes.
    /// A node is active if it has connections or if a node with greater index has connections
    pub const fn active_nodes(&self) -> usize {
        let mut active_nodes = EIGHT;

        while let Some(i) = active_nodes.checked_sub(1) {
            if self.adjacencies[i].eq_const(&BitSet8::EMPTY) {
                active_nodes = i;
            } else {
                return active_nodes;
            }
        }
        return 0;
    }

    /// Return an iterator of permutations that map `sub` to a subgraph of `self`
    /// TODO cache results of this
    pub fn iter_subgraph_permutations(
        self,
        potential_subgraph: &Self,
    ) -> impl Iterator<Item = GraphPermutation8> + DoubleEndedIterator {
        let potential_subgraph = potential_subgraph.clone();
        let active_nodes = self.active_nodes();
        let iter = GraphPermutation8::all_for_n_elements(active_nodes).filter(move |permutation| {
            let mut clone = potential_subgraph.clone();
            clone.apply_permutation(*permutation);

            self.is_super_graph(&clone)
        });

        iter
    }

    ///Is this a super graph of a subgraph
    pub const fn is_super_graph(&self, subgraph: &Self) -> bool {
        let mut index = 0;
        while index < EIGHT {
            let self_adjacencies = self.adjacencies[index];
            let subgraph_adjacencies = subgraph.adjacencies[index];

            if !subgraph_adjacencies.is_subset_const(&self_adjacencies) {
                return false;
            }
            index += 1;
        }
        return true;
    }

    /// Whether this could be a super graph based numbers of connections
    pub fn could_be_super_graph(&self, subgraph: &Self) -> bool {
        struct Counts {
            min_connections: u32,
            max_connections: u32,
            total_connections: u32,
        }

        fn calculate_counts(sets: &[BitSet8]) -> Counts {
            let mut min_connections: u32 = u32::MAX;
            let mut max_connections: u32 = 0;
            let mut total_connections: u32 = 0;
            for x in sets {
                let c = x.len_const();
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

        let self_counts = calculate_counts(&self.adjacencies);
        let subgraph_counts = calculate_counts(&subgraph.adjacencies);

        match self_counts
            .total_connections
            .cmp(&subgraph_counts.total_connections)
        {
            std::cmp::Ordering::Less => return false,
            std::cmp::Ordering::Equal => {
                return self_counts.min_connections == subgraph_counts.min_connections
                    && self_counts.max_connections == subgraph_counts.max_connections;
            }
            std::cmp::Ordering::Greater => {
                return self_counts.min_connections >= subgraph_counts.min_connections
                    && self_counts.max_connections >= subgraph_counts.max_connections;
            }
        }
    }

    pub fn is_unchanged_under_permutation(&self, permutation: GraphPermutation8) -> bool {
        let mut clone = self.clone();
        for (index, swap) in permutation.swaps().enumerate() {
            let other_index = swap.index as usize;
            if self.adjacencies[index].len_const() != self.adjacencies[other_index].len_const() {
                return false;
            }
            clone.swap_nodes(NodeIndex(index as u8), NodeIndex(other_index as u8));
        }
        self == &clone
    }

    pub fn apply_permutation(&mut self, permutation: GraphPermutation8) {
        for (index, swap) in permutation.swaps().enumerate() {
            let index = index as u8;
            let other_index = swap.index;
            self.swap_nodes(NodeIndex(index), NodeIndex(other_index));
        }
    }

    pub fn get_symmetries(&self) -> Symmetries8 {
        Symmetries8::new(self)
    }

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
        let last_adj_count = other.adjacencies[last_index].len_const();

        stack[last_index] = BitSet8::from_iter(
            self.adjacencies
                .iter()
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
                index = index + 1;
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
                return GraphPermutation8::try_from_swaps(swaps.into_iter());
            };

            let other_adj = other.adjacencies[next_index];
            let other_adj_count = other_adj.len_const();
            let last_n = BitSet8::from_first_n_const(index as u32).with_negated();
            let other_intersect_last_n = other_adj.with_intersect(&last_n);

            let possible_swaps = BitSet8::from_iter(
                self.adjacencies
                    .iter()
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

    /// Finds the minimum connections set
    /// Uses a thread local cache
    pub fn find_min_connections_set(&self) -> Connections8 {
        thread_local! {
            static CACHE: RefCell<BTreeMap<(Connections8, u32), Connections8>>  = const{RefCell::new(BTreeMap::new())}  ;
        }

        CACHE.with(|c| self.find_min_connections_set_with_cache(&mut c.borrow_mut()))
    }

    fn find_min_connections_set_with_cache(
        &self,
        cache: &mut BTreeMap<(Connections8, u32), Connections8>,
    ) -> Connections8 {
        fn find_next_target_adjacencies(graph: &Graph8, unfrozen: u32) -> BitSet8 {
            let mut possibles = BitSet8::from_first_n_const(unfrozen);
            //check frozen bits to find referenced sets
            for frozen_index in (unfrozen..8).rev() {
                // The set of nodes which are not connected to this frozen node
                let ai = graph.adjacencies[frozen_index as usize].with_negated();
                let p = possibles.with_intersect(&ai);

                if !p.is_empty() {
                    possibles = p;
                }
                if possibles.len() == 1 {
                    return possibles;
                }
            }

            //find things with the fewest elements
            possibles = possibles.min_set_by_key(|x| graph.adjacencies[x as usize].len());

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
            } else {
                if let Some(cached) = cache.get(&(as_thin, unfrozen)) {
                    *cached
                } else {
                    let possibles = find_next_target_adjacencies(graph, unfrozen);

                    let r = possibles
                        .into_iter()
                        .map(|index| {
                            let mut graph = graph.clone();
                            graph.swap_nodes(
                                NodeIndex((unfrozen - 1) as u8),
                                NodeIndex(index as u8),
                            );

                            find_min_thin_graph_inner(&graph, cache, unfrozen - 1)
                        })
                        .min()
                        .unwrap_or_else(|| as_thin);

                    r
                }
            };

            cache.insert((as_thin, unfrozen), min_thin);
            min_thin

            //graph.adjacencies
        }

        let r = find_min_thin_graph_inner(&self, cache, self.active_nodes() as u32);
        r
    }

    #[inline]
    pub fn swap_nodes(&mut self, i: NodeIndex, j: NodeIndex) {
        //todo use a u64 and make more efficient
        if i.0 == j.0 {
            return;
        }

        self.adjacencies.swap(i.0 as usize, j.0 as usize);

        let mut index = 0;
        while index < EIGHT {
            self.adjacencies[index].swap_bits_const(i.0 as u32, j.0 as u32);
            index += 1;
        }
    }

    #[inline]
    #[must_use]
    pub fn to_connection_set(&self) -> Connections8 {
        Connections8::from_graph(self)
    }

    /// Counts connections between nodes.
    /// Each connection is effectively counted twice - once in each direction
    pub const fn count_connections(&self) -> u32 {
        let mut total = 0u32;
        let mut index = 0;
        while index < EIGHT {
            let adj = self.adjacencies[index];
            total += adj.len_const();
            index += 1;
        }

        total
    }

    #[must_use]
    pub const fn negate(&self) -> Self {
        let mut adjacencies: [BitSet8; EIGHT] = self.adjacencies;

        let mut index = 0;
        while index < EIGHT {
            adjacencies[index].negate_const();
            adjacencies[index].remove_const(index as u32);
            index += 1;
        }

        Self { adjacencies }
    }

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
    pub fn iter_paths(self) -> impl Iterator<Item = GraphPath8> + FusedIterator + Clone {
        GraphPathIter::new(self)
    }
}

impl Display for Graph8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for (index, adjacencies) in self.adjacencies.iter().enumerate() {
            for other in adjacencies.into_iter().skip_while(|x| *x <= index as u32) {
                use std::fmt::Write;
                if first {
                    first = false
                } else {
                    f.write_char(',')?;
                }
                write!(f, "{index}{other}")?;
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
            if x.len() > 0 {
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
    use itertools::Itertools;

    use super::{Graph8, EIGHT};
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
            writeln!(data, "{}", subgraph.to_string()).unwrap();
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

        assert_eq!(mg1, mg2, "{} {}", mg1.to_string(), mg2.to_string());
        assert_eq!(mg1.inner(), 7);

        assert_eq!(mg2.to_graph(), g1);
    }

    #[test]
    fn test_min_thin_connections_set2() {
        let g1 = parse_graph("12,14,15");
        let g2 = parse_graph("01,02,03");

        let mg1 = g1.find_min_connections_set();
        let mg2 = g2.find_min_connections_set();

        assert_eq!(mg1, mg2, "{} {}", mg1.to_string(), mg2.to_string());
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
        let mut adj = [const_sized_bit_set::BitSet8::EMPTY; EIGHT];
        adj[1].insert_const(1);
        let g1 = Graph8::from_adjacencies_unchecked(adj);

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
