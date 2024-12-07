use std::collections::BTreeSet;
use std::fmt::Display;
use std::iter::FusedIterator;

use std::collections::BTreeMap;
use std::num::ParseIntError;
use std::str::FromStr;
use std::u32;

use const_sized_bit_set::bit_set_trait::BitSetTrait;
use const_sized_bit_set::BitSet8;
use importunate::Permutation;

use crate::connections8::Connections8;
use crate::graph_path_iter::{GraphPath8, GraphPathIter};
use crate::{NodeIndex, EIGHT};

/// A graph with up to 8 nodes
#[derive(Debug, Clone, PartialEq)]
pub struct Graph8 {
    pub(crate) adjacencies: [BitSet8; 8],
}

impl Graph8 {
    pub const EMPTY: Self = Self {
        adjacencies: [BitSet8::EMPTY; EIGHT],
    };

    pub const fn insert(&mut self, a: NodeIndex, b: NodeIndex) {
        self.adjacencies[a.0 as usize].insert_const(b.0 as u32);
        self.adjacencies[b.0 as usize].insert_const(a.0 as u32);
    }

    /// The number of active nodes.
    /// A noe is active if it has connections or if a node with greater index has connections
    pub(crate) const fn active_nodes(&self) -> usize {
        let mut used_nodes = BitSet8::EMPTY;
        let mut index = 0;
        while index < EIGHT {
            used_nodes.union_with_const(&self.adjacencies[index]);
            index += 1;
        }

        EIGHT - used_nodes.inner_const().leading_zeros() as usize
    }

    /// Return an iterator of permutations that map `sub` to a subgraph of `self`
    /// TODO cache results of this
    pub fn iter_subgraph_permutations(
        self,
        potential_subgraph: &Self,
    ) -> impl Iterator<Item = Permutation<u16, EIGHT>> + DoubleEndedIterator {
        let potential_subgraph = potential_subgraph.clone();
        let iter = Permutation::all().filter(move |permutation| {
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

    pub fn is_unchanged_under_permutation<const N: usize>(
        &self,
        permutation: Permutation<u16, N>,
    ) -> bool {
        let mut clone = self.clone();
        for (index, swap) in permutation.swaps().enumerate() {
            if self.adjacencies[index].len_const()
                != self.adjacencies[index + swap as usize].len_const()
            {
                return false;
            }
            clone.swap_nodes(NodeIndex(index as u8), NodeIndex(index as u8 + swap as u8));
        }
        self == &clone
    }

    pub fn apply_permutation<const N: usize>(&mut self, permutation: Permutation<u16, N>) {
        for (index, swap) in permutation.swaps().enumerate() {
            self.swap_nodes(NodeIndex(index as u8), NodeIndex(index as u8 + swap as u8));
        }
    }

    // pub(crate) fn find_class_indices(&self) -> BitSet8 {
    //     let mut seen: BTreeSet<Connections8> = Default::default();
    //     let mut set: BitSet8 = BitSet8::EMPTY;
    //     for index in 0..EIGHT {
    //         let mut clone = self.clone();
    //         clone.swap_nodes(NodeIndex(0), NodeIndex(index as u8));
    //         if seen.insert(clone.to_connection_set()) {
    //             set.insert(index as u32);
    //         }
    //     }

    //     set
    // }

    /// Find swaps to map this graph to `other`
    ///
    /// for (i, j) in swaps.into_iter().enumerate() {
    ///  board.runes.swap(i, j as usize);
    /// }
    pub fn find_mapping_swaps(mut self, other: &Self) -> Option<[u8; EIGHT]> {
        let mut swaps = [0; EIGHT];

        let mut stack = [BitSet8::EMPTY; EIGHT];
        let zero_adj_count = other.adjacencies[0].len_const();
        stack[0] = BitSet8::from_iter(
            self.adjacencies
                .iter()
                .enumerate()
                .filter(|x| x.1.len_const() == zero_adj_count)
                .map(|x| x.0 as u32),
        );

        let mut index: usize = 0;

        loop {
            let Some(top) = stack.get_mut(index) else {
                return Some(swaps);
            };

            let Some(next_adj) = top.pop_const() else {
                index = index.checked_sub(1)?;

                self.swap_nodes(NodeIndex(index as u8), NodeIndex(swaps[index]));

                continue;
            };
            swaps[index] = next_adj as u8;
            self.swap_nodes(NodeIndex(index as u8), NodeIndex(next_adj as u8));

            //println!("{c}: Top {:016b} Swaps {:?}",top.inner(),  swaps);
            let next_index = index + 1;
            if next_index >= EIGHT {
                //todo actually just go up to the max of the active node counts
                return Some(swaps);
            };

            let other_adj = other.adjacencies[next_index];
            let other_adj_count = other_adj.len_const();
            let first_n = BitSet8::from_first_n_const(next_index as u32);
            let other_first_n = other_adj.with_intersect(&first_n);

            let possible_swaps = BitSet8::from_iter(
                self.adjacencies
                    .iter()
                    .enumerate()
                    .skip(next_index)
                    .filter(|x| {
                        x.1.len_const() == other_adj_count
                            && x.1.with_intersect(&first_n) == other_first_n
                    })
                    .map(|x| x.0 as u32),
            );
            stack[next_index] = possible_swaps;

            index = next_index;
        }
    }

    pub fn find_min_thin_graph(
        &self,
        cache: &mut BTreeMap<(Connections8, u32), Connections8>,
    ) -> Connections8 {
        fn find_next_target_adjacencies(graph: &Graph8, frozen: u32) -> BitSet8 {
            let mut possibles = BitSet8::ALL.with_except(&BitSet8::from_first_n_const(frozen));
            //check frozen bits to find referenced sets
            for i in 0..frozen {
                let ai = graph.adjacencies[i as usize];
                let p = ai.with_intersect(&possibles);

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
            frozen: u32,
        ) -> Connections8 {
            let as_thin = graph.to_connection_set();

            let min_thin = if frozen as usize >= EIGHT {
                as_thin
            } else {
                if let Some(cached) = cache.get(&(as_thin, frozen)) {
                    *cached
                } else {
                    let possibles = find_next_target_adjacencies(graph, frozen);

                    let r = possibles
                        .into_iter()
                        .map(|index| {
                            let mut graph = graph.clone();
                            graph.swap_nodes(NodeIndex(frozen as u8), NodeIndex(index as u8));

                            find_min_thin_graph_inner(&graph, cache, frozen + 1)
                        })
                        .min()
                        .unwrap_or_else(|| as_thin);

                    r
                }
            };

            cache.insert((as_thin, frozen), min_thin);
            min_thin

            //graph.adjacencies
        }

        let r = find_min_thin_graph_inner(&self, cache, 0);
        r
    }

    #[inline]
    pub fn swap_nodes(&mut self, i: NodeIndex, j: NodeIndex) {
        if i == j {
            return;
        }

        self.adjacencies.swap(i.0 as usize, j.0 as usize);

        for adj in self.adjacencies.iter_mut() {
            *adj = adj.with_bits_swapped(i.0 as u32, j.0 as u32);
        }
    }

    #[inline]
    #[must_use]
    pub fn to_connection_set(&self) -> Connections8 {
        //todo impl into
        Connections8::from_graph(self)
    }

    /// Each connection is only counted once e.g. 0-1 is the same as 1-0
    pub fn count_connections(&self) -> u32 {
        self.adjacencies.iter().map(|x| x.len()).sum::<u32>() / 2
    }

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
    use std::{collections::BTreeMap, str::FromStr};

    use importunate::Permutation;

    use crate::NodeIndex;

    use super::Graph8;

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
        let supergraph = parse_graph("01,02,23");
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

        let swap_01 = Permutation::<u16, 2>::rotate_left();

        assert!(g1.is_unchanged_under_permutation(swap_01));
        assert!(!g2.is_unchanged_under_permutation(swap_01));
    }

    #[test]
    fn test_apply_permutation() {
        let mut graph = parse_graph("01,02,23");

        let swap_01 = Permutation::<u16, 2>::rotate_left();
        graph.apply_permutation(swap_01);
        assert_eq!(graph.to_string(), "01,12,23");
    }

    #[test]
    fn test_find_mapping_swaps_success() {
        let g1 = parse_graph("01,02,12");
        let g2 = parse_graph("01,03,13");

        let swaps = g1
            .find_mapping_swaps(&g2)
            .expect("Should be able to find swaps");

        // Swap nodes 2 and 3
        assert_eq!(swaps, [0, 1, 3, 3, 4, 5, 6, 7])
    }

    #[test]
    fn test_find_mapping_swaps_fail() {
        let g1 = parse_graph("01,02,13");
        let g2 = parse_graph("01,03,13");

        let swaps = g1.find_mapping_swaps(&g2);

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
    fn test_min_thin_graph() {
        let g1 = parse_graph("01,02,12");
        let g2 = parse_graph("01,03,13");
        let mut cache: BTreeMap<_, _> = BTreeMap::default();

        let mg1 = g1.find_min_thin_graph(&mut cache);
        let mg2 = g2.find_min_thin_graph(&mut cache);

        //assert_eq!(g1.to_connection_set().inner(), 7);
        //assert_eq!(g2.to_connection_set().inner(), 25);

        assert_eq!(mg1, mg2);
        assert_eq!(mg1.inner(), 202375168);

        assert_eq!(mg2.to_graph(), g1);
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

        assert_eq!(g1.count_connections(), 3);
    }

    // #[test]
    // fn test_swap_board_to_new_layout() {
    //     let old_board = DynamicBoard::PLUS1THROUGH10;

    //     let old_layout = TileLayout::from_inner(536903685);
    //     let new_layout = TileLayout::from_inner(16405);

    //     let new_board = swap_board_to_new_layout(old_layout, new_layout, &old_board);

    //     let expected_new_board = DynamicBoard {
    //         runes: [
    //             Rune::Plus1,
    //             Rune::Plus3,
    //             Rune::Plus4,
    //             Rune::Plus2,
    //             Rune::Plus5,
    //             Rune::Plus6,
    //             Rune::Plus7,
    //             Rune::Plus8,
    //             Rune::Plus9,
    //             Rune::Plus10,
    //         ],
    //     };

    //     assert_eq!(new_board, Some(expected_new_board));
    // }

    // #[test]
    // fn test_graph_min_thins_4() {
    //     let pairs = vec![
    //         // T shape
    //         (65557, 527366),
    //         (1074069508, 527366),
    //         //4 in a row
    //         (85, 525314),
    //         (8213, 525314),
    //         (4398314962945, 525314),
    //         // Square
    //         (81925, 527374),
    //         (327700, 527374),
    //     ];

    //     for (tl, expected_thin) in pairs {
    //         let tile_layout = TileLayout::from_inner(tl);
    //         //let expanded = tile_layout.expand_uncached();
    //         let fat_graph: Graph8<4> = Graph8::from_tile_layout(tile_layout);

    //         let mut cache = BTreeMap::new();

    //         let min_thin: number_search::ConnectionsSet = fat_graph.find_min_thin_graph(&mut cache);

    //         // for (graph, min_graph) in cache.iter(){
    //         //     println!("From: {}\nTo  : {}\n", graph.list_connections::<4>(), min_graph.list_connections::<4>());
    //         // }

    //         assert_eq!(
    //             min_thin.inner(),
    //             expected_thin,
    //             "Layout {tl} {}",
    //             min_thin.list_connections()
    //         );
    //     }
    // }

    // #[test]
    // fn test_graph_min_thins_5() {
    //     let pairs = vec![(16810025, 1573890), (4325413, 1573890)];

    //     for (tl, expected_thin) in pairs {
    //         let tile_layout = TileLayout::from_inner(tl);
    //         //let expanded = tile_layout.expand_uncached();
    //         let fat_graph: Graph8<5> = Graph8::from_tile_layout(tile_layout);

    //         let mut cache = BTreeMap::new();

    //         let min_thin = fat_graph.find_min_thin_graph(&mut cache);

    //         // for (graph, min_graph) in cache.iter(){
    //         //     println!("From: {}\nTo  : {}\n", graph.list_connections::<4>(), min_graph.list_connections::<4>());
    //         // }

    //         assert_eq!(
    //             min_thin.inner(),
    //             expected_thin,
    //             "Layout {tl} {}",
    //             min_thin.list_connections()
    //         );
    //     }
    // }

    // #[test]
    // fn test_subgraph_permutations() {
    //     let pyramid: Graph8<6> =
    //         Graph8::from_tile_layout(TileLayout::from_inner(TileLayout::PYRAMID));

    //     let fish: Graph8<6> = Graph8::from_tile_layout(TileLayout::from_inner(1170378915848));

    //     let permutations = pyramid.iter_subgraph_permutations(fish).collect_vec();

    //     let board = DynamicBoard::PLUS1THROUGH10;

    //     let pyramid_layout = TileLayout::from_inner(TileLayout::PYRAMID).get_expanded();

    //     let permuted_boards = permutations
    //         .into_iter()
    //         .map(|p| {
    //             let mut b2 = board.clone();
    //             p.apply(&mut b2.runes);

    //             pyramid_layout.format_board_multiline(&b2)
    //         })
    //         .join("\n\n");

    //     insta::assert_snapshot!(permuted_boards)
    // }

    // #[test]
    // fn test_graphs() {
    //     for (name, tile_layout) in TileLayout::NAMED_VARIANTS {
    //         let tile_layout = TileLayout::from_inner(*tile_layout);

    //         let fat_graph: Graph8<9> = Graph8::from_tile_layout(tile_layout);

    //         assert_eq!(
    //             tile_layout.count_connections(),
    //             fat_graph.connection_count(),
    //             "{name} connection count"
    //         );

    //         let thin_graph = fat_graph.to_thin();
    //         let fat2 = thin_graph.to_fat_graph();

    //         assert_eq!(
    //             fat_graph,
    //             fat2,
    //             "{name} round trip\n\n{}\n\n{}",
    //             fat_graph.list_connections(),
    //             fat2.list_connections()
    //         );

    //         let mut path_count_4 = 1;
    //         let mut path_count_6 = 1;
    //         let mut path_count_9 = 1;

    //         for path in fat_graph
    //             .iter_paths()
    //             .filter(|x| (x.tiles[0].0 as usize) < tile_layout.count())
    //         {
    //             //println!("{path}");

    //             // if! found_paths.insert(path.clone()){
    //             //     println!("Duplicate Path {path}")
    //             // }
    //             if path.tiles.len() <= 4 {
    //                 path_count_4 += 1;
    //             }
    //             if path.tiles.len() <= 6 {
    //                 path_count_6 += 1;
    //             }
    //             if path.tiles.len() <= 9 {
    //                 path_count_9 += 1;
    //             }
    //         }

    //         let info = TileLayoutInfo::calculate_uncached(tile_layout);

    //         // println!(
    //         //     r#"{name}
    //         // Path Count 4: Actual {path_count_4} Expected {}
    //         // Path Count 6: Actual {path_count_6} Expected {}
    //         // Path Count 9: Actual {path_count_9} Expected {}
    //         // "#,
    //         //     info.path_count_4, info.path_count_6, info.path_count_9
    //         // );

    //         assert_eq!(path_count_4, info.path_count_4, "{name} path count 4",);
    //         assert_eq!(path_count_6, info.path_count_6, "{name} path count 6",);
    //         assert_eq!(path_count_9, info.path_count_9, "{name} path count 9",);
    //     }
    // }
}
