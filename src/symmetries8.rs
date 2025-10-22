use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    sync::LazyLock,
};

use crate::{connections8::Connections8, graph8::Graph8, graph_permutation8::GraphPermutation8};

#[cfg(any(test, feature = "serde"))]
use serde::{Deserialize, Serialize};

/// Set of disjoint symmetries for a graph.
/// Does not include the identity symmetry
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(any(test, feature = "serde"), derive(Serialize, Deserialize), serde(transparent))]
pub struct Symmetries8 {
    symmetries: Vec<GraphPermutation8>,
}

static EMPTY: LazyLock<Symmetries8> = LazyLock::new(|| Symmetries8 {
    symmetries: Vec::new(),
});

impl Symmetries8 {
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.symmetries.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.symmetries.len()
    }

    #[inline]
    pub fn as_slice(&self) -> &[GraphPermutation8] {
        self.symmetries.as_slice()
    }

    pub fn new_cached(graph: &Graph8) -> Self {
        thread_local! {
            static CACHE: RefCell<BTreeMap<Connections8, Symmetries8>>  = const{RefCell::new(BTreeMap::new())}  ;
        }

        CACHE.with(|cache| {
            let cs = graph.to_connection_set();

            match cache.borrow_mut().entry(cs) {
                std::collections::btree_map::Entry::Vacant(vacant_entry) => {
                    let symmetries = Self::new_uncached(graph);
                    vacant_entry.insert(symmetries).clone()
                }
                std::collections::btree_map::Entry::Occupied(occupied_entry) => {
                    occupied_entry.get().clone()
                }
            }
        })
    }

    fn new_uncached(graph: &Graph8) -> Self {
        let mut disjoint_symmetries: Vec<GraphPermutation8> = vec![];
        let mut all_symmetries =
            BTreeSet::<GraphPermutation8>::from_iter([GraphPermutation8::IDENTITY]);

        for permutation in GraphPermutation8::all_for_n_elements(graph.active_nodes()) {
            if graph.is_unchanged_under_permutation(permutation)
                && all_symmetries.insert(permutation)
            {
                for c in permutation
                    .generate_cycle()
                    .take_while(|x| !x.is_identity())
                {
                    all_symmetries.insert(c);
                    for d in disjoint_symmetries.iter() {
                        let combined = c.combine(d);
                        all_symmetries.insert(combined);
                    }
                }

                disjoint_symmetries.push(permutation);
            }
        }

        if disjoint_symmetries.is_empty() {
            return EMPTY.clone();
        }

        Self {
            symmetries: disjoint_symmetries,
        }
    }
}

#[cfg(test)]

mod tests {
    use std::str::FromStr;

    use itertools::Itertools;

    use crate::graph8::Graph8;

    #[test]
    pub fn test_graph_symmetries() {
        let graph = Graph8::from_str("01,02,12,13,23").unwrap();

        let symmetries = graph.get_symmetries();

        let symmetries_string = symmetries
            .as_slice()
            .iter()
            .map(|x| x.get_array().into_iter().join(","))
            .join("\n");

        //Swap elements 1 and 2 or Swap elements 0 and 3
        assert_eq!(symmetries_string, "0,2,1,3,4,5,6,7\n3,1,2,0,4,5,6,7")
    }
}
