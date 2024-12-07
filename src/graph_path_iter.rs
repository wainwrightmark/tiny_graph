use arrayvec::ArrayVec;
use const_sized_bit_set::{bit_set_trait::BitSetTrait, BitSet8};

use crate::{graph8::Graph8, NodeIndex, EIGHT};

#[derive(Debug, Clone)]
pub struct GraphPathIter {
    graph: Graph8,
    next_steps: ArrayVec<BitSet8, EIGHT>,
    current: ArrayVec<NodeIndex, EIGHT>,
    used: BitSet8,
}

impl GraphPathIter {
    pub fn new(graph: Graph8) -> Self {
        let mut next_steps = ArrayVec::new();
        next_steps.push(BitSet8::ALL);
        Self {
            graph,
            next_steps,
            current: ArrayVec::new(),
            used: BitSet8::EMPTY,
        }
    }
}

impl std::iter::FusedIterator for GraphPathIter {}
impl Iterator for GraphPathIter {
    type Item = GraphPath8;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let top = self.next_steps.last_mut()?;

            let Some(next_index) = top.pop_const() else {
                let _ = self.next_steps.pop();

                if let Some(top_index) = self.current.pop() {
                    self.used.remove_const(top_index.0 as u32);
                }

                continue;
            };

            let next_options = self.graph.adjacencies[next_index as usize].with_except(&self.used);

            let next_tile_index = NodeIndex(next_index as u8);

            if next_options.is_empty() {
                let mut tiles = self.current.clone();
                tiles.push(next_tile_index);

                return Some(GraphPath8 { tiles });
            } else {
                self.current.push(next_tile_index);
                self.next_steps.push(next_options);
                self.used.insert_const(next_index);

                return Some(GraphPath8 {
                    tiles: self.current.clone(),
                });
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphPath8 {
    pub tiles: ArrayVec<NodeIndex, EIGHT>,
}

impl std::fmt::Display for GraphPath8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (position, node) in self.tiles.iter().enumerate() {
            if position != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", node.0)?;
        }

        Ok(())
    }
}