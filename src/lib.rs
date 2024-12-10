pub mod connections8;
pub mod graph8;
pub mod graph_path_iter;
pub mod graph_permutation8;
pub mod symmetries8;

pub (crate) const EIGHT: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeIndex(pub(crate) u8);

impl NodeIndex{
    pub const fn inner_usize(self)-> usize{
        self.0 as usize
    }

    pub const fn inner(self)-> u8{
        self.0 as u8
    }
}

impl From<u8> for NodeIndex {
    fn from(value: u8) -> Self {
        Self(value)
    }
}


impl std::fmt::Display for NodeIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}