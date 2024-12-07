pub mod connections8;
pub mod graph8;
pub mod graph_path_iter;

pub (crate) const EIGHT: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeIndex(pub(crate) u8);

impl From<u8> for NodeIndex {
    fn from(value: u8) -> Self {
        Self(value)
    }
}
