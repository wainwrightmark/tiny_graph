pub mod connections8;
pub mod graph8;
pub mod graph_path_iter;
pub mod graph_permutation8;
pub mod symmetries8;
pub mod graph_permutation14;

#[cfg(any(test, feature = "serde"))]
use serde::{Deserialize, Serialize};

pub(crate) const EIGHT: usize = 8;
pub(crate) const FOURTEEN: usize = 14;

#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(any(test, feature = "serde"), derive(Serialize, Deserialize), serde(transparent))]
pub struct NodeIndex(pub(crate) u8);

impl NodeIndex {
    #[inline]
    pub const fn inner_usize(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub const fn inner(self) -> u8 {
        self.0
    }

    #[inline]
    pub const fn from_inner(inner: u8) -> Self {
        Self(inner)
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
