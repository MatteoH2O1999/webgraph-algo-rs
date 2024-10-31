pub mod algo;
pub mod utils;

/// Module exposing all traits in a single level.
pub mod traits {
    use super::*;
    pub use algo::traits::*;
    pub use utils::traits::*;
}

/// Use `use webgraph_algo::prelude::*;` to import common utilities, modules and
/// all traits.
pub mod prelude {
    use super::*;
    pub use algo::visits::bfv;
    pub use algo::visits::dfv;
    pub use traits::*;
    #[doc(hidden)]
    pub use utils::MmapFlags;
    pub use utils::TempMmapOptions;
}
