pub use crate::api;
pub use crate::ffi;
pub use crate::stepper;
pub use crate::nonlinear;
pub use crate::linear;
pub use crate::control;
pub use crate::diagnostics;
pub use crate::vendor;

pub mod shared {
    pub mod problem {
        pub use crate::stepper::problem::*;
    }
    pub mod diagnostics {
        pub use crate::diagnostics::*;
    }
    pub mod callbacks {
        pub use crate::ffi::callbacks::*;
    }
}

pub mod _0_ffi {
    pub use crate::api::*;
}

pub mod _1_orchestrator {
    pub mod bisection {
        pub use crate::control::bisection::*;
    }
    pub mod protocol {
        pub use crate::control::sequence::*;
    }
}

pub mod _2_stepper {
    pub mod bdf {
        pub use crate::stepper::bdf::*;
    }
    pub mod history {
        pub use crate::stepper::history::*;
    }
}

pub mod _3_nonlinear {
    pub mod newton {
        pub use crate::nonlinear::newton::*;
    }
    pub mod constraints {
        pub use crate::nonlinear::constraints::*;
    }
    pub mod algebraic_roots {
        pub use crate::nonlinear::algebraic_roots::*;
    }
}

pub mod _4_linear {
    pub mod jacobian {
        pub use crate::linear::jacobian::*;
    }
    pub mod sparse_lu {
        pub use crate::linear::sparse_lu::*;
    }
    pub mod gmres {
        pub use crate::linear::gmres::*;
    }
}

pub mod sundials {
    pub use crate::vendor::sundials::*;
}
