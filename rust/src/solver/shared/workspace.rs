// Workspace struct dismantled in Phase 3.
// Replaced by segregated SystemState, BdfHistory, and StepScratchpad in crate::stepper::state.
#[allow(unused_imports)]
pub use crate::stepper::state::{SystemState, StepScratchpad};
#[allow(unused_imports)]
pub use crate::stepper::history::BdfHistory;