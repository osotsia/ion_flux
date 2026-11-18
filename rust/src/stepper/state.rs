use crate::api::SolverError;
use crate::stepper::history::BdfHistory;
use crate::stepper::problem::Problem;
use crate::linear::solver::LinearSolver;
use crate::diagnostics::{Diagnostics, CrashReport, SequenceTransition};
use crate::stepper::bdf;
use crate::nonlinear::algebraic_roots;

#[derive(Clone, Debug)]
pub struct SystemState {
    pub t: f64,
    pub y: Vec<f64>,
    pub ydot: Vec<f64>,
    pub p: Vec<f64>,
}

impl SystemState {
    pub fn new(y0: Vec<f64>, ydot0: Vec<f64>, p: Vec<f64>) -> Self {
        Self {
            t: 0.0,
            y: y0,
            ydot: ydot0,
            p,
        }
    }

    pub fn clone_tuple(&self) -> (f64, Vec<f64>, Vec<f64>) {
        (self.t, self.y.clone(), self.ydot.clone())
    }

    pub fn restore(&mut self, t: f64, y: Vec<f64>, ydot: Vec<f64>) {
        self.t = t;
        self.y = y;
        self.ydot = ydot;
    }
}

pub struct StepScratchpad {
    pub y_pred: Vec<f64>,
    pub ydot_pred: Vec<f64>,
    pub res: Vec<f64>,
    pub dy: Vec<f64>,
    pub ee: Vec<f64>,
    pub weights: Vec<f64>,
    pub err_delta: Vec<f64>,
}

impl StepScratchpad {
    pub fn new(n: usize) -> Self {
        Self {
            y_pred: vec![0.0; n],
            ydot_pred: vec![0.0; n],
            res: vec![0.0; n],
            dy: vec![0.0; n],
            ee: vec![0.0; n],
            weights: vec![0.0; n],
            err_delta: vec![0.0; n],
        }
    }
}

#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub t: f64,
    pub y: Vec<f64>,
    pub ydot: Vec<f64>,
}

pub trait TimeStepper {
    fn t(&self) -> f64;
    fn y(&self) -> &[f64];
    fn ydot(&self) -> &[f64];
    fn p(&self) -> &[f64];
    fn p_mut(&mut self) -> &mut [f64];
    fn calc_algebraic_roots(&mut self, prob: &Problem) -> Result<(), SolverError>;
    fn step(
        &mut self,
        prob: &Problem,
        dt: f64,
        history_cache: Option<&mut Vec<(f64, Vec<f64>, Vec<f64>)>>,
    ) -> Result<(), SolverError>;
    fn advance_to(
        &mut self,
        prob: &Problem,
        target_t: f64,
        history_cache: Option<&mut Vec<(f64, Vec<f64>, Vec<f64>)>>,
    ) -> Result<(), SolverError> {
        let dt = target_t - self.t();
        if dt <= 1e-14 {
            return Ok(());
        }
        self.step(prob, dt, history_cache)
    }
    fn record_transition(&mut self, _tr: SequenceTransition) {}
    fn checkpoint(&self) -> Checkpoint;
    fn rollback(&mut self, ckpt: &Checkpoint);
}

pub struct SolverInstance {
    pub state: SystemState,
    pub history: BdfHistory,
    pub scratch: StepScratchpad,
    pub linear_solver: LinearSolver,
    pub diag: Diagnostics,
}

impl SolverInstance {
    pub fn new(n: usize, bw: isize, y0: Vec<f64>, ydot0: Vec<f64>, p: Vec<f64>) -> Self {
        Self {
            state: SystemState::new(y0, ydot0, p),
            history: BdfHistory::new(n),
            scratch: StepScratchpad::new(n),
            linear_solver: LinearSolver::new(n, bw),
            diag: Diagnostics::new(n),
        }
    }

    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            t: self.state.t,
            y: self.state.y.clone(),
            ydot: self.state.ydot.clone(),
        }
    }

    pub fn rollback(&mut self, ckpt: &Checkpoint) {
        self.state.restore(ckpt.t, ckpt.y.clone(), ckpt.ydot.clone());
        self.linear_solver.mark_stale();
        self.diag.accepted_steps = 0;
        self.history.restore_state();
    }

    pub fn step(
        &mut self,
        prob: &Problem,
        dt: f64,
        history_cache: Option<&mut Vec<(f64, Vec<f64>, Vec<f64>)>>,
    ) -> Result<(), CrashReport> {
        bdf::step(
            prob,
            &mut self.state,
            &mut self.history,
            &mut self.scratch,
            &mut self.linear_solver,
            &mut self.diag,
            dt,
            history_cache,
        )
    }

    pub fn advance_to(
        &mut self,
        prob: &Problem,
        target_t: f64,
        history_cache: Option<&mut Vec<(f64, Vec<f64>, Vec<f64>)>>,
    ) -> Result<(), CrashReport> {
        let dt = target_t - self.state.t;
        if dt <= 1e-14 {
            return Ok(());
        }
        self.step(prob, dt, history_cache)
    }

    pub fn calc_algebraic_roots(&mut self, prob: &Problem) -> Result<(), String> {
        algebraic_roots::calc_algebraic_roots(
            prob,
            &mut self.state,
            &mut self.history,
            &mut self.scratch,
            &mut self.linear_solver,
            &mut self.diag,
        )
    }
}

impl TimeStepper for SolverInstance {
    fn t(&self) -> f64 { self.state.t }
    fn y(&self) -> &[f64] { &self.state.y }
    fn ydot(&self) -> &[f64] { &self.state.ydot }
    fn p(&self) -> &[f64] { &self.state.p }
    fn p_mut(&mut self) -> &mut [f64] { &mut self.state.p }

    fn calc_algebraic_roots(&mut self, prob: &Problem) -> Result<(), SolverError> {
        SolverInstance::calc_algebraic_roots(self, prob).map_err(SolverError::Message)
    }

    fn step(
        &mut self,
        prob: &Problem,
        dt: f64,
        history_cache: Option<&mut Vec<(f64, Vec<f64>, Vec<f64>)>>,
    ) -> Result<(), SolverError> {
        SolverInstance::step(self, prob, dt, history_cache).map_err(SolverError::Crash)
    }

    fn checkpoint(&self) -> Checkpoint {
        SolverInstance::checkpoint(self)
    }

    fn record_transition(&mut self, tr: SequenceTransition) {
        self.diag.sequence_transitions.push(tr);
    }

    fn rollback(&mut self, ckpt: &Checkpoint) {
        SolverInstance::rollback(self, ckpt)
    }
}
