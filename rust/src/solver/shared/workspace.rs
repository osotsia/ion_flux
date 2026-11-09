use crate::solver::_2_stepper::history::BdfHistory;
use crate::linear::solver::LinearSolver;
use crate::solver::shared::diagnostics::Diagnostics;

pub struct Workspace {
    pub t: f64,
    pub y: Vec<f64>,
    pub ydot: Vec<f64>,
    pub p: Vec<f64>,
    
    // Pre-allocated hot-loop arrays
    pub y_pred: Vec<f64>,
    pub ydot_pred: Vec<f64>,
    pub res: Vec<f64>,
    pub dy: Vec<f64>,
    pub ee: Vec<f64>,
    pub weights: Vec<f64>,

    pub history: BdfHistory,
    pub linear_solver: LinearSolver,
    pub diag: Diagnostics,
}

impl Workspace {
    pub fn new(n: usize, bw: isize, y0: Vec<f64>, ydot0: Vec<f64>, p: Vec<f64>) -> Self {
        Self {
            t: 0.0, y: y0, ydot: ydot0, p,
            y_pred: vec![0.0; n], ydot_pred: vec![0.0; n], res: vec![0.0; n],
            dy: vec![0.0; n], ee: vec![0.0; n], weights: vec![0.0; n],
            history: BdfHistory::new(n),
            linear_solver: LinearSolver::new(n, bw),
            diag: Diagnostics::new(n),
        }
    }

    pub fn clone_state(&self) -> (f64, Vec<f64>, Vec<f64>) {
        (self.t, self.y.clone(), self.ydot.clone())
    }

    pub fn restore_state(&mut self, t: f64, y: Vec<f64>, ydot: Vec<f64>) {
        self.t = t; self.y = y; self.ydot = ydot;
        self.linear_solver.mark_stale();
        self.diag.accepted_steps = 0; 
        self.history.restore_state();
    }
}