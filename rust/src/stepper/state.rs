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
