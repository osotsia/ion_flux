use crate::solver::shared::problem::{Problem, Callbacks, CprData, SolverConfig};
use crate::stepper::state::{SystemState, StepScratchpad};
use crate::stepper::history::BdfHistory;
use crate::linear::solver::LinearSolver;
use crate::solver::shared::diagnostics::Diagnostics;

pub const MOCK_N: usize = 6;
pub const MOCK_N_PARAMS: usize = 2;
pub const MOCK_N_OBS: usize = 2;

/// Deterministic 6x6 residual function:
/// Rows 0..4: tridiagonal diffusion-reaction ODEs
/// Row 5: dense arrowhead algebraic constraint: y_5 - \sum_{j=0}^4 y_j - p_1 = 0
pub unsafe extern "C" fn mock_evaluate_residual(
    y: *const f64,
    ydot: *const f64,
    p: *const f64,
    _m: *const f64,
    res: *mut f64,
) {
    let y = std::slice::from_raw_parts(y, MOCK_N);
    let ydot = std::slice::from_raw_parts(ydot, MOCK_N);
    let p = std::slice::from_raw_parts(p, MOCK_N_PARAMS);
    let res = std::slice::from_raw_parts_mut(res, MOCK_N);

    res[0] = ydot[0] + (2.0 + p[0]) * y[0] - y[1];
    res[1] = ydot[1] - y[0] + 2.0 * y[1] - y[2];
    res[2] = ydot[2] - y[1] + 2.0 * y[2] - y[3];
    res[3] = ydot[3] - y[2] + 2.0 * y[3] - y[4];
    res[4] = ydot[4] - y[3] + 2.0 * y[4] - y[5];
    res[5] = y[5] - (y[0] + y[1] + y[2] + y[3] + y[4]) - p[1];
}

/// JVP: out = J(c_j) * v = (dF/dy + c_j * dF/dydot) * v
pub unsafe extern "C" fn mock_evaluate_jvp(
    _y: *const f64,
    _ydot: *const f64,
    p: *const f64,
    _m: *const f64,
    c_j: f64,
    v: *const f64,
    jvp_out: *mut f64,
) {
    let p = std::slice::from_raw_parts(p, MOCK_N_PARAMS);
    let v = std::slice::from_raw_parts(v, MOCK_N);
    let out = std::slice::from_raw_parts_mut(jvp_out, MOCK_N);

    out[0] = (c_j + 2.0 + p[0]) * v[0] - v[1];
    out[1] = -v[0] + (c_j + 2.0) * v[1] - v[2];
    out[2] = -v[1] + (c_j + 2.0) * v[2] - v[3];
    out[3] = -v[2] + (c_j + 2.0) * v[3] - v[4];
    out[4] = -v[3] + (c_j + 2.0) * v[4] - v[5];
    out[5] = -v[0] - v[1] - v[2] - v[3] - v[4] + v[5];
}

/// VJP:
/// dy_out = (dF/dy)^T * lambda
/// dydot_out = (dF/dydot)^T * lambda
/// dp_out = (dF/dp)^T * lambda
pub unsafe extern "C" fn mock_evaluate_vjp(
    y: *const f64,
    _ydot: *const f64,
    p: *const f64,
    _m: *const f64,
    lambda_vec: *const f64,
    dp_out: *mut f64,
    dy_out: *mut f64,
    dydot_out: *mut f64,
) {
    let y = std::slice::from_raw_parts(y, MOCK_N);
    let p = std::slice::from_raw_parts(p, MOCK_N_PARAMS);
    let lam = std::slice::from_raw_parts(lambda_vec, MOCK_N);
    let dp = std::slice::from_raw_parts_mut(dp_out, MOCK_N_PARAMS);
    let dy = std::slice::from_raw_parts_mut(dy_out, MOCK_N);
    let dydot = std::slice::from_raw_parts_mut(dydot_out, MOCK_N);

    // dp_out = (dF/dp)^T * lambda
    dp[0] = lam[0] * y[0];
    dp[1] = -lam[5];

    // dy_out = (dF/dy)^T * lambda
    dy[0] = (2.0 + p[0]) * lam[0] - lam[1] - lam[5];
    dy[1] = -lam[0] + 2.0 * lam[1] - lam[2] - lam[5];
    dy[2] = -lam[1] + 2.0 * lam[2] - lam[3] - lam[5];
    dy[3] = -lam[2] + 2.0 * lam[3] - lam[4] - lam[5];
    dy[4] = -lam[3] + 2.0 * lam[4] - lam[5];
    dy[5] = -lam[4] + lam[5];

    // dydot_out = (dF/dydot)^T * lambda
    dydot[0] = lam[0];
    dydot[1] = lam[1];
    dydot[2] = lam[2];
    dydot[3] = lam[3];
    dydot[4] = lam[4];
    dydot[5] = 0.0;
}

pub unsafe extern "C" fn mock_evaluate_obs(
    y: *const f64,
    _ydot: *const f64,
    _p: *const f64,
    _m: *const f64,
    obs: *mut f64,
) {
    let y = std::slice::from_raw_parts(y, MOCK_N);
    let obs = std::slice::from_raw_parts_mut(obs, MOCK_N_OBS);
    obs[0] = y[0];
    obs[1] = y[5];
}

/// Computes the exact analytical 6x6 Jacobian J(c_j) = dF/dy + c_j * dF/dydot
pub fn mock_analytical_jacobian(p0: f64, c_j: f64) -> [[f64; 6]; 6] {
    [
        [c_j + 2.0 + p0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0, c_j + 2.0, -1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, c_j + 2.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, c_j + 2.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0, c_j + 2.0, -1.0],
        [-1.0, -1.0, -1.0, -1.0, -1.0, 1.0],
    ]
}

/// Builds CPR metadata for the 6x6 mock system:
/// 3 colors for sparse tridiagonal rows 0..4 + 1 dense row (index 5)
pub fn build_mock_cpr() -> CprData {
    CprData {
        color_seeds: vec![
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], // Color 0: cols 0, 3
            vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0], // Color 1: cols 1, 4
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0], // Color 2: cols 2, 5
        ],
        color_ptrs: vec![0, 5, 10, 14],
        color_rows: vec![
            0, 1, 2, 3, 4, // for Color 0
            0, 1, 2, 3, 4, // for Color 1
            1, 2, 3, 4,    // for Color 2
        ],
        color_cols: vec![
            0, 0, 3, 3, 3, // for Color 0
            1, 1, 1, 4, 4, // for Color 1
            2, 2, 2, 5,    // for Color 2
        ],
        dense_rows: vec![5],
    }
}

/// Constructs an in-memory Problem and segregated execution components for deterministic testing
pub fn create_mock_fixture() -> (Problem, SystemState, BdfHistory, StepScratchpad, LinearSolver, Diagnostics) {
    let fns = Callbacks {
        res_fn: mock_evaluate_residual,
        obs_fn: Some(mock_evaluate_obs),
        jvp_fn: Some(mock_evaluate_jvp),
        vjp_fn: Some(mock_evaluate_vjp),
        set_threads_fn: None,
    };

    let cpr = build_mock_cpr();
    let id = vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
    let constraints = vec![0.0; MOCK_N];
    let m = vec![1.0; 10];
    let spatial_diag = vec![0.0; MOCK_N];
    let max_steps = vec![0.0; MOCK_N];
    let config = SolverConfig::default();

    let prob = Problem {
        n: MOCK_N,
        bw: 0,
        n_obs: MOCK_N_OBS,
        id,
        constraints,
        m,
        spatial_diag,
        max_steps,
        cpr,
        config,
        fns,
    };

    let p = vec![0.5, 1.0];
    let y0 = vec![1.0, 1.0, 1.0, 1.0, 1.0, 6.0];
    let ydot0 = vec![0.0; MOCK_N];

    let state = SystemState::new(y0, ydot0, p);
    let history = BdfHistory::new(MOCK_N);
    let scratch = StepScratchpad::new(MOCK_N);
    let linear_solver = LinearSolver::new(MOCK_N, 0);
    let diag = Diagnostics::new(MOCK_N);

    (prob, state, history, scratch, linear_solver, diag)
}
