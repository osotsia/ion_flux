#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, ToPyArray};
use std::os::raw::{c_double, c_int, c_void};
use std::io::Write;
use crate::ffi::callbacks::{Callbacks, NativeIcFn, NativeResFn, NativeJvpFn, NativeVjpFn, NativeObsFn};
use crate::stepper::problem::{CprData, Problem, SolverConfig};
use crate::stepper::state::{Checkpoint, TimeStepper};
use crate::control::sequence::{run_sequence, SolverError};
use crate::api::batch_dispatcher::{RawProtocolStep, parse_protocol_steps};
use crate::linear::jacobian::assemble_triplets;
use crate::vendor::sundials::sys::*;

pub struct SundialsUserData {
    pub res_fn: NativeResFn,
    pub jvp_fn: Option<NativeJvpFn>,
    pub vjp_fn: Option<NativeVjpFn>,
    pub obs_fn: Option<NativeObsFn>,
    pub p: Vec<f64>,
    pub m: Vec<f64>,
    pub cpr: CprData,
    pub triplets: Vec<(usize, usize, f64)>,
    pub is_banded: bool,
    pub sun_bw: isize,
    pub n: usize,
    pub id: Vec<f64>,
}

unsafe extern "C" fn ida_res_callback(_t: c_double, y: N_Vector, yp: N_Vector, res: N_Vector, user_data: *mut c_void) -> c_int {
    let ud = &*(user_data as *const SundialsUserData);
    (ud.res_fn)(N_VGetArrayPointer(y), N_VGetArrayPointer(yp), ud.p.as_ptr(), ud.m.as_ptr(), N_VGetArrayPointer(res));
    0
}

unsafe extern "C" fn ida_jac_callback(_t: c_double, c_j: c_double, y: N_Vector, yp: N_Vector, _r: N_Vector, jac: SUNMatrix, user_data: *mut c_void, _t1: N_Vector, _t2: N_Vector, _t3: N_Vector) -> c_int {
    let ud = &mut *(user_data as *mut SundialsUserData);
    let n = ud.n;

    assemble_triplets(
        n, std::slice::from_raw_parts(N_VGetArrayPointer(y), n),
        std::slice::from_raw_parts(N_VGetArrayPointer(yp), n),
        &ud.p, &ud.m, c_j,
        ud.jvp_fn, ud.vjp_fn,
        &mut ud.triplets,
        &ud.cpr,
        &ud.id
    );

    if ud.is_banded {
        let a_data = SUNBandMatrix_Data(jac);
        let ldim = SUNBandMatrix_LDim(jac) as usize;
        let mu = SUNBandMatrix_UpperBandwidth(jac) as isize;

        for i in 0..(ldim * n) { *a_data.add(i) = 0.0; }

        for &(r, c, val) in &ud.triplets {
            let band_idx = (r as isize - c as isize + mu) as usize + c * ldim;
            *a_data.add(band_idx) = val;
        }
    } else {
        let a_data = SUNDenseMatrix_Data(jac);
        for i in 0..(n*n) { *a_data.add(i) = 0.0; }
        for &(r, c, val) in &ud.triplets { *a_data.add(c * n + r) = val; }
    }
    0
}

#[pyclass(unsendable)]
pub struct SundialsHandle {
    _lib: libloading::Library,
    ida_mem: *mut c_void,
    y_vec: N_Vector,
    yp_vec: N_Vector,
    id_vec: N_Vector,
    sunctx: SUNContext,
    a_mat: SUNMatrix,
    ls: SUNLinearSolver,
    user_data: Box<SundialsUserData>,
    pub n: usize,
    pub t: f64,
    pub n_obs: usize,
    
    pub _y_data: Vec<f64>,
    pub _yp_data: Vec<f64>,
    pub _id_data: Vec<f64>,
}

#[pymethods]
impl SundialsHandle {
    #[new]
    #[pyo3(signature = (
        lib_path, n, y0, ydot0, id, p, m, n_obs,
        cpr_seeds, cpr_ptrs, cpr_rows, cpr_cols, cpr_dense,
        rel_tol=None, abs_tol=None
    ))]
    pub fn new(
        lib_path: String, n: usize, mut y0: Vec<f64>, mut ydot0: Vec<f64>, mut id: Vec<f64>, 
        p: Vec<f64>, m: Vec<f64>, n_obs: usize, 
        cpr_seeds: Vec<Vec<f64>>, cpr_ptrs: Vec<usize>, cpr_rows: Vec<usize>, cpr_cols: Vec<usize>, cpr_dense: Vec<usize>,
        rel_tol: Option<f64>, abs_tol: Option<f64>
    ) -> PyResult<Self> {
        let lib = unsafe { libloading::Library::new(&lib_path).expect("Failed to load JIT shared library.") };
        let res_fn: NativeResFn = unsafe { *lib.get::<NativeResFn>(b"evaluate_residual\0").unwrap() };
        let jvp_fn: Option<NativeJvpFn> = unsafe { lib.get::<NativeJvpFn>(b"evaluate_jvp\0").map(|s| *s).ok() };
        let vjp_fn: Option<NativeVjpFn> = unsafe { lib.get::<NativeVjpFn>(b"evaluate_vjp\0").map(|s| *s).ok() };
        let obs_fn: Option<NativeObsFn> = unsafe { lib.get::<NativeObsFn>(b"evaluate_observables\0").map(|s| *s).ok() };
        
        let cpr = CprData { color_seeds: cpr_seeds, color_ptrs: cpr_ptrs, color_rows: cpr_rows, color_cols: cpr_cols, dense_rows: cpr_dense };
        let sunctx = unsafe { get_safe_sundials_context() };
        
        let y_vec = unsafe { N_VMake_Serial(n as i64, y0.as_mut_ptr(), sunctx) };
        let yp_vec = unsafe { N_VMake_Serial(n as i64, ydot0.as_mut_ptr(), sunctx) };
        let id_vec = unsafe { N_VMake_Serial(n as i64, id.as_mut_ptr(), sunctx) }; 
        
        let ida_mem = unsafe { IDACreate(sunctx) };
        
        let mut triplets = Vec::with_capacity(n * 50);
        assemble_triplets(n, &y0, &ydot0, &p, &m, 1.0, jvp_fn, vjp_fn, &mut triplets, &cpr, &id);
        
        let mut actual_bw: isize = 0;
        for &(r, c, _) in &triplets {
            let dist: isize = (r as isize - c as isize).abs();
            if dist > actual_bw { actual_bw = dist; }
        }
        let is_banded = actual_bw < (n as isize) / 2;
        let sun_bw = actual_bw;
        
        let user_data = Box::new(SundialsUserData { 
            res_fn, jvp_fn, vjp_fn, obs_fn, p, m, cpr, triplets, is_banded, sun_bw, n, id: id.clone()
        });
        
        unsafe {
            IDAInit(ida_mem, Some(ida_res_callback), 0.0, y_vec, yp_vec);
            IDASStolerances(ida_mem, rel_tol.unwrap_or(1e-6), abs_tol.unwrap_or(1e-8));
            IDASetSuppressAlg(ida_mem, 1);
            IDASetUserData(ida_mem, &*user_data as *const _ as *mut c_void);
            IDASetId(ida_mem, id_vec);
            IDASetMaxNumSteps(ida_mem, 15000);
        }
        
        let (a_mat, ls) = if is_banded {
            let a = unsafe { SUNBandMatrix(n as i64, sun_bw as i64, sun_bw as i64, sunctx) };
            let l = unsafe { SUNLinSol_Band(y_vec, a, sunctx) };
            (a, l)
        } else {
            let a = unsafe { SUNDenseMatrix(n as i64, n as i64, sunctx) };
            let l = unsafe { SUNLinSol_Dense(y_vec, a, sunctx) };
            (a, l)
        };
        
        unsafe {
            IDASetLinearSolver(ida_mem, ls, a_mat);
            IDASetJacFn(ida_mem, Some(ida_jac_callback));
        }
        
        let mut handle = SundialsHandle {
            _lib: lib, ida_mem, y_vec, yp_vec, id_vec, sunctx, a_mat, ls, user_data,
            n, t: 0.0, n_obs, _y_data: y0, _yp_data: ydot0, _id_data: id
        };
        
        handle.calc_algebraic_roots()?;
        Ok(handle)
    }
    
    pub fn step(&mut self, dt: f64) -> PyResult<()> {
        let tout = self.t + dt;
        let mut tret: f64 = 0.0;
        let res = unsafe { IDASolve(self.ida_mem, tout, &mut tret, self.y_vec, self.yp_vec, 1) };
        if res < 0 { return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("SUNDIALS IDASolve failed with error code {}", res))); }
        self.t = tret;
        self.sync_from_sundials();
        Ok(())
    }
    
    pub fn get_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        numpy::ndarray::Array1::from_vec(self._y_data.clone()).to_pyarray(py)
    }

    pub fn get_observables_py<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut obs = vec![0.0; self.n_obs];
        self.get_observables(&mut obs)?;
        Ok(numpy::ndarray::Array1::from_vec(obs).to_pyarray(py))
    }
    
    pub fn set_parameter(&mut self, idx: usize, val: f64) {
        if idx < self.user_data.p.len() { self.user_data.p[idx] = val; }
    }
    
    pub fn calc_algebraic_roots(&mut self) -> PyResult<()> {
        let max_iters = 50;
        for _ in 0..max_iters {
            let mut res = vec![0.0; self.n];
            unsafe { (self.user_data.res_fn)(self._y_data.as_ptr(), self._yp_data.as_ptr(), self.user_data.p.as_ptr(), self.user_data.m.as_ptr(), res.as_mut_ptr()); }

            let mut max_res = 0.0_f64;
            for i in 0..self.n { if self._id_data[i] < 0.5 && res[i].abs() > max_res { max_res = res[i].abs(); } }
            if max_res < 1e-8 { break; }

            assemble_triplets(
                self.n, &self._y_data, &self._yp_data, &self.user_data.p, &self.user_data.m, 0.0,
                self.user_data.jvp_fn, self.user_data.vjp_fn, &mut self.user_data.triplets, &self.user_data.cpr, &self._id_data 
            );

            for &(r, c, val) in &self.user_data.triplets {
                if r == c && self._id_data[r] < 0.5 { 
                    if val.abs() > 1e-12 {
                        let mut step: f64 = -res[r] / val;
                        if step.abs() > 0.1_f64 { step = step.signum() * 0.1_f64; } 
                        self._y_data[r] += step * 0.5; 
                    }
                }
            }
        }

        let y_ptr = unsafe { N_VGetArrayPointer(self.y_vec) };
        for i in 0..self.n { unsafe { *y_ptr.add(i) = self._y_data[i]; } }

        let err = unsafe { IDAReInit(self.ida_mem, self.t, self.y_vec, self.yp_vec) };
        if err < 0 { return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("SUNDIALS IDAReInit failed."))); }
        let ic_res = unsafe { IDACalcIC(self.ida_mem, 1, self.t + 1e-6) }; 
        if ic_res < 0 { 
            let ic_res2 = unsafe { IDACalcIC(self.ida_mem, 1, self.t + 1e-8) };
            if ic_res2 < 0 { return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("SUNDIALS IDACalcIC failed."))); }
        }
        
        self.sync_from_sundials();
        Ok(())
    }

    pub fn reach_steady_state(&mut self) -> PyResult<()> { self.step(1000.0) }
}

impl SundialsHandle {
    pub fn get_observables(&mut self, obs: &mut [f64]) -> PyResult<()> {
        if let Some(f) = self.user_data.obs_fn {
            unsafe { f(self._y_data.as_ptr(), self._yp_data.as_ptr(), self.user_data.p.as_ptr(), self.user_data.m.as_ptr(), obs.as_mut_ptr()); }
        }
        Ok(())
    }

    fn sync_from_sundials(&mut self) {
        let y_ptr = unsafe { N_VGetArrayPointer(self.y_vec) };
        let yp_ptr = unsafe { N_VGetArrayPointer(self.yp_vec) };
        for i in 0..self.n {
            self._y_data[i] = unsafe { *y_ptr.add(i) };
            self._yp_data[i] = unsafe { *yp_ptr.add(i) };
        }
    }
}

impl TimeStepper for SundialsHandle {
    fn t(&self) -> f64 { self.t }
    fn y(&self) -> &[f64] { &self._y_data }
    fn ydot(&self) -> &[f64] { &self._yp_data }
    fn p(&self) -> &[f64] { &self.user_data.p }
    fn p_mut(&mut self) -> &mut [f64] { &mut self.user_data.p }

    fn calc_algebraic_roots(&mut self, _prob: &Problem) -> Result<(), SolverError> {
        SundialsHandle::calc_algebraic_roots(self).map_err(|e| SolverError::Message(e.to_string()))
    }

    fn step(
        &mut self,
        _prob: &Problem,
        dt: f64,
        _history_cache: Option<&mut Vec<(f64, Vec<f64>, Vec<f64>)>>,
    ) -> Result<(), SolverError> {
        SundialsHandle::step(self, dt).map_err(|e| SolverError::Message(e.to_string()))
    }

    fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            t: self.t,
            y: self._y_data.clone(),
            ydot: self._yp_data.clone(),
        }
    }

    fn rollback(&mut self, ckpt: &Checkpoint) {
        self.t = ckpt.t;
        self._y_data.copy_from_slice(&ckpt.y);
        self._yp_data.copy_from_slice(&ckpt.ydot);
        unsafe {
            let y_ptr = N_VGetArrayPointer(self.y_vec);
            let yp_ptr = N_VGetArrayPointer(self.yp_vec);
            for i in 0..self.n {
                *y_ptr.add(i) = self._y_data[i];
                *yp_ptr.add(i) = self._yp_data[i];
            }
            IDAReInit(self.ida_mem, self.t, self.y_vec, self.yp_vec);
        }
    }
}

impl Drop for SundialsHandle {
    fn drop(&mut self) {
        unsafe {
            IDAFree(&mut self.ida_mem); SUNLinSolFree(self.ls); SUNMatDestroy(self.a_mat);
            N_VDestroy(self.y_vec); N_VDestroy(self.yp_vec); N_VDestroy(self.id_vec); SUNContext_Free(&mut self.sunctx);
        }
    }
}

fn evaluate_sundials_ic(lib_path: &str, n: usize, p_list: &[f64], m_list: &[f64], fallback_y0: Vec<f64>, fallback_ydot0: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
    if let Ok(lib) = unsafe { libloading::Library::new(lib_path) } {
        if let Ok(ic_sym) = unsafe { lib.get::<NativeIcFn>(b"evaluate_ic\0") } {
            let ic_fn = *ic_sym;
            let mut y0 = vec![0.0; n];
            let mut ydot0 = vec![0.0; n];
            unsafe { ic_fn(p_list.as_ptr(), m_list.as_ptr(), y0.as_mut_ptr(), ydot0.as_mut_ptr()); }
            return (y0, ydot0);
        }
    }
    (fallback_y0, fallback_ydot0)
}

#[pyfunction]
#[pyo3(signature = (lib_path, y0_py, ydot0_py, id_py, p_list, m_list, t_eval, n_obs, cpr_seeds, cpr_ptrs, cpr_rows, cpr_cols, cpr_dense, show_progress=true, v_idx=-1, rel_tol=None, abs_tol=None))]
pub fn solve_ida_sundials<'py>(
    py: Python<'py>, lib_path: String, y0_py: Vec<f64>, ydot0_py: Vec<f64>, id_py: Vec<f64>, p_list: Vec<f64>, m_list: Vec<f64>, t_eval: Vec<f64>, n_obs: usize, 
    cpr_seeds: Vec<Vec<f64>>, cpr_ptrs: Vec<usize>, cpr_rows: Vec<usize>, cpr_cols: Vec<usize>, cpr_dense: Vec<usize>,
    show_progress: bool, v_idx: i32, rel_tol: Option<f64>, abs_tol: Option<f64>
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {

    let n = y0_py.len();
    let (y0_init, ydot0_init) = evaluate_sundials_ic(&lib_path, n, &p_list, &m_list, y0_py, ydot0_py);
    let mut handle = SundialsHandle::new(
        lib_path, n, y0_init, ydot0_init, id_py, p_list, m_list, n_obs, 
        cpr_seeds, cpr_ptrs, cpr_rows, cpr_cols, cpr_dense, rel_tol, abs_tol
    )?;
    
    let mut out_traj = vec![0.0; t_eval.len() * handle.n];
    let mut out_obs = vec![0.0; t_eval.len() * n_obs];
    for i in 0..handle.n { out_traj[i] = handle._y_data[i]; }
    
    let mut step_obs = vec![0.0; n_obs];
    handle.get_observables(&mut step_obs)?;
    for i in 0..n_obs { out_obs[i] = step_obs[i]; }
    
    let total_steps = t_eval.len().saturating_sub(1);
    
    for step in 1..t_eval.len() {
        let dt = t_eval[step] - t_eval[step - 1];
        handle.step(dt)?;
        for i in 0..handle.n { out_traj[step * handle.n + i] = handle._y_data[i]; }
        
        handle.get_observables(&mut step_obs)?;
        for i in 0..n_obs { out_obs[step * n_obs + i] = step_obs[i]; }
        
        if show_progress && total_steps > 0 {
            let is_final = step == total_steps;
            let pct = (step as f64 / total_steps as f64).clamp(0.0, 1.0);
            let v_str = if v_idx >= 0 { format!(" | V: {:.3}V", handle._y_data[v_idx as usize]) } else { String::new() };
            if is_final {
                print!("\r▶ {:<4} [██████████████████████████████] 100.0% | t: {:.1}s{}   \n", "Sund", t_eval[step], v_str);
            } else {
                let filled = (pct * 30.0) as usize;
                let bar: String = std::iter::repeat('█').take(filled).chain(std::iter::repeat('-').take(30 - filled)).collect();
                print!("\r▶ {:<4} [{}] {:.1}% | t: {:.1}s{}   ", "Sund", bar, pct * 100.0, t_eval[step], v_str);
            }
            std::io::stdout().flush().unwrap();
        }
    }
    
    let res_y = numpy::ndarray::Array2::from_shape_vec((t_eval.len(), handle.n), out_traj).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?.to_pyarray(py);
    let res_obs = numpy::ndarray::Array2::from_shape_vec((t_eval.len(), n_obs), out_obs).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?.to_pyarray(py);
    let empty_t = numpy::ndarray::Array1::<f64>::zeros(0).to_pyarray(py);
    let empty_y = numpy::ndarray::Array2::<f64>::zeros((0, handle.n)).to_pyarray(py);
    Ok((res_y, res_obs, empty_t, empty_y.clone(), empty_y))
}

#[pyfunction]
#[pyo3(signature = (lib_path, y0_py, ydot0_py, id_py, p_list, m_list, n_obs, cpr_seeds, cpr_ptrs, cpr_rows, cpr_cols, cpr_dense, protocol_steps, show_progress=true, v_idx=-1, rel_tol=None, abs_tol=None))]
pub fn solve_ida_sequence<'py>(
    py: Python<'py>,
    lib_path: String,
    y0_py: Vec<f64>,
    ydot0_py: Vec<f64>,
    id_py: Vec<f64>,
    p_list: Vec<f64>,
    m_list: Vec<f64>,
    n_obs: usize,
    cpr_seeds: Vec<Vec<f64>>,
    cpr_ptrs: Vec<usize>,
    cpr_rows: Vec<usize>,
    cpr_cols: Vec<usize>,
    cpr_dense: Vec<usize>,
    protocol_steps: Vec<RawProtocolStep>,
    show_progress: bool,
    v_idx: i32,
    rel_tol: Option<f64>,
    abs_tol: Option<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let n = y0_py.len();
    let (y0_init, ydot0_init) = evaluate_sundials_ic(&lib_path, n, &p_list, &m_list, y0_py, ydot0_py);
    let mut handle = SundialsHandle::new(
        lib_path,
        n,
        y0_init,
        ydot0_init,
        id_py.clone(),
        p_list,
        m_list.clone(),
        n_obs,
        cpr_seeds,
        cpr_ptrs,
        cpr_rows,
        cpr_cols,
        cpr_dense,
        rel_tol,
        abs_tol,
    )?;

    let prob = Problem {
        n,
        bw: 0,
        n_obs,
        id: id_py,
        constraints: vec![0.0; n],
        m: m_list,
        spatial_diag: vec![0.0; n],
        max_steps: vec![0.0; n],
        cpr: handle.user_data.cpr.clone(),
        config: SolverConfig::default(),
        fns: Callbacks {
            res_fn: handle.user_data.res_fn,
            obs_fn: handle.user_data.obs_fn,
            jvp_fn: handle.user_data.jvp_fn,
            vjp_fn: handle.user_data.vjp_fn,
            set_threads_fn: None,
        },
    };

    let steps = parse_protocol_steps(&protocol_steps);
    let mut out_t = Vec::new();
    let mut out_traj = Vec::new();
    let mut out_obs = Vec::new();

    run_sequence(&prob, &mut handle, &steps, &mut out_t, &mut out_traj, &mut out_obs, None, None, show_progress, v_idx)
        .map_err(|e| match e {
            SolverError::Crash(r) => pyo3::exceptions::PyRuntimeError::new_err(r.reason),
            SolverError::Message(m) => pyo3::exceptions::PyRuntimeError::new_err(m),
        })?;

    let len_t = out_t.len();
    let res_t = numpy::ndarray::Array1::from_vec(out_t).to_pyarray(py);
    let res_y = numpy::ndarray::Array2::from_shape_vec((len_t, n), out_traj)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        .to_pyarray(py);
    let res_obs = numpy::ndarray::Array2::from_shape_vec((len_t, n_obs), out_obs)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        .to_pyarray(py);

    Ok((res_t, res_y, res_obs))
}
