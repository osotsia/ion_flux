use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, ToPyArray};
use crate::ffi::callbacks::{Callbacks, NativeResFn, NativeObsFn, NativeJvpFn, NativeVjpFn, NativeIcFn, NativeSetThreadsFn};
use crate::stepper::problem::CprData;
use crate::linear::jacobian::assemble_triplets_analytical;

#[pyclass]
pub struct NativeLibrary {
    _lib: libloading::Library,
    #[pyo3(get)]
    pub lib_path: String,
    pub n_states: usize,
    pub n_params: usize,
    pub n_obs: usize,
    res_fn: NativeResFn,
    obs_fn: Option<NativeObsFn>,
    audit_obs_fn: Option<NativeObsFn>,
    jvp_fn: Option<NativeJvpFn>,
    vjp_fn: Option<NativeVjpFn>,
    ic_fn: Option<NativeIcFn>,
    set_threads_fn: Option<NativeSetThreadsFn>,
}

#[pymethods]
impl NativeLibrary {
    #[new]
    pub fn new(lib_path: String, n_states: usize, n_params: usize, n_obs: usize) -> PyResult<Self> {
        let lib = unsafe {
            libloading::Library::new(&lib_path)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load compiled binary at {}: {}", lib_path, e)))?
        };
        let fns = Callbacks::from_lib(&lib).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        let ic_fn = Callbacks::load_ic(&lib);
        let audit_obs_fn: Option<NativeObsFn> = unsafe { lib.get(b"evaluate_audit_observables\0").map(|s| *s).ok() };

        Ok(Self {
            _lib: lib,
            lib_path,
            n_states,
            n_params,
            n_obs,
            res_fn: fns.res_fn,
            obs_fn: fns.obs_fn,
            audit_obs_fn,
            jvp_fn: fns.jvp_fn,
            vjp_fn: fns.vjp_fn,
            ic_fn,
            set_threads_fn: fns.set_threads_fn,
        })
    }

    pub fn set_spatial_threads(&self, num_threads: i32) {
        if let Some(f) = self.set_threads_fn {
            unsafe { f(num_threads); }
        }
    }

    pub fn evaluate_ic(&self, p: Vec<f64>, m: Vec<f64>) -> PyResult<(Vec<f64>, Vec<f64>)> {
        if p.len() != self.n_params {
            return Err(pyo3::exceptions::PyValueError::new_err("Parameter slice length mismatch"));
        }
        let mut y0 = vec![0.0; self.n_states];
        let mut ydot0 = vec![0.0; self.n_states];
        if let Some(ic_fn) = self.ic_fn {
            unsafe { ic_fn(p.as_ptr(), m.as_ptr(), y0.as_mut_ptr(), ydot0.as_mut_ptr()); }
        }
        Ok((y0, ydot0))
    }

    pub fn evaluate_residual(&self, y: Vec<f64>, ydot: Vec<f64>, p: Vec<f64>, m: Vec<f64>) -> PyResult<Vec<f64>> {
        if y.len() != self.n_states || ydot.len() != self.n_states || p.len() != self.n_params {
            return Err(pyo3::exceptions::PyValueError::new_err("State or parameter slice length mismatch"));
        }
        let mut res = vec![0.0; self.n_states];
        unsafe { (self.res_fn)(y.as_ptr(), ydot.as_ptr(), p.as_ptr(), m.as_ptr(), res.as_mut_ptr()); }
        Ok(res)
    }

    pub fn evaluate_observables(&self, y: Vec<f64>, ydot: Vec<f64>, p: Vec<f64>, m: Vec<f64>) -> PyResult<Vec<f64>> {
        if y.len() != self.n_states || ydot.len() != self.n_states || p.len() != self.n_params {
            return Err(pyo3::exceptions::PyValueError::new_err("State or parameter slice length mismatch"));
        }
        let mut obs = vec![0.0; self.n_obs];
        if let Some(obs_fn) = self.obs_fn {
            unsafe { obs_fn(y.as_ptr(), ydot.as_ptr(), p.as_ptr(), m.as_ptr(), obs.as_mut_ptr()); }
        }
        Ok(obs)
    }

    pub fn evaluate_audit_observables(&self, y: Vec<f64>, ydot: Vec<f64>, p: Vec<f64>, m: Vec<f64>, n_audit_obs: usize) -> PyResult<Vec<f64>> {
        if y.len() != self.n_states || ydot.len() != self.n_states || p.len() != self.n_params {
            return Err(pyo3::exceptions::PyValueError::new_err("State or parameter slice length mismatch"));
        }
        let mut audit_obs = vec![0.0; n_audit_obs];
        if n_audit_obs > 0 {
            if let Some(audit_fn) = self.audit_obs_fn {
                unsafe { audit_fn(y.as_ptr(), ydot.as_ptr(), p.as_ptr(), m.as_ptr(), audit_obs.as_mut_ptr()); }
            }
        }
        Ok(audit_obs)
    }

    pub fn evaluate_jvp<'py>(&self, py: Python<'py>, y: Vec<f64>, ydot: Vec<f64>, p: Vec<f64>, m: Vec<f64>, c_j: f64, v: Vec<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if y.len() != self.n_states || ydot.len() != self.n_states || v.len() != self.n_states || p.len() != self.n_params {
            return Err(pyo3::exceptions::PyValueError::new_err("State, vector, or parameter slice length mismatch"));
        }
        let jvp_fn = self.jvp_fn.ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("evaluate_jvp missing"))?;
        let mut jvp_dy = vec![0.0; self.n_states];
        let mut jvp_dydot = vec![0.0; self.n_states];
        let mut res_dummy = vec![0.0; self.n_states];
        let mut jvp_out = vec![0.0; self.n_states];
        unsafe {
            jvp_fn(
                y.as_ptr(), ydot.as_ptr(), p.as_ptr(), m.as_ptr(), c_j, v.as_ptr(),
                jvp_dy.as_mut_ptr(), jvp_dydot.as_mut_ptr(), res_dummy.as_mut_ptr(), jvp_out.as_mut_ptr(),
            );
        }
        Ok(numpy::ndarray::Array1::from_vec(jvp_out).to_pyarray(py))
    }

    #[pyo3(signature = (y, ydot, p, m, c_j, cpr_seeds, cpr_ptrs, cpr_rows, cpr_cols, cpr_dense))]
    pub fn evaluate_jacobian_triplets(
        &self,
        y: Vec<f64>,
        ydot: Vec<f64>,
        p: Vec<f64>,
        m: Vec<f64>,
        c_j: f64,
        cpr_seeds: Vec<Vec<f64>>,
        cpr_ptrs: Vec<usize>,
        cpr_rows: Vec<usize>,
        cpr_cols: Vec<usize>,
        cpr_dense: Vec<usize>,
    ) -> PyResult<(Vec<usize>, Vec<usize>, Vec<f64>)> {
        if y.len() != self.n_states || ydot.len() != self.n_states || p.len() != self.n_params {
            return Err(pyo3::exceptions::PyValueError::new_err("State or parameter slice length mismatch"));
        }
        let n = self.n_states;
        let cpr = CprData {
            color_seeds: cpr_seeds,
            color_ptrs: cpr_ptrs,
            color_rows: cpr_rows,
            color_cols: cpr_cols,
            dense_rows: cpr_dense,
        };
        let mut triplets = Vec::with_capacity(cpr.color_rows.len() + cpr.dense_rows.len() * n);
        assemble_triplets_analytical(n, &y, &ydot, &p, &m, c_j, self.jvp_fn, self.vjp_fn, &mut triplets, &cpr);
        let mut rows = Vec::with_capacity(triplets.len());
        let mut cols = Vec::with_capacity(triplets.len());
        let mut vals = Vec::with_capacity(triplets.len());
        for (r, c, v) in triplets {
            rows.push(r);
            cols.push(c);
            vals.push(v);
        }
        Ok((rows, cols, vals))
    }

    pub fn evaluate_vjp(&self, y: Vec<f64>, ydot: Vec<f64>, p: Vec<f64>, m: Vec<f64>, lambda_vec: Vec<f64>) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        if y.len() != self.n_states || ydot.len() != self.n_states || lambda_vec.len() != self.n_states || p.len() != self.n_params {
            return Err(pyo3::exceptions::PyValueError::new_err("State, lambda, or parameter slice length mismatch"));
        }
        let vjp_fn = self.vjp_fn.ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("evaluate_vjp missing"))?;
        let mut res_dummy = vec![0.0; self.n_states];
        let mut dp_out = vec![0.0; self.n_params];
        let mut dy_out = vec![0.0; self.n_states];
        let mut dydot_out = vec![0.0; self.n_states];
        unsafe {
            vjp_fn(
                y.as_ptr(), ydot.as_ptr(), p.as_ptr(), m.as_ptr(), lambda_vec.as_ptr(),
                res_dummy.as_mut_ptr(), dp_out.as_mut_ptr(), dy_out.as_mut_ptr(), dydot_out.as_mut_ptr(),
            );
        }
        Ok((dp_out, dy_out, dydot_out))
    }

    #[pyo3(signature = (y, ydot, p, m, c_j, cpr_seeds, cpr_ptrs, cpr_rows, cpr_cols, cpr_dense))]
    pub fn evaluate_jacobian<'py>(
        &self,
        py: Python<'py>,
        y: Vec<f64>,
        ydot: Vec<f64>,
        p: Vec<f64>,
        m: Vec<f64>,
        c_j: f64,
        cpr_seeds: Vec<Vec<f64>>,
        cpr_ptrs: Vec<usize>,
        cpr_rows: Vec<usize>,
        cpr_cols: Vec<usize>,
        cpr_dense: Vec<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if y.len() != self.n_states || ydot.len() != self.n_states || p.len() != self.n_params {
            return Err(pyo3::exceptions::PyValueError::new_err("State or parameter slice length mismatch"));
        }
        let n = self.n_states;
        let cpr = CprData {
            color_seeds: cpr_seeds,
            color_ptrs: cpr_ptrs,
            color_rows: cpr_rows,
            color_cols: cpr_cols,
            dense_rows: cpr_dense,
        };
        let mut triplets = Vec::with_capacity(cpr.color_rows.len() + cpr.dense_rows.len() * n);
        assemble_triplets_analytical(n, &y, &ydot, &p, &m, c_j, self.jvp_fn, self.vjp_fn, &mut triplets, &cpr);

        let mut j_flat = vec![0.0; n * n];
        for (r, c, val) in triplets {
            j_flat[r * n + c] = val;
        }
        Ok(numpy::ndarray::Array2::from_shape_vec((n, n), j_flat).unwrap().to_pyarray(py))
    }

    #[pyo3(signature = (y, ydot, p, m, input_param_idx, cpr_seeds, cpr_ptrs, cpr_rows, cpr_cols, cpr_dense))]
    pub fn extract_eis_system<'py>(
        &self,
        py: Python<'py>,
        y: Vec<f64>,
        ydot: Vec<f64>,
        p: Vec<f64>,
        m: Vec<f64>,
        input_param_idx: usize,
        cpr_seeds: Vec<Vec<f64>>,
        cpr_ptrs: Vec<usize>,
        cpr_rows: Vec<usize>,
        cpr_cols: Vec<usize>,
        cpr_dense: Vec<usize>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
        if y.len() != self.n_states || ydot.len() != self.n_states || p.len() != self.n_params {
            return Err(pyo3::exceptions::PyValueError::new_err("State or parameter slice length mismatch"));
        }
        if input_param_idx >= self.n_params {
            return Err(pyo3::exceptions::PyValueError::new_err("input_param_idx out of bounds"));
        }
        let vjp_fn = self.vjp_fn.ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("evaluate_vjp missing"))?;
        let n = self.n_states;
        let cpr = CprData {
            color_seeds: cpr_seeds,
            color_ptrs: cpr_ptrs,
            color_rows: cpr_rows,
            color_cols: cpr_cols,
            dense_rows: cpr_dense,
        };

        let mut triplets = Vec::with_capacity(cpr.color_rows.len() + cpr.dense_rows.len() * n);
        assemble_triplets_analytical(n, &y, &ydot, &p, &m, 0.0, self.jvp_fn, self.vjp_fn, &mut triplets, &cpr);
        let mut j_steady = vec![0.0; n * n];
        for &(r, c, val) in &triplets {
            j_steady[r * n + c] = val;
        }

        assemble_triplets_analytical(n, &y, &ydot, &p, &m, 1.0, self.jvp_fn, self.vjp_fn, &mut triplets, &cpr);
        let mut j_1 = vec![0.0; n * n];
        for &(r, c, val) in &triplets {
            j_1[r * n + c] = val;
        }

        let mut mass = vec![0.0; n * n];
        for i in 0..(n * n) {
            mass[i] = j_1[i] - j_steady[i];
        }

        let mut df_dp_input = vec![0.0; n];
        let mut lam = vec![0.0; n];
        let mut res_dummy = vec![0.0; n];
        let mut dp_out = vec![0.0; self.n_params];
        let mut dy_out = vec![0.0; n];
        let mut dydot_out = vec![0.0; n];

        for i in 0..n {
            lam[i] = 1.0;
            unsafe {
                vjp_fn(
                    y.as_ptr(), ydot.as_ptr(), p.as_ptr(), m.as_ptr(), lam.as_ptr(),
                    res_dummy.as_mut_ptr(), dp_out.as_mut_ptr(), dy_out.as_mut_ptr(), dydot_out.as_mut_ptr(),
                );
            }
            df_dp_input[i] = dp_out[input_param_idx];
            lam[i] = 0.0;
        }

        Ok((
            numpy::ndarray::Array2::from_shape_vec((n, n), mass).unwrap().to_pyarray(py),
            numpy::ndarray::Array2::from_shape_vec((n, n), j_steady).unwrap().to_pyarray(py),
            numpy::ndarray::Array1::from_vec(df_dp_input).to_pyarray(py),
        ))
    }
}
