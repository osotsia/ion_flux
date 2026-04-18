#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use pyo3::prelude::*;
use numpy::{PyArray1, ToPyArray};
use std::os::raw::{c_char, c_double, c_int, c_long, c_void};
use super::{NativeResFn, NativeJacFn, NativeObsFn};

pub type SunRealType = c_double;
pub type SunIndexType = c_long; 

#[repr(C)] pub struct _generic_SUNContext { _private: [u8; 0] }
pub type SUNContext = *mut _generic_SUNContext;
#[repr(C)] pub struct _generic_N_Vector { _private: [u8; 0] }
pub type N_Vector = *mut _generic_N_Vector;
#[repr(C)] pub struct _generic_SUNMatrix { _private:[u8; 0] }
pub type SUNMatrix = *mut _generic_SUNMatrix;
#[repr(C)] pub struct _generic_SUNLinearSolver { _private: [u8; 0] }
pub type SUNLinearSolver = *mut _generic_SUNLinearSolver;

extern "C" {
    pub fn SUNContext_Create(comm: *mut c_void, ctx: *mut SUNContext) -> c_int;
    pub fn SUNContext_Free(ctx: *mut SUNContext) -> c_int;
    
    pub fn N_VMake_Serial(vec_length: SunIndexType, v_data: *mut SunRealType, ctx: SUNContext) -> N_Vector;
    pub fn N_VDestroy(v: N_Vector);
    pub fn N_VGetArrayPointer(v: N_Vector) -> *mut SunRealType;
    
    pub fn SUNDenseMatrix(M: SunIndexType, N: SunIndexType, ctx: SUNContext) -> SUNMatrix;
    pub fn SUNMatDestroy(A: SUNMatrix);
    pub fn SUNDenseMatrix_Data(A: SUNMatrix) -> *mut SunRealType;
    
    pub fn SUNLinSol_Dense(y: N_Vector, A: SUNMatrix, ctx: SUNContext) -> SUNLinearSolver;
    pub fn SUNLinSolFree(LS: SUNLinearSolver);
    pub fn SUNBandMatrix(N: SunIndexType, mupper: SunIndexType, mlower: SunIndexType, ctx: SUNContext) -> SUNMatrix;
    pub fn SUNLinSol_Band(y: N_Vector, A: SUNMatrix, ctx: SUNContext) -> SUNLinearSolver;
    pub fn SUNBandMatrix_Data(A: SUNMatrix) -> *mut SunRealType;
    pub fn SUNBandMatrix_LDim(A: SUNMatrix) -> SunIndexType;
    pub fn SUNBandMatrix_UpperBandwidth(A: SUNMatrix) -> SunIndexType;
    
    pub fn IDACreate(ctx: SUNContext) -> *mut c_void;
    pub fn IDAFree(ida_mem: *mut *mut c_void);
    pub fn IDAInit(ida_mem: *mut c_void, res: Option<unsafe extern "C" fn(SunRealType, N_Vector, N_Vector, N_Vector, *mut c_void) -> c_int>, t0: SunRealType, yy0: N_Vector, yp0: N_Vector) -> c_int;
    pub fn IDAReInit(ida_mem: *mut c_void, t0: SunRealType, yy0: N_Vector, yp0: N_Vector) -> c_int;
    pub fn IDASStolerances(ida_mem: *mut c_void, reltol: SunRealType, abstol: SunRealType) -> c_int;
    pub fn IDASetSuppressAlg(ida_mem: *mut c_void, suppressalg: c_int) -> c_int;
    pub fn IDASetUserData(ida_mem: *mut c_void, user_data: *mut c_void) -> c_int;
    pub fn IDASetId(ida_mem: *mut c_void, id: N_Vector) -> c_int;
    pub fn IDASetLinearSolver(ida_mem: *mut c_void, LS: SUNLinearSolver, A: SUNMatrix) -> c_int;
    pub fn IDASetJacFn(ida_mem: *mut c_void, jac: Option<unsafe extern "C" fn(SunRealType, SunRealType, N_Vector, N_Vector, N_Vector, SUNMatrix, *mut c_void, N_Vector, N_Vector, N_Vector) -> c_int>) -> c_int;
    pub fn IDACalcIC(ida_mem: *mut c_void, icopt: c_int, tout1: SunRealType) -> c_int;
    pub fn IDASolve(ida_mem: *mut c_void, tout: SunRealType, tret: *mut SunRealType, yret: N_Vector, ypret: N_Vector, itask: c_int) -> c_int;
    pub fn IDASetMaxNumSteps(ida_mem: *mut c_void, mxsteps: c_long) -> c_int;
    
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
}

pub struct SundialsUserData {
    pub res_fn: NativeResFn,
    pub jac_fn: NativeJacFn,
    pub obs_fn: Option<NativeObsFn>,
    pub p: Vec<f64>,
    pub m: Vec<f64>,
    pub jac_buffer: Vec<f64>,
    pub is_banded: bool,
    pub sun_bw: isize,
}

unsafe extern "C" fn ida_res_callback(_t: c_double, y: N_Vector, yp: N_Vector, res: N_Vector, user_data: *mut c_void) -> c_int {
    let ud = &*(user_data as *const SundialsUserData);
    (ud.res_fn)(N_VGetArrayPointer(y), N_VGetArrayPointer(yp), ud.p.as_ptr(), ud.m.as_ptr(), N_VGetArrayPointer(res));
    0
}

unsafe extern "C" fn ida_jac_callback(_t: c_double, c_j: c_double, y: N_Vector, yp: N_Vector, _r: N_Vector, jac: SUNMatrix, user_data: *mut c_void, _t1: N_Vector, _t2: N_Vector, _t3: N_Vector) -> c_int {
    let ud = &mut *(user_data as *mut SundialsUserData);
    let n = (ud.jac_buffer.len() as f64).sqrt() as usize;

    (ud.jac_fn)(N_VGetArrayPointer(y), N_VGetArrayPointer(yp), ud.p.as_ptr(), ud.m.as_ptr(), c_j, ud.jac_buffer.as_mut_ptr());

    if ud.is_banded {
        let a_data = SUNBandMatrix_Data(jac);
        let ldim = SUNBandMatrix_LDim(jac) as usize;
        let mu = SUNBandMatrix_UpperBandwidth(jac) as isize;

        for c in 0..n {
            let start_r = (c as isize - ud.sun_bw).max(0) as usize;
            let end_r = (c as isize + ud.sun_bw).min(n as isize - 1) as usize;
            for r in start_r..=end_r {
                let band_idx = (r as isize - c as isize + mu) as usize + c * ldim;
                *a_data.add(band_idx) = ud.jac_buffer[c * n + r];
            }
        }
    } else {
        let a_data = SUNDenseMatrix_Data(jac);
        for i in 0..(n*n) {
            *a_data.add(i) = ud.jac_buffer[i];
        }
    }
    0
}

/// Safely generates a SUNContext by querying the currently loaded memory mapping.
/// If SUNDIALS was linked with MPI, those symbols will be forcefully resolved to ensure
/// OpenMPI receives the correct `ompi_mpi_comm_world` instance it expects.
unsafe fn get_safe_sundials_context() -> SUNContext {
    let mut sunctx: SUNContext = std::ptr::null_mut();
    let mut comm: *mut c_void = std::ptr::null_mut();
    
    #[cfg(unix)]
    {
        #[cfg(target_os = "macos")] let rtld_search = -2isize as *mut c_void; 
        #[cfg(not(target_os = "macos"))] let rtld_search = 0isize as *mut c_void;  
        
        let mut get_sym = |name: &str| -> *mut c_void {
            let c_name = std::ffi::CString::new(name).unwrap();
            let mut ptr = dlsym(rtld_search, c_name.as_ptr());
            
            if ptr.is_null() {
                #[cfg(target_os = "macos")] let flags = 2 | 8; 
                #[cfg(target_os = "linux")] let flags = 2 | 256; 
                #[cfg(not(any(target_os = "macos", target_os = "linux")))] let flags = 2;
                
                let libs =["libsundials_core.dylib\0", "libsundials_core.so\0", "libsundials_core.7.6.0.dylib\0"];
                for lib in &libs {
                    let h = dlopen(lib.as_ptr() as *const c_char, flags);
                    if !h.is_null() {
                        ptr = dlsym(h, c_name.as_ptr());
                        if !ptr.is_null() { break; }
                    }
                }
            }
            ptr
        };
        
        let mpi_init_ptr = get_sym("MPI_Init");
        if !mpi_init_ptr.is_null() {
            let mpi_init: unsafe extern "C" fn(*mut c_int, *mut *mut *mut i8) -> c_int = std::mem::transmute(mpi_init_ptr);
            let mpi_init_check_ptr = get_sym("MPI_Initialized");
            
            if !mpi_init_check_ptr.is_null() {
                let mpi_init_check: unsafe extern "C" fn(*mut c_int) -> c_int = std::mem::transmute(mpi_init_check_ptr);
                let mut flag = 0;
                mpi_init_check(&mut flag);
                if flag == 0 { mpi_init(std::ptr::null_mut(), std::ptr::null_mut()); }
            }
            
            let ompi_world = get_sym("ompi_mpi_comm_world");
            if !ompi_world.is_null() {
                comm = ompi_world;
            } else {
                let mpich_sym = get_sym("MPIR_Comm_direct");
                if !mpich_sym.is_null() { comm = 0x44000000_usize as *mut c_void; }
            }
        }
    }
    
    SUNContext_Create(comm, &mut sunctx);
    sunctx
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
    pub fn new(lib_path: String, n: usize, mut y0: Vec<f64>, mut ydot0: Vec<f64>, mut id: Vec<f64>, p: Vec<f64>, m: Vec<f64>, n_obs: usize) -> PyResult<Self> {
        let lib = unsafe { libloading::Library::new(&lib_path).expect("Failed to load JIT shared library.") };
        let res_fn: NativeResFn = unsafe { *lib.get::<NativeResFn>(b"evaluate_residual\0").unwrap() };
        let jac_fn: NativeJacFn = unsafe { *lib.get::<NativeJacFn>(b"evaluate_jacobian\0").unwrap() };
        let obs_fn: Option<NativeObsFn> = unsafe { lib.get::<NativeObsFn>(b"evaluate_observables\0").map(|s| *s).ok() };
        
        let sunctx = unsafe { get_safe_sundials_context() };
        
        let y_vec = unsafe { N_VMake_Serial(n as i64, y0.as_mut_ptr(), sunctx) };
        let yp_vec = unsafe { N_VMake_Serial(n as i64, ydot0.as_mut_ptr(), sunctx) };
        let id_vec = unsafe { N_VMake_Serial(n as i64, id.as_mut_ptr(), sunctx) };
        
        let ida_mem = unsafe { IDACreate(sunctx) };
        
        // Dynamically detect the physical bandwidth from the initial Jacobian
        let mut jac_dense = vec![0.0; n * n];
        unsafe { (jac_fn)(y0.as_ptr(), ydot0.as_ptr(), p.as_ptr(), m.as_ptr(), 1.0, jac_dense.as_mut_ptr()) };
        
        let mut actual_bw = 0;
        for c in 0..n {
            for r in 0..n {
                if jac_dense[c * n + r].abs() > 1e-12 {
                    let dist = (r as isize - c as isize).abs();
                    if dist > actual_bw { actual_bw = dist; }
                }
            }
        }
        
        // Threshold heuristic: Use Band solver if bandwidth is less than half the total states
        let is_banded = actual_bw < (n as isize) / 2;
        let sun_bw = actual_bw;
        
        let user_data = Box::new(SundialsUserData { 
            res_fn, jac_fn, obs_fn, p, m, 
            jac_buffer: jac_dense, is_banded, sun_bw 
        });
        
        unsafe {
            IDAInit(ida_mem, Some(ida_res_callback), 0.0, y_vec, yp_vec);
            IDASStolerances(ida_mem, 1e-6, 1e-8);
            IDASetSuppressAlg(ida_mem, 1); // CRITICAL: Prevent algebraic variables from dominating the truncation error step limiter
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
        if res < 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("SUNDIALS IDASolve failed with error code {}", res)));
        }
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
            for i in 0..self.n {
                if self._id_data[i] < 0.5 && res[i].abs() > max_res { max_res = res[i].abs(); }
            }
            if max_res < 1e-8 { break; }

            let mut jac = vec![0.0; self.n * self.n];
            unsafe { (self.user_data.jac_fn)(self._y_data.as_ptr(), self._yp_data.as_ptr(), self.user_data.p.as_ptr(), self.user_data.m.as_ptr(), 0.0, jac.as_mut_ptr()); }

            for i in 0..self.n {
                if self._id_data[i] < 0.5 { 
                    let diag = jac[i * self.n + i];
                    if diag.abs() > 1e-12 {
                        let mut step = -res[i] / diag;
                        if step.abs() > 0.1 { step = step.signum() * 0.1; } // Safety clamp
                        self._y_data[i] += step * 0.5; // Under-relax diagonal step
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
            if ic_res2 < 0 {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("SUNDIALS IDACalcIC failed."))); 
            }
        }
        
        self.sync_from_sundials();
        Ok(())
    }

    pub fn reach_steady_state(&mut self) -> PyResult<()> {
        self.step(1000.0)
    }
}

// Internal Rust helpers
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

impl Drop for SundialsHandle {
    fn drop(&mut self) {
        unsafe {
            IDAFree(&mut self.ida_mem);
            SUNLinSolFree(self.ls);
            SUNMatDestroy(self.a_mat);
            N_VDestroy(self.y_vec);
            N_VDestroy(self.yp_vec);
            N_VDestroy(self.id_vec);
            SUNContext_Free(&mut self.sunctx);
        }
    }
}