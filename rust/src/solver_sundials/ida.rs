#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use pyo3::prelude::*;
use numpy::{PyArray2, ToPyArray};
use std::os::raw::{c_void, c_int, c_long, c_double};

// -----------------------------------------------------------------------------
// Minimalist SUNDIALS C-ABI Bindings (Targeting v6 / v7+)
// -----------------------------------------------------------------------------
pub enum N_VectorOps {}
pub type N_Vector = *mut N_VectorOps;

extern "C" {
    fn SUNContext_Create(comm: *mut c_void, ctx: *mut *mut c_void) -> c_int;
    fn SUNContext_Free(ctx: *mut *mut c_void) -> c_int;
    
    fn N_VNew_Serial(vec_length: c_long, ctx: *mut c_void) -> N_Vector;
    fn N_VGetArrayPointer(v: N_Vector) -> *mut c_double;
    fn N_VDestroy(v: N_Vector);
    
    fn SUNDenseMatrix(M: c_long, N: c_long, ctx: *mut c_void) -> *mut c_void;
    fn SUNDenseMatrix_Data(A: *mut c_void) -> *mut c_double; // Replaces SUNMatGetArray_Dense in v7+
    fn SUNMatDestroy(A: *mut c_void);
    
    fn SUNLinSol_Dense(y: N_Vector, A: *mut c_void, ctx: *mut c_void) -> *mut c_void;
    fn SUNLinSolFree(S: *mut c_void);
    
    fn IDACreate(ctx: *mut c_void) -> *mut c_void;
    fn IDAInit(ida_mem: *mut c_void, res: extern "C" fn(c_double, N_Vector, N_Vector, N_Vector, *mut c_void) -> c_int, t0: c_double, yy0: N_Vector, yp0: N_Vector) -> c_int;
    fn IDASStolerances(ida_mem: *mut c_void, reltol: c_double, abstol: c_double) -> c_int;
    fn IDASetUserData(ida_mem: *mut c_void, user_data: *mut c_void) -> c_int;
    fn IDASetId(ida_mem: *mut c_void, id: N_Vector) -> c_int;
    fn IDASetLinearSolver(ida_mem: *mut c_void, LS: *mut c_void, A: *mut c_void) -> c_int;
    fn IDASetJacFn(ida_mem: *mut c_void, jac: extern "C" fn(c_double, c_double, N_Vector, N_Vector, N_Vector, *mut c_void, *mut c_void, N_Vector, N_Vector, N_Vector) -> c_int) -> c_int;
    fn IDACalcIC(ida_mem: *mut c_void, icopt: c_int, tout1: c_double) -> c_int;
    fn IDASolve(ida_mem: *mut c_void, tout: c_double, tret: *mut c_double, yret: N_Vector, ypret: N_Vector, itask: c_int) -> c_int;
    fn IDAFree(ida_mem: *mut *mut c_void);
}

// -----------------------------------------------------------------------------
// JIT Integration State and Callbacks
// -----------------------------------------------------------------------------
type NativeResFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *mut c_double);
type NativeJacFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, c_double, *mut c_double);

struct UserData<'a> {
    res_fn: libloading::Symbol<'a, NativeResFn>,
    jac_fn: libloading::Symbol<'a, NativeJacFn>,
    p: Vec<c_double>,
}

extern "C" fn rust_res_cb(_t: c_double, yy: N_Vector, yp: N_Vector, rr: N_Vector, user_data: *mut c_void) -> c_int {
    unsafe {
        let ud = &*(user_data as *const UserData);
        let y_ptr = N_VGetArrayPointer(yy);
        let yp_ptr = N_VGetArrayPointer(yp);
        let r_ptr = N_VGetArrayPointer(rr);
        (ud.res_fn)(y_ptr, yp_ptr, ud.p.as_ptr(), r_ptr);
    }
    0
}

extern "C" fn rust_jac_cb(
    _t: c_double, 
    c_j: c_double, 
    yy: N_Vector, 
    yp: N_Vector, 
    _rr: N_Vector, 
    jac: *mut c_void, 
    user_data: *mut c_void, 
    _t1: N_Vector, 
    _t2: N_Vector, 
    _t3: N_Vector
) -> c_int {
    unsafe {
        let ud = &*(user_data as *const UserData);
        let y_ptr = N_VGetArrayPointer(yy);
        let yp_ptr = N_VGetArrayPointer(yp);
        let jac_ptr = SUNDenseMatrix_Data(jac);
        (ud.jac_fn)(y_ptr, yp_ptr, ud.p.as_ptr(), c_j, jac_ptr);
    }
    0
}

// -----------------------------------------------------------------------------
// PyO3 Orchestrator
// -----------------------------------------------------------------------------
#[pyfunction]
pub fn solve_ida_sundials<'py>(
    py: Python<'py>,
    lib_path: String,
    y0_py: Vec<f64>,
    ydot0_py: Vec<f64>,
    id_py: Vec<f64>,
    p_list: Vec<f64>,
    t_eval: Vec<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    
    let n_states = y0_py.len() as c_long;
    let n_steps = t_eval.len();
    
    let mut out_traj = vec![0.0; n_steps * (n_states as usize)];

    unsafe {
        let lib = libloading::Library::new(lib_path).expect("Failed to load JIT shared library.");
        let res_fn: libloading::Symbol<NativeResFn> = lib.get(b"evaluate_residual\0").unwrap();
        let jac_fn: libloading::Symbol<NativeJacFn> = lib.get(b"evaluate_jacobian\0").unwrap();

        let mut user_data = UserData {
            res_fn,
            jac_fn,
            p: p_list,
        };

        let mut ctx: *mut c_void = std::ptr::null_mut();
        SUNContext_Create(std::ptr::null_mut(), &mut ctx);

        let ida_mem = IDACreate(ctx);
        
        let y = N_VNew_Serial(n_states, ctx);
        let ydot = N_VNew_Serial(n_states, ctx);
        let id = N_VNew_Serial(n_states, ctx);
        
        let y_ptr = N_VGetArrayPointer(y);
        let ydot_ptr = N_VGetArrayPointer(ydot);
        let id_ptr = N_VGetArrayPointer(id);
        
        for i in 0..(n_states as usize) {
            *y_ptr.add(i) = y0_py[i];
            *ydot_ptr.add(i) = ydot0_py[i];
            *id_ptr.add(i) = id_py[i];
        }

        IDAInit(ida_mem, rust_res_cb, t_eval[0], y, ydot);
        IDASStolerances(ida_mem, 1e-6, 1e-8);
        IDASetUserData(ida_mem, &mut user_data as *mut _ as *mut c_void);
        IDASetId(ida_mem, id);

        let jac_mat = SUNDenseMatrix(n_states, n_states, ctx);
        let lin_sol = SUNLinSol_Dense(y, jac_mat, ctx);
        IDASetLinearSolver(ida_mem, lin_sol, jac_mat);
        IDASetJacFn(ida_mem, rust_jac_cb);

        let ic_status = IDACalcIC(ida_mem, 1, t_eval[1]);
        if ic_status < 0 {
            SUNLinSolFree(lin_sol);
            SUNMatDestroy(jac_mat);
            N_VDestroy(y);
            N_VDestroy(ydot);
            N_VDestroy(id);
            IDAFree(&mut (ida_mem as *mut c_void));
            SUNContext_Free(&mut ctx);
            panic!("IDACalcIC failed with code {}. Initial conditions may be inconsistent.", ic_status);
        }

        for i in 0..(n_states as usize) {
            out_traj[i] = *y_ptr.add(i);
        }

        let mut tret = 0.0;
        for step in 1..n_steps {
            let status = IDASolve(ida_mem, t_eval[step], &mut tret, y, ydot, 1);
            if status < 0 {
                SUNLinSolFree(lin_sol);
                SUNMatDestroy(jac_mat);
                N_VDestroy(y);
                N_VDestroy(ydot);
                N_VDestroy(id);
                IDAFree(&mut (ida_mem as *mut c_void));
                SUNContext_Free(&mut ctx);
                panic!("SUNDIALS Integration failed at t={} with code {}", tret, status);
            }
            
            for i in 0..(n_states as usize) {
                out_traj[step * (n_states as usize) + i] = *y_ptr.add(i);
            }
        }

        SUNLinSolFree(lin_sol);
        SUNMatDestroy(jac_mat);
        N_VDestroy(y);
        N_VDestroy(ydot);
        N_VDestroy(id);
        IDAFree(&mut (ida_mem as *mut c_void));
        SUNContext_Free(&mut ctx);
    }

    let ndarray = numpy::ndarray::Array2::from_shape_vec((n_steps, n_states as usize), out_traj).unwrap();
    Ok(ndarray.to_pyarray_bound(py))
}
