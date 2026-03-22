pub mod linalg;
pub mod integrator;
pub mod session;
pub mod adjoint;
pub mod bindings;

use std::os::raw::c_double;

pub type NativeResFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *const c_double, *mut c_double);
pub type NativeJacFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *const c_double, c_double, *mut c_double);
pub type NativeJvpFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *const c_double, c_double, *const c_double, *mut c_double);

pub type NativeVjpFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *const c_double, *const c_double, *mut c_double, *mut c_double, *mut c_double);