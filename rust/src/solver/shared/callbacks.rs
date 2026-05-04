use std::os::raw::{c_double, c_int};

pub type NativeResFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *const c_double, *mut c_double);
pub type NativeObsFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *const c_double, *mut c_double);
pub type NativeJvpFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *const c_double, c_double, *const c_double, *mut c_double);
pub type NativeVjpFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *const c_double, *const c_double, *mut c_double, *mut c_double, *mut c_double);
pub type NativeSetThreadsFn = unsafe extern "C" fn(c_int);