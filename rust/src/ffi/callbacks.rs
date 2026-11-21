use std::os::raw::{c_double, c_int};

pub type NativeResFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *const c_double, *mut c_double);
pub type NativeObsFn = unsafe extern "C" fn(*const c_double, *const c_double, *const c_double, *const c_double, *mut c_double);
pub type NativeJvpFn = unsafe extern "C" fn(
    *const c_double, *const c_double, *const c_double, *const c_double,
    c_double, *const c_double,
    *mut c_double, *mut c_double, *mut c_double, *mut c_double
);
pub type NativeVjpFn = unsafe extern "C" fn(
    *const c_double, *const c_double, *const c_double, *const c_double,
    *const c_double, *mut c_double,
    *mut c_double, *mut c_double, *mut c_double
);
pub type NativeIcFn = unsafe extern "C" fn(*const c_double, *const c_double, *mut c_double, *mut c_double);
pub type NativeSetThreadsFn = unsafe extern "C" fn(c_int);

#[derive(Clone, Copy)]
pub struct Callbacks {
    pub res_fn: NativeResFn,
    pub obs_fn: Option<NativeObsFn>,
    pub jvp_fn: Option<NativeJvpFn>,
    pub vjp_fn: Option<NativeVjpFn>,
    pub set_threads_fn: Option<NativeSetThreadsFn>,
}

impl Callbacks {
    pub fn from_lib(lib: &libloading::Library) -> Result<Self, String> {
        unsafe {
            Ok(Self {
                res_fn: *lib.get(b"evaluate_residual\0").map_err(|e| format!("Missing evaluate_residual: {e}"))?,
                obs_fn: lib.get(b"evaluate_observables\0").map(|s| *s).ok(),
                jvp_fn: lib.get(b"evaluate_jvp\0").map(|s| *s).ok(),
                vjp_fn: lib.get(b"evaluate_vjp\0").map(|s| *s).ok(),
                set_threads_fn: lib.get(b"set_spatial_threads\0").map(|s| *s).ok(),
            })
        }
    }

    pub fn load_ic(lib: &libloading::Library) -> Option<NativeIcFn> {
        unsafe { lib.get(b"evaluate_ic\0").map(|s| *s).ok() }
    }
}
