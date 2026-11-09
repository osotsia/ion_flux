#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::os::raw::{c_char, c_double, c_int, c_long, c_void};

pub type SunRealType = c_double;
pub type SunIndexType = c_long;

#[repr(C)]
pub struct _generic_SUNContext {
    _private: [u8; 0],
}
pub type SUNContext = *mut _generic_SUNContext;

#[repr(C)]
pub struct _generic_N_Vector {
    _private: [u8; 0],
}
pub type N_Vector = *mut _generic_N_Vector;

#[repr(C)]
pub struct _generic_SUNMatrix {
    _private: [u8; 0],
}
pub type SUNMatrix = *mut _generic_SUNMatrix;

#[repr(C)]
pub struct _generic_SUNLinearSolver {
    _private: [u8; 0],
}
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
    pub fn IDAInit(
        ida_mem: *mut c_void,
        res: Option<unsafe extern "C" fn(SunRealType, N_Vector, N_Vector, N_Vector, *mut c_void) -> c_int>,
        t0: SunRealType,
        yy0: N_Vector,
        yp0: N_Vector,
    ) -> c_int;
    pub fn IDAReInit(ida_mem: *mut c_void, t0: SunRealType, yy0: N_Vector, yp0: N_Vector) -> c_int;
    pub fn IDASStolerances(ida_mem: *mut c_void, reltol: SunRealType, abstol: SunRealType) -> c_int;
    pub fn IDASetSuppressAlg(ida_mem: *mut c_void, suppressalg: c_int) -> c_int;
    pub fn IDASetUserData(ida_mem: *mut c_void, user_data: *mut c_void) -> c_int;
    pub fn IDASetId(ida_mem: *mut c_void, id: N_Vector) -> c_int;
    pub fn IDASetLinearSolver(ida_mem: *mut c_void, LS: SUNLinearSolver, A: SUNMatrix) -> c_int;
    pub fn IDASetJacFn(
        ida_mem: *mut c_void,
        jac: Option<
            unsafe extern "C" fn(
                SunRealType,
                SunRealType,
                N_Vector,
                N_Vector,
                N_Vector,
                SUNMatrix,
                *mut c_void,
                N_Vector,
                N_Vector,
                N_Vector,
            ) -> c_int,
        >,
    ) -> c_int;
    pub fn IDACalcIC(ida_mem: *mut c_void, icopt: c_int, tout1: SunRealType) -> c_int;
    pub fn IDASolve(
        ida_mem: *mut c_void,
        tout: SunRealType,
        tret: *mut SunRealType,
        yret: N_Vector,
        ypret: N_Vector,
        itask: c_int,
    ) -> c_int;
    pub fn IDASetMaxNumSteps(ida_mem: *mut c_void, mxsteps: c_long) -> c_int;

    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
}

/// Safely constructs a SUNDIALS SUNContext, handling dynamic symbol lookups
/// and isolating MPI runtime dependencies.
pub unsafe fn get_safe_sundials_context() -> SUNContext {
    let mut sunctx: SUNContext = std::ptr::null_mut();
    let mut comm: *mut c_void = std::ptr::null_mut();

    #[cfg(unix)]
    {
        #[cfg(target_os = "macos")]
        let rtld_search = -2isize as *mut c_void;
        #[cfg(not(target_os = "macos"))]
        let rtld_search = std::ptr::null_mut();

        let get_sym = |name: &str| -> *mut c_void {
            let c_name = std::ffi::CString::new(name).unwrap();
            let mut ptr = dlsym(rtld_search, c_name.as_ptr());

            if ptr.is_null() {
                #[cfg(target_os = "macos")]
                let flags = 2 | 8;
                #[cfg(target_os = "linux")]
                let flags = 2 | 256;
                #[cfg(not(any(target_os = "macos", target_os = "linux")))]
                let flags = 2;

                let libs = [
                    "libsundials_core.dylib\0",
                    "libsundials_core.so\0",
                    "libsundials_core.7.6.0.dylib\0",
                ];
                for lib in &libs {
                    let h = dlopen(lib.as_ptr() as *const c_char, flags);
                    if !h.is_null() {
                        ptr = dlsym(h, c_name.as_ptr());
                        if !ptr.is_null() {
                            break;
                        }
                    }
                }
            }
            ptr
        };

        let mpi_init_ptr = get_sym("MPI_Init");
        if !mpi_init_ptr.is_null() {
            let mpi_init: unsafe extern "C" fn(*mut c_int, *mut *mut *mut i8) -> c_int =
                std::mem::transmute(mpi_init_ptr);
            let mpi_init_check_ptr = get_sym("MPI_Initialized");

            if !mpi_init_check_ptr.is_null() {
                let mpi_init_check: unsafe extern "C" fn(*mut c_int) -> c_int =
                    std::mem::transmute(mpi_init_check_ptr);
                let mut flag = 0;
                mpi_init_check(&mut flag);
                if flag == 0 {
                    mpi_init(std::ptr::null_mut(), std::ptr::null_mut());
                }
            }

            let ompi_world = get_sym("ompi_mpi_comm_world");
            if !ompi_world.is_null() {
                comm = ompi_world;
            } else {
                let mpich_sym = get_sym("MPIR_Comm_direct");
                if !mpich_sym.is_null() {
                    comm = 0x44000000_usize as *mut c_void;
                }
            }
        }
    }
    SUNContext_Create(comm, &mut sunctx);
    sunctx
}
