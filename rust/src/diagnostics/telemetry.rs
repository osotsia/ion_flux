use crate::diagnostics::crash_report::{Offender, CrashReport};

pub fn get_peak_rss_bytes() -> u64 {
    unsafe {
        let mut usage = std::mem::zeroed::<libc::rusage>();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0 {
            #[cfg(target_os = "macos")]
            {
                usage.ru_maxrss as u64
            }
            #[cfg(not(target_os = "macos"))]
            {
                (usage.ru_maxrss as u64) * 1024
            }
        } else {
            0
        }
    }
}

pub struct PmuGuard {
    #[cfg(target_os = "linux")]
    fd_cycles: libc::c_int,
    #[cfg(target_os = "linux")]
    fd_instr: libc::c_int,
}

impl PmuGuard {
    pub fn new() -> Self {
        #[cfg(target_os = "linux")]
        {
            let open_counter = |config: u64| -> libc::c_int {
                #[repr(C)]
                struct PerfAttr {
                    r#type: u32,
                    size: u32,
                    config: u64,
                    sample_period: u64,
                    sample_type: u64,
                    read_format: u64,
                    flags: u64,
                    wakeup_events: u32,
                    bp_type: u32,
                    config1: u64,
                    config2: u64,
                    branch_sample_type: u64,
                    sample_regs_user: u64,
                    sample_stack_user: u32,
                    clockid: i32,
                    sample_user_regs: u64,
                    aux_watermark: u32,
                    sample_max_stack: u16,
                    reserved_size: u16,
                }
                let mut attr: PerfAttr = unsafe { std::mem::zeroed() };
                attr.size = std::mem::size_of::<PerfAttr>() as u32;
                attr.r#type = 0; // PERF_TYPE_HARDWARE
                attr.config = config;
                attr.flags = 1 | 2; // disabled = 1, exclude_kernel = 1

                unsafe {
                    libc::syscall(
                        libc::SYS_perf_event_open,
                        &attr as *const _,
                        0 as libc::pid_t,
                        -1 as libc::c_int,
                        -1 as libc::c_int,
                        0 as libc::c_ulong,
                    ) as libc::c_int
                }
            };

            let fd_cycles = open_counter(0); // PERF_COUNT_HW_CPU_CYCLES
            let fd_instr = if fd_cycles >= 0 { open_counter(1) } else { -1 }; // PERF_COUNT_HW_INSTRUCTIONS

            if fd_cycles >= 0 && fd_instr >= 0 {
                unsafe {
                    libc::ioctl(fd_cycles, 0x2400); // PERF_EVENT_IOC_ENABLE
                    libc::ioctl(fd_instr, 0x2400);
                }
                Self { fd_cycles, fd_instr }
            } else {
                if fd_cycles >= 0 { unsafe { libc::close(fd_cycles); } }
                Self { fd_cycles: -1, fd_instr: -1 }
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            Self {}
        }
    }

    pub fn finish(self) -> (u64, u64) {
        #[cfg(target_os = "linux")]
        {
            if self.fd_cycles >= 0 && self.fd_instr >= 0 {
                let mut cycles: u64 = 0;
                let mut instr: u64 = 0;
                unsafe {
                    libc::ioctl(self.fd_cycles, 0x2401); // PERF_EVENT_IOC_DISABLE
                    libc::ioctl(self.fd_instr, 0x2401);
                    libc::read(self.fd_cycles, &mut cycles as *mut _ as *mut _, 8);
                    libc::read(self.fd_instr, &mut instr as *mut _ as *mut _, 8);
                    libc::close(self.fd_cycles);
                    libc::close(self.fd_instr);
                }
                return (cycles, instr);
            }
        }
        (0, 0)
    }
}

#[derive(Clone)]
pub struct Diagnostics {
    pub total_steps: usize,
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub newton_iterations: usize,
    pub jacobian_evaluations: usize,
    pub numeric_factorizations: usize,
    pub symbolic_factorizations: usize,
    
    pub jacobian_assembly_time_us: u128,
    pub linear_solve_time_us: u128,
    pub residual_time_us: u128,

    pub peak_rss_bytes: u64,
    pub cpu_cycles: u64,
    pub instructions: u64,
    
    pub trace_t: Vec<f64>,
    pub trace_dt: Vec<f64>,
    pub trace_order: Vec<usize>,
    pub trace_iters: Vec<usize>,
    pub trace_err: Vec<f64>,
    
    pub last_res: Vec<f64>,
    pub last_dy: Vec<f64>,
    pub last_weights: Vec<f64>,
    pub last_rho: f64,
    
    pub jac_max: f64,
    pub jac_min: f64,
    pub t0_max_res: f64,
    pub t0_max_res_idx: usize,
    pub recent_newton_norms: std::collections::VecDeque<(usize, f64, f64)>,
}

impl Diagnostics {
    pub fn new(n: usize) -> Self {
        Self {
            total_steps: 0, accepted_steps: 0, rejected_steps: 0, newton_iterations: 0,
            jacobian_evaluations: 0, numeric_factorizations: 0, symbolic_factorizations: 0,
            jacobian_assembly_time_us: 0, linear_solve_time_us: 0, residual_time_us: 0,
            peak_rss_bytes: 0, cpu_cycles: 0, instructions: 0,
            trace_t: Vec::with_capacity(1000), trace_dt: Vec::with_capacity(1000),
            trace_order: Vec::with_capacity(1000), trace_iters: Vec::with_capacity(1000),
            trace_err: Vec::with_capacity(1000),
            last_res: vec![0.0; n], last_dy: vec![0.0; n], last_weights: vec![0.0; n], last_rho: 0.0,
            jac_max: 0.0, jac_min: 0.0, t0_max_res: 0.0, t0_max_res_idx: 0,
            recent_newton_norms: std::collections::VecDeque::with_capacity(10),
        }
    }

    pub fn record_pmu(&mut self, cycles: u64, instructions: u64) {
        self.cpu_cycles = cycles;
        self.instructions = instructions;
        self.peak_rss_bytes = get_peak_rss_bytes();
    }

    pub fn build_crash_report(&self, y: &[f64], ydot: &[f64], id: &[f64], reason: String) -> CrashReport {
        let mut raw_offenders: Vec<(usize, f64, f64, f64, f64, f64, f64, bool)> = self.last_res.iter().enumerate()
            .map(|(i, &res)| {
                let weight = self.last_weights.get(i).copied().unwrap_or(0.0);
                let dy = self.last_dy.get(i).copied().unwrap_or(0.0);
                let err = dy * weight;
                let is_diff = id.get(i).unwrap_or(&0.0) > &0.5;
                let y_v = y.get(i).copied().unwrap_or(0.0);
                let ydot_v = ydot.get(i).copied().unwrap_or(0.0);
                (i, res, err.abs(), y_v, ydot_v, dy, weight, is_diff)
            }).collect();
        
        raw_offenders.sort_by(|a, b| {
            let a_nan = !a.1.is_finite() || !a.3.is_finite() || !a.4.is_finite();
            let b_nan = !b.1.is_finite() || !b.3.is_finite() || !b.4.is_finite();
            if a_nan && !b_nan { return std::cmp::Ordering::Less; }
            if !a_nan && b_nan { return std::cmp::Ordering::Greater; }
            b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        let offenders: Vec<Offender> = raw_offenders.into_iter().take(15).map(|(i, res, _err, y_v, ydot_v, dy, weight, is_diff)| {
            Offender {
                index: i, is_diff, y_val: y_v, ydot_val: ydot_v,
                residual: res, proposed_step_dy: dy, solver_weight: weight, weighted_error: _err,
            }
        }).collect();

        let trace = self.recent_newton_norms.iter().map(|&(iter, fnorm, dynorm)| (iter, fnorm, dynorm)).collect();
        let cond_warning = self.jac_max > 0.0 && self.jac_min > 0.0 && (self.jac_max / self.jac_min) > 1e12;

        CrashReport {
            reason,
            accepted_steps: self.accepted_steps,
            t0_max_res: self.t0_max_res,
            t0_max_res_idx: self.t0_max_res_idx,
            jac_max: self.jac_max,
            jac_min: self.jac_min,
            cond_warning,
            trace,
            offenders,
        }
    }
}
