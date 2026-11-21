use std::time::Instant;
use crate::ffi::callbacks::NativeJvpFn;
use crate::diagnostics::Diagnostics;

pub struct GmresSolver {
    pub n: usize,
    pub m: usize,
    pub triplets: Vec<(usize, usize, f64)>,
    v: Vec<Vec<f64>>,
    h: Vec<Vec<f64>>,
    cs: Vec<f64>,
    sn: Vec<f64>,
    g: Vec<f64>,
    b_pre: Vec<f64>,
    temp_jvp: Vec<f64>,
    jvp_dy: Vec<f64>,
    jvp_dydot: Vec<f64>,
    res_scratch: Vec<f64>,
    y_sol: Vec<f64>,

    y_ptr: *const f64,
    ydot_ptr: *const f64,
    p_ptr: *const f64,
    m_ptr: *const f64,
    c_j: f64,
    id_ptr: *const f64,
    spatial_diag_ptr: *const f64,
    jvp_fn: Option<NativeJvpFn>,
}

unsafe impl Send for GmresSolver {}
unsafe impl Sync for GmresSolver {}

impl GmresSolver {
    pub fn new(n: usize) -> Self {
        let m = std::cmp::min(n, 30);
        Self {
            n,
            m,
            triplets: Vec::new(),
            v: vec![vec![0.0; n]; m + 1],
            h: vec![vec![0.0; m]; m + 1],
            cs: vec![0.0; m],
            sn: vec![0.0; m],
            g: vec![0.0; m + 1],
            b_pre: vec![0.0; n],
            temp_jvp: vec![0.0; n],
            jvp_dy: vec![0.0; n],
            jvp_dydot: vec![0.0; n],
            res_scratch: vec![0.0; n],
            y_sol: vec![0.0; m],
            y_ptr: std::ptr::null(),
            ydot_ptr: std::ptr::null(),
            p_ptr: std::ptr::null(),
            m_ptr: std::ptr::null(),
            c_j: 0.0,
            id_ptr: std::ptr::null(),
            spatial_diag_ptr: std::ptr::null(),
            jvp_fn: None,
        }
    }

    pub fn set_context(
        &mut self,
        y: *const f64,
        ydot: *const f64,
        p: *const f64,
        m: *const f64,
        c_j: f64,
        id: *const f64,
        spatial_diag: *const f64,
        jvp_fn: Option<NativeJvpFn>,
    ) {
        self.y_ptr = y;
        self.ydot_ptr = ydot;
        self.p_ptr = p;
        self.m_ptr = m;
        self.c_j = c_j;
        self.id_ptr = id;
        self.spatial_diag_ptr = spatial_diag;
        self.jvp_fn = jvp_fn;
    }

    pub fn solve(&mut self, b: &mut [f64], diag: &mut Diagnostics) -> Result<(), String> {
        let start_time = Instant::now();
        let n = self.n;
        let m = self.m;
        let jvp_fn = self.jvp_fn.ok_or_else(|| "evaluate_jvp function pointer missing for GMRES".to_string())?;

        let y_ptr = self.y_ptr;
        let ydot_ptr = self.ydot_ptr;
        let p_ptr = self.p_ptr;
        let m_ptr = self.m_ptr;
        let c_j = self.c_j;
        let id_ptr = self.id_ptr;
        let sdiag_ptr = self.spatial_diag_ptr;
        let jvp_dy = &mut self.jvp_dy;
        let jvp_dydot = &mut self.jvp_dydot;
        let res_scratch = &mut self.res_scratch;

        let jvp = |v_k: &[f64], out: &mut [f64]| unsafe {
            jvp_fn(
                y_ptr,
                ydot_ptr,
                p_ptr,
                m_ptr,
                c_j,
                v_k.as_ptr(),
                jvp_dy.as_mut_ptr(),
                jvp_dydot.as_mut_ptr(),
                res_scratch.as_mut_ptr(),
                out.as_mut_ptr(),
            );
        };
        let precond = |src: &[f64], dst: &mut [f64]| {
            for i in 0..n {
                let id_val = if !id_ptr.is_null() { unsafe { *id_ptr.add(i) } } else { 0.0 };
                let sdiag_val = if !sdiag_ptr.is_null() { unsafe { *sdiag_ptr.add(i) } } else { 0.0 };
                dst[i] = src[i] / (c_j * id_val + sdiag_val + 1.0);
            }
        };

        run_gmres(
            n, m, b,
            &mut self.v, &mut self.h, &mut self.cs, &mut self.sn, &mut self.g,
            &mut self.b_pre, &mut self.temp_jvp, &mut self.y_sol,
            jvp, precond,
        )?;

        diag.linear_solve_time_us += start_time.elapsed().as_micros();
        Ok(())
    }
}

/// Shared Arnoldi/Givens left-preconditioned GMRES kernel used by both forward and adjoint solvers.
#[allow(clippy::too_many_arguments)]
fn run_gmres<F, P>(
    n: usize,
    m: usize,
    b: &mut [f64],
    v: &mut [Vec<f64>],
    h: &mut [Vec<f64>],
    cs: &mut [f64],
    sn: &mut [f64],
    g: &mut [f64],
    b_pre: &mut [f64],
    temp_jvp: &mut [f64],
    y_sol: &mut [f64],
    mut jvp: F,
    mut precond: P,
) -> Result<(), String>
where
    F: FnMut(&[f64], &mut [f64]),
    P: FnMut(&[f64], &mut [f64]),
{
    precond(b, b_pre);
    let b_norm = b_pre.iter().map(|&x| x * x).sum::<f64>().sqrt();

    if b_norm <= 1e-30 {
        b.fill(0.0);
        return Ok(());
    }

    for i in 0..n {
        v[0][i] = b_pre[i] / b_norm;
    }
    g.fill(0.0);
    g[0] = b_norm;

    let mut k = 0;
    while k < m {
        {
            let (left, right) = v.split_at_mut(k + 1);
            let v_k = &left[k];
            let v_kp1 = &mut right[0];
            jvp(v_k, temp_jvp);
            precond(temp_jvp, v_kp1);

            for i in 0..=k {
                let v_i = &left[i];
                let mut dot = 0.0;
                for j in 0..n {
                    dot += v_i[j] * v_kp1[j];
                }
                h[i][k] = dot;
                for j in 0..n {
                    v_kp1[j] -= dot * v_i[j];
                }
            }

            let w_norm = v_kp1.iter().map(|&x| x * x).sum::<f64>().sqrt();
            h[k + 1][k] = w_norm;
            if w_norm > 1e-14 {
                for j in 0..n {
                    v_kp1[j] /= w_norm;
                }
            }
        }

        for i in 0..k {
            let temp = cs[i] * h[i][k] + sn[i] * h[i + 1][k];
            h[i + 1][k] = -sn[i] * h[i][k] + cs[i] * h[i + 1][k];
            h[i][k] = temp;
        }

        let beta = (h[k][k] * h[k][k] + h[k + 1][k] * h[k + 1][k]).sqrt();
        if beta > 1e-14 {
            cs[k] = h[k][k] / beta;
            sn[k] = h[k + 1][k] / beta;
        } else {
            cs[k] = 1.0;
            sn[k] = 0.0;
        }

        h[k][k] = cs[k] * h[k][k] + sn[k] * h[k + 1][k];
        h[k + 1][k] = 0.0;
        g[k + 1] = -sn[k] * g[k];
        g[k] = cs[k] * g[k];

        if g[k + 1].abs() < 1e-6 * b_norm {
            k += 1;
            break;
        }
        k += 1;
    }

    for i in (0..k).rev() {
        y_sol[i] = g[i];
        for j in (i + 1)..k {
            y_sol[i] -= h[i][j] * y_sol[j];
        }
        if h[i][i].abs() < 1e-14 {
            return Err("GMRES Singular H matrix".to_string());
        }
        y_sol[i] /= h[i][i];
    }

    b.fill(0.0);
    for j in 0..k {
        for i in 0..n {
            b[i] += v[j][i] * y_sol[j];
        }
    }
    Ok(())
}

pub fn solve_gmres<F, P>(n: usize, b: &mut [f64], jvp: F, precond: P) -> Result<(), String>
where
    F: FnMut(&[f64], &mut [f64]),
    P: FnMut(&[f64], &mut [f64]),
{
    let m = std::cmp::min(n, 30);
    let mut v = vec![vec![0.0; n]; m + 1];
    let mut h = vec![vec![0.0; m]; m + 1];
    let mut cs = vec![0.0; m];
    let mut sn = vec![0.0; m];
    let mut g = vec![0.0; m + 1];
    let mut b_pre = vec![0.0; n];
    let mut temp_jvp = vec![0.0; n];
    let mut y_sol = vec![0.0; m];

    run_gmres(
        n, m, b,
        &mut v, &mut h, &mut cs, &mut sn, &mut g,
        &mut b_pre, &mut temp_jvp, &mut y_sol,
        jvp, precond,
    )
}
