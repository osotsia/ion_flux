use std::time::Instant;
use crate::ffi::callbacks::NativeJvpFn;
use crate::solver::shared::diagnostics::Diagnostics;

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
        let jvp = self.jvp_fn.ok_or_else(|| "evaluate_jvp function pointer missing for GMRES".to_string())?;

        let y_ptr = self.y_ptr;
        let ydot_ptr = self.ydot_ptr;
        let p_ptr = self.p_ptr;
        let m_ptr = self.m_ptr;
        let c_j = self.c_j;
        let id_ptr = self.id_ptr;
        let sdiag_ptr = self.spatial_diag_ptr;

        let mut b_norm = 0.0;
        for i in 0..n {
            let id_val = if !id_ptr.is_null() { unsafe { *id_ptr.add(i) } } else { 0.0 };
            let sdiag_val = if !sdiag_ptr.is_null() { unsafe { *sdiag_ptr.add(i) } } else { 0.0 };
            let p_fac = c_j * id_val + sdiag_val + 1.0;
            let val = b[i] / p_fac;
            self.b_pre[i] = val;
            b_norm += val * val;
        }
        b_norm = b_norm.sqrt();

        if b_norm < 1e-12 {
            for i in 0..n { b[i] = 0.0; }
            diag.linear_solve_time_us += start_time.elapsed().as_micros();
            return Ok(());
        }

        for i in 0..n { self.v[0][i] = self.b_pre[i] / b_norm; }
        self.g.fill(0.0);
        self.g[0] = b_norm;

        let mut k = 0;
        while k < m {
            let v_k = &self.v[k];
            unsafe {
                jvp(
                    y_ptr,
                    ydot_ptr,
                    p_ptr,
                    m_ptr,
                    c_j,
                    v_k.as_ptr(),
                    self.jvp_dy.as_mut_ptr(),
                    self.jvp_dydot.as_mut_ptr(),
                    self.res_scratch.as_mut_ptr(),
                    self.temp_jvp.as_mut_ptr(),
                );
            }

            for i in 0..n {
                let id_val = if !id_ptr.is_null() { unsafe { *id_ptr.add(i) } } else { 0.0 };
                let sdiag_val = if !sdiag_ptr.is_null() { unsafe { *sdiag_ptr.add(i) } } else { 0.0 };
                self.v[k + 1][i] = self.temp_jvp[i] / (c_j * id_val + sdiag_val + 1.0);
            }

            for i in 0..=k {
                let mut dot = 0.0;
                for j in 0..n { dot += self.v[i][j] * self.v[k + 1][j]; }
                self.h[i][k] = dot;
                for j in 0..n { self.v[k + 1][j] -= dot * self.v[i][j]; }
            }

            let mut w_norm = 0.0;
            for j in 0..n { w_norm += self.v[k + 1][j] * self.v[k + 1][j]; }
            w_norm = w_norm.sqrt();
            self.h[k + 1][k] = w_norm;
            if w_norm > 1e-14 {
                for j in 0..n { self.v[k + 1][j] /= w_norm; }
            }

            for i in 0..k {
                let temp = self.cs[i] * self.h[i][k] + self.sn[i] * self.h[i + 1][k];
                self.h[i + 1][k] = -self.sn[i] * self.h[i][k] + self.cs[i] * self.h[i + 1][k];
                self.h[i][k] = temp;
            }

            let beta = (self.h[k][k] * self.h[k][k] + self.h[k + 1][k] * self.h[k + 1][k]).sqrt();
            if beta > 1e-14 {
                self.cs[k] = self.h[k][k] / beta;
                self.sn[k] = self.h[k + 1][k] / beta;
            } else {
                self.cs[k] = 1.0;
                self.sn[k] = 0.0;
            }

            self.h[k][k] = self.cs[k] * self.h[k][k] + self.sn[k] * self.h[k + 1][k];
            self.h[k + 1][k] = 0.0;
            self.g[k + 1] = -self.sn[k] * self.g[k];
            self.g[k] = self.cs[k] * self.g[k];

            if self.g[k + 1].abs() < 1e-6 * b_norm {
                k += 1;
                break;
            }
            k += 1;
        }

        for i in (0..k).rev() {
            self.y_sol[i] = self.g[i];
            for j in (i + 1)..k {
                self.y_sol[i] -= self.h[i][j] * self.y_sol[j];
            }
            if self.h[i][i].abs() < 1e-14 {
                return Err("GMRES Singular H matrix".to_string());
            }
            self.y_sol[i] /= self.h[i][i];
        }

        for i in 0..n { b[i] = 0.0; }
        for j in 0..k {
            for i in 0..n {
                b[i] += self.v[j][i] * self.y_sol[j];
            }
        }

        diag.linear_solve_time_us += start_time.elapsed().as_micros();
        Ok(())
    }
}

pub fn solve_gmres<F, P>(n: usize, b: &mut [f64], mut jvp: F, mut precond: P) -> Result<(), String>
where F: FnMut(&[f64], &mut [f64]), P: FnMut(&[f64], &mut[f64]) {
    let m = std::cmp::min(n, 30);
    let mut v = vec![vec![0.0; n]; m + 1];
    let mut h = vec![vec![0.0; m]; m + 1];
    let mut cs = vec![0.0; m];
    let mut sn = vec![0.0; m];
    let mut g = vec![0.0; m + 1];
    let mut b_pre = vec![0.0; n];
    precond(b, &mut b_pre);

    let mut b_norm = 0.0;
    for i in 0..n { b_norm += b_pre[i] * b_pre[i]; }
    b_norm = b_norm.sqrt();
    if b_norm < 1e-12 {
        for i in 0..n { b[i] = 0.0; }
        return Ok(());
    }
    for i in 0..n { v[0][i] = b_pre[i] / b_norm; }
    g[0] = b_norm;

    let mut k = 0;
    let mut temp_jvp = vec![0.0; n];

    while k < m {
        let (left, right) = v.split_at_mut(k + 1);
        let v_k = &left[k];
        let v_kp1 = &mut right[0];
        
        jvp(v_k, &mut temp_jvp);
        precond(&temp_jvp, v_kp1);

        for i in 0..=k {
            let v_i = &left[i];
            let mut dot = 0.0;
            for j in 0..n { dot += v_i[j] * v_kp1[j]; }
            h[i][k] = dot;
            for j in 0..n { v_kp1[j] -= dot * v_i[j]; }
        }

        let mut w_norm = 0.0;
        for j in 0..n { w_norm += v_kp1[j] * v_kp1[j]; }
        w_norm = w_norm.sqrt();
        h[k + 1][k] = w_norm;
        if w_norm > 1e-14 { for j in 0..n { v_kp1[j] /= w_norm; } }

        for i in 0..k {
            let temp = cs[i] * h[i][k] + sn[i] * h[i + 1][k];
            h[i + 1][k] = -sn[i] * h[i][k] + cs[i] * h[i + 1][k];
            h[i][k] = temp;
        }

        let beta = (h[k][k] * h[k][k] + h[k + 1][k] * h[k + 1][k]).sqrt();
        if beta > 1e-14 {
            cs[k] = h[k][k] / beta; sn[k] = h[k + 1][k] / beta;
        } else { cs[k] = 1.0; sn[k] = 0.0; }

        h[k][k] = cs[k] * h[k][k] + sn[k] * h[k + 1][k];
        h[k + 1][k] = 0.0;
        g[k + 1] = -sn[k] * g[k];
        g[k] = cs[k] * g[k];

        if g[k + 1].abs() < 1e-6 * b_norm { k += 1; break; }
        k += 1;
    }

    let mut y = vec![0.0; k];
    for i in (0..k).rev() {
        y[i] = g[i];
        for j in (i + 1)..k { y[i] -= h[i][j] * y[j]; }
        if h[i][i].abs() < 1e-14 { return Err("GMRES Singular H matrix".to_string()); }
        y[i] /= h[i][i];
    }

    for i in 0..n { b[i] = 0.0; }
    for j in 0..k {
        for i in 0..n { b[i] += v[j][i] * y[j]; }
    }
    Ok(())
}
