#[derive(Clone)]
pub struct BdfHistory {
    pub order: usize,
    pub max_order: usize,
    pub k_used: usize,
    pub phase: usize, 
    pub ns: usize,    
    pub phi: Vec<Vec<f64>>, 
    pub psi: [f64; 6],
    pub alpha: [f64; 6],
    pub beta: [f64; 6],
    pub sigma: [f64; 6],
    pub gamma: [f64; 6],
    pub c_j: f64,
    pub c_j_old: f64,
    pub h_used: f64,
    pub h: f64,
}

impl BdfHistory {
    pub fn new(n: usize) -> Self {
        Self {
            order: 1, max_order: 5, k_used: 0, phase: 0, ns: 0,
            phi: vec![vec![0.0; n]; 6], psi: [0.0; 6], alpha: [0.0; 6], beta: [0.0; 6], sigma: [0.0; 6], gamma: [0.0; 6],
            c_j: 0.0, c_j_old: 0.0, h_used: 0.0, h: 0.0,
        }
    }

    pub fn set_coeffs(&mut self) -> f64 {
        if self.h != self.h_used || self.order != self.k_used { self.ns = 0; }
        self.ns = std::cmp::min(self.ns + 1, self.k_used + 2);
        
        if self.order + 1 >= self.ns {
            self.beta[0] = 1.0; self.alpha[0] = 1.0; self.gamma[0] = 0.0; self.sigma[0] = 1.0;
            let mut temp1 = self.h;
            for i in 1..=self.order {
                let temp2 = self.psi[i-1];
                self.psi[i-1] = temp1;
                self.beta[i] = self.beta[i-1] * self.psi[i-1] / temp2;
                temp1 = temp2 + self.h;
                self.alpha[i] = self.h / temp1;
                self.sigma[i] = (i as f64) * self.sigma[i-1] * self.alpha[i];
                self.gamma[i] = self.gamma[i-1] + self.alpha[i-1] / self.h;
            }
            self.psi[self.order] = temp1;
        }

        let (mut alphas, mut alpha0) = (0.0, 0.0);
        for i in 0..self.order { alphas -= 1.0 / ((i + 1) as f64); alpha0 -= self.alpha[i]; }
        self.c_j = -alphas / self.h;
        
        let ck = (self.alpha[self.order] + alphas - alpha0).abs().max(self.alpha[self.order]);
        if self.ns <= self.order {
            for i in self.ns..=self.order {
                let scale = self.beta[i];
                for j in 0..self.phi[i].len() { self.phi[i][j] *= scale; }
            }
        }
        ck
    }

    pub fn predict(&self, y_pred: &mut [f64], ydot_pred: &mut [f64]) {
        y_pred.fill(0.0); ydot_pred.fill(0.0);
        for j in 0..=self.order { for i in 0..y_pred.len() { y_pred[i] += self.phi[j][i]; } }
        for j in 0..self.order {
            let g = self.gamma[j+1];
            for i in 0..ydot_pred.len() { ydot_pred[i] += g * self.phi[j+1][i]; }
        }
    }

    pub fn restore(&mut self) {
        for i in 1..=self.order { self.psi[i-1] = self.psi[i] - self.h; }
        if self.ns <= self.order {
            for i in self.ns..=self.order {
                let inv_beta = 1.0 / self.beta[i];
                for j in 0..self.phi[i].len() { self.phi[i][j] *= inv_beta; }
            }
        }
    }

    pub fn complete_step(&mut self, err_k: f64, err_km1: f64, err_km2: f64, err_kp1: f64, ee: &[f64], min_dt: f64, max_dt: f64) {
        let n = ee.len();
        let kdiff = self.order as isize - self.k_used as isize;
        self.k_used = self.order; self.h_used = self.h;
        if (self.order == self.order - 1) || self.order == self.max_order { self.phase = 1; }

        if self.phase == 0 {
            if self.order < self.max_order { self.order += 1; }
            self.h = (self.h * 2.0).clamp(min_dt, max_dt); 
        } else {
            let action; let mut err_knew = err_k;
            let terr_k = (self.order as f64 + 1.0) * err_k;
            let terr_kp1 = (self.order as f64 + 2.0) * err_kp1;

            if self.order + 1 >= self.ns || kdiff == 1 || self.order == self.max_order { action = 0; } 
            else {
                let terr_km1 = (self.order as f64) * err_km1;
                let terr_km2 = (self.order as f64 - 1.0) * err_km2;

                if self.order == 1 {
                    action = if terr_kp1 >= 0.5 * terr_k { 0 } else { 1 };
                } else {
                    if terr_km1.max(terr_km2) <= terr_k { action = -1; }
                    else if terr_kp1 >= terr_k { action = 0; }
                    else { action = 1; }
                }
            }

            if action == 1 { self.order += 1; err_knew = err_kp1; }
            else if action == -1 { self.order -= 1; err_knew = err_km1; }

            let tmp = (2.0 * err_knew + 0.0001).powf(-1.0 / (self.order as f64 + 1.0));
            let mut eta = 1.0;
            if tmp >= 2.0 { eta = tmp.min(2.0); } else if tmp <= 1.0 { eta = tmp.min(0.9).max(0.5); }
            self.h = (self.h * eta).clamp(min_dt, max_dt);
        }

        if self.k_used < self.max_order { self.phi[self.k_used + 1].copy_from_slice(ee); }
        for i in 0..n { self.phi[self.k_used][i] += ee[i]; }
        for j in (0..self.k_used).rev() { for i in 0..n { self.phi[j][i] += self.phi[j + 1][i]; } }
    }

    pub fn restore_state(&mut self) {
        self.order = 1; self.k_used = 0; self.ns = 0;
        self.h = 0.0; self.h_used = 0.0; self.c_j = 0.0; self.c_j_old = 0.0;
        for j in 0..6 { self.phi[j].fill(0.0); }
    }
}