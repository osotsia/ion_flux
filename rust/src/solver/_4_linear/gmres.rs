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