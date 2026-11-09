pub mod alloc_tracker;
pub mod mock_c_abi;

#[global_allocator]
static GLOBAL_ALLOCATOR: alloc_tracker::TrackingAllocator = alloc_tracker::TrackingAllocator;

use alloc_tracker::track_allocations;
use mock_c_abi::*;
use crate::solver::shared::diagnostics::Diagnostics;
use crate::solver::_4_linear::jacobian::assemble;
use crate::solver::_3_nonlinear::newton;
use crate::solver::_2_stepper::bdf;

#[test]
fn test_mock_c_abi_jvp_and_vjp_duality() {
    let p = [0.75, 2.0];
    let y = [1.2, 0.8, 1.5, 0.9, 1.1, 7.5];
    let ydot = [0.1, -0.2, 0.05, -0.1, 0.3, 0.0];
    let m = [0.0; 10];
    let c_j = 3.5;

    let v = [0.5, -0.3, 0.8, -0.2, 0.4, 1.2];
    let lambda = [0.2, 0.4, -0.1, 0.6, -0.3, 0.5];

    let mut jvp_out = [0.0; MOCK_N];
    unsafe {
        mock_evaluate_jvp(
            y.as_ptr(),
            ydot.as_ptr(),
            p.as_ptr(),
            m.as_ptr(),
            c_j,
            v.as_ptr(),
            jvp_out.as_mut_ptr(),
        );
    }

    let mut dp_out = [0.0; MOCK_N_PARAMS];
    let mut dy_out = [0.0; MOCK_N];
    let mut dydot_out = [0.0; MOCK_N];
    unsafe {
        mock_evaluate_vjp(
            y.as_ptr(),
            ydot.as_ptr(),
            p.as_ptr(),
            m.as_ptr(),
            lambda.as_ptr(),
            dp_out.as_mut_ptr(),
            dy_out.as_mut_ptr(),
            dydot_out.as_mut_ptr(),
        );
    }

    // Adjoint duality check: <lambda, J(c_j) * v> == <(dF/dy)^T lambda + c_j * (dF/dydot)^T lambda, v>
    let lhs: f64 = lambda.iter().zip(jvp_out.iter()).map(|(l, j)| l * j).sum();
    let rhs: f64 = v
        .iter()
        .zip(dy_out.iter().zip(dydot_out.iter()))
        .map(|(v_i, (dy_i, dydot_i))| v_i * (dy_i + c_j * dydot_i))
        .sum();

    let diff = (lhs - rhs).abs();
    assert!(
        diff < 1e-13,
        "JVP and VJP duality mismatch: lhs={lhs}, rhs={rhs}, diff={diff}"
    );

    assert!((dp_out[0] - lambda[0] * y[0]).abs() < 1e-14);
    assert!((dp_out[1] - (-lambda[5])).abs() < 1e-14);
}

#[test]
fn test_mock_c_abi_finite_difference_jvp() {
    let p = [0.5, 1.0];
    let y = [1.0, 1.2, 0.9, 1.1, 0.8, 6.0];
    let ydot = [0.0; MOCK_N];
    let m = [0.0; 10];
    let c_j = 2.0;
    let eps = 1e-7;

    let v = [1.0, 0.5, -0.5, 0.2, -0.1, 1.1];

    let mut jvp_analytical = [0.0; MOCK_N];
    unsafe {
        mock_evaluate_jvp(
            y.as_ptr(),
            ydot.as_ptr(),
            p.as_ptr(),
            m.as_ptr(),
            c_j,
            v.as_ptr(),
            jvp_analytical.as_mut_ptr(),
        );
    }

    let mut y_fwd = y;
    let mut ydot_fwd = ydot;
    for i in 0..MOCK_N {
        y_fwd[i] += eps * v[i];
        ydot_fwd[i] += c_j * eps * v[i];
    }

    let mut res0 = [0.0; MOCK_N];
    let mut res_fwd = [0.0; MOCK_N];
    unsafe {
        mock_evaluate_residual(y.as_ptr(), ydot.as_ptr(), p.as_ptr(), m.as_ptr(), res0.as_mut_ptr());
        mock_evaluate_residual(
            y_fwd.as_ptr(),
            ydot_fwd.as_ptr(),
            p.as_ptr(),
            m.as_ptr(),
            res_fwd.as_mut_ptr(),
        );
    }

    for i in 0..MOCK_N {
        let fd = (res_fwd[i] - res0[i]) / eps;
        let ana = jvp_analytical[i];
        assert!(
            (fd - ana).abs() < 1e-5,
            "Finite difference mismatch at index {i}: fd={fd}, analytical={ana}"
        );
    }
}

#[test]
fn test_alloc_tracker_detects_allocations() {
    let (_, stats_alloc) = track_allocations(|| {
        let v: Vec<f64> = (0..1000).map(|x| x as f64).collect();
        assert_eq!(v.len(), 1000);
    });
    assert!(
        stats_alloc.alloc_count >= 1,
        "Expected at least 1 allocation, got {}",
        stats_alloc.alloc_count
    );
    assert!(
        stats_alloc.bytes_allocated >= 8000,
        "Expected >= 8000 bytes, got {}",
        stats_alloc.bytes_allocated
    );

    let (val, stats_zero) = track_allocations(|| {
        let mut x = 1.0_f64;
        for _ in 0..100 {
            x = (x + 2.0) * 0.5;
        }
        x
    });
    assert_eq!(stats_zero.alloc_count, 0, "Expected 0 allocations, got {}", stats_zero.alloc_count);
    assert!(val > 0.0);
}

#[test]
fn test_mock_cpr_jacobian_assembly_exactness() {
    let (prob, mut wk) = create_mock_fixture();
    let c_j = 2.5;

    assemble(&prob, &mut wk, c_j);

    let j_true = mock_analytical_jacobian(wk.p[0], c_j);
    let mut expected_nonzeros = 0;
    for r in 0..MOCK_N {
        for c in 0..MOCK_N {
            if j_true[r][c] != 0.0 {
                expected_nonzeros += 1;
            }
        }
    }
    assert_eq!(wk.lu_solver.triplets.len(), expected_nonzeros);

    for &(r, c, val) in &wk.lu_solver.triplets {
        let expected = j_true[r][c];
        assert!(
            (val - expected).abs() < 1e-14,
            "Triplet mismatch at ({r}, {c}): assembled={val}, expected={expected}"
        );
    }
}

#[test]
fn test_mock_sparse_lu_solve_exactness() {
    let (prob, mut wk) = create_mock_fixture();
    let c_j = 1.5;

    assemble(&prob, &mut wk, c_j);

    let mut diag = Diagnostics::new(MOCK_N);
    wk.lu_solver.factorize_from_triplets(&mut diag).expect("Factorization failed");

    let b_orig = vec![1.0, 2.0, -1.0, 3.0, 0.5, 4.0];
    let mut b = b_orig.clone();
    wk.lu_solver.solve(&mut b, &mut diag).expect("Linear solve failed");

    let j_true = mock_analytical_jacobian(wk.p[0], c_j);
    let mut j_times_x = vec![0.0; MOCK_N];
    for r in 0..MOCK_N {
        for c in 0..MOCK_N {
            j_times_x[r] += j_true[r][c] * b[c];
        }
    }

    for i in 0..MOCK_N {
        assert!(
            (j_times_x[i] - b_orig[i]).abs() < 1e-11,
            "Linear residual too large at row {i}: got {}, expected {}",
            j_times_x[i],
            b_orig[i]
        );
    }
}

#[test]
fn test_mock_algebraic_root_convergence() {
    let (prob, mut wk) = create_mock_fixture();
    wk.y[5] = 0.0;

    let res = newton::calc_algebraic_roots(&prob, &mut wk);
    assert!(res.is_ok(), "calc_algebraic_roots failed: {:?}", res);

    let p1 = wk.p[1];
    let sum_0_to_4: f64 = wk.y[0..5].iter().sum();
    let expected_y5 = sum_0_to_4 + p1;

    assert!(
        (wk.y[5] - expected_y5).abs() < 1e-8,
        "Algebraic state did not converge: y[5]={}, expected={expected_y5}",
        wk.y[5]
    );
}

#[test]
fn test_mock_bdf_single_step() {
    let (prob, mut wk) = create_mock_fixture();

    let res = bdf::step(&prob, &mut wk, 0.01, None);
    assert!(res.is_ok(), "bdf::step failed on mock problem");
    assert!((wk.t - 0.01).abs() < 1e-12, "Time did not advance to 0.01: t={}", wk.t);

    for (i, val) in wk.y.iter().enumerate() {
        assert!(val.is_finite(), "State y[{i}] is not finite: {val}");
    }
}

#[test]
fn test_sundials_context_isolation() {
    unsafe {
        let ctx = crate::vendor::sundials::sys::get_safe_sundials_context();
        assert!(!ctx.is_null(), "Failed to allocate isolated SUNDIALS context");
        let mut ctx_ptr = ctx;
        let free_res = crate::vendor::sundials::sys::SUNContext_Free(&mut ctx_ptr);
        assert_eq!(free_res, 0, "SUNContext_Free failed");
    }
}
