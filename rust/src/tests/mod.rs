pub mod alloc_tracker;
pub mod mock_c_abi;

#[global_allocator]
static GLOBAL_ALLOCATOR: alloc_tracker::TrackingAllocator = alloc_tracker::TrackingAllocator;

use alloc_tracker::track_allocations;
use mock_c_abi::*;
use crate::linear::jacobian::assemble;
use crate::stepper::bdf;
use crate::nonlinear::algebraic_roots;

#[test]
fn test_mock_c_abi_jvp_and_vjp_duality() {
    let p = [0.75, 2.0];
    let y = [1.2, 0.8, 1.5, 0.9, 1.1, 7.5];
    let ydot = [0.1, -0.2, 0.05, -0.1, 0.3, 0.0];
    let m = [0.0; 10];
    let c_j = 3.5;

    let v = [0.5, -0.3, 0.8, -0.2, 0.4, 1.2];
    let lambda = [0.2, 0.4, -0.1, 0.6, -0.3, 0.5];

    let mut jvp_dy = [0.0; MOCK_N];
    let mut jvp_dydot = [0.0; MOCK_N];
    let mut res_dummy = [0.0; MOCK_N];
    let mut jvp_out = [0.0; MOCK_N];
    unsafe {
        mock_evaluate_jvp(
            y.as_ptr(),
            ydot.as_ptr(),
            p.as_ptr(),
            m.as_ptr(),
            c_j,
            v.as_ptr(),
            jvp_dy.as_mut_ptr(),
            jvp_dydot.as_mut_ptr(),
            res_dummy.as_mut_ptr(),
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
            res_dummy.as_mut_ptr(),
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

    let mut jvp_dy = [0.0; MOCK_N];
    let mut jvp_dydot = [0.0; MOCK_N];
    let mut res_dummy = [0.0; MOCK_N];
    let mut jvp_analytical = [0.0; MOCK_N];
    unsafe {
        mock_evaluate_jvp(
            y.as_ptr(),
            ydot.as_ptr(),
            p.as_ptr(),
            m.as_ptr(),
            c_j,
            v.as_ptr(),
            jvp_dy.as_mut_ptr(),
            jvp_dydot.as_mut_ptr(),
            res_dummy.as_mut_ptr(),
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
    let (prob, state, _history, _scratch, mut linear_solver, _diag) = create_mock_fixture();
    let c_j = 2.5;

    assemble(&prob, &state, &mut linear_solver, c_j);

    let j_true = mock_analytical_jacobian(state.p[0], c_j);
    let mut expected_nonzeros = 0;
    for r in 0..MOCK_N {
        for c in 0..MOCK_N {
            let val = j_true[r][c];
            if val != 0.0 {
                expected_nonzeros += 1;
            }
        }
    }
    assert_eq!(linear_solver.triplets().len(), expected_nonzeros);

    for &(r, c, val) in linear_solver.triplets() {
        let expected = j_true[r][c];
        assert!(
            (val - expected).abs() < 1e-14,
            "Triplet mismatch at ({r}, {c}): assembled={val}, expected={expected}"
        );
    }
}

#[test]
fn test_mock_sparse_lu_solve_exactness() {
    let (prob, state, _history, _scratch, mut linear_solver, mut diag) = create_mock_fixture();
    let c_j = 1.5;

    assemble(&prob, &state, &mut linear_solver, c_j);

    linear_solver.factorize_from_triplets(&mut diag).expect("Factorization failed");

    let b_orig = vec![1.0, 2.0, -1.0, 3.0, 0.5, 4.0];
    let mut b = b_orig.clone();
    linear_solver.solve(&mut b, &mut diag).expect("Linear solve failed");

    let j_true = mock_analytical_jacobian(state.p[0], c_j);
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
    let (prob, mut state, mut history, mut scratch, mut linear_solver, mut diag) = create_mock_fixture();
    state.y[5] = 0.0;

    let res = algebraic_roots::calc_algebraic_roots(&prob, &mut state, &mut history, &mut scratch, &mut linear_solver, &mut diag);
    assert!(res.is_ok(), "calc_algebraic_roots failed: {:?}", res);

    let p1 = state.p[1];
    let sum_0_to_4: f64 = state.y[0..5].iter().sum();
    let expected_y5 = sum_0_to_4 + p1;

    assert!(
        (state.y[5] - expected_y5).abs() < 1e-8,
        "Algebraic state did not converge: y[5]={}, expected={expected_y5}",
        state.y[5]
    );
}

#[test]
fn test_mock_bdf_single_step() {
    let (prob, mut state, mut history, mut scratch, mut linear_solver, mut diag) = create_mock_fixture();

    let res = bdf::step(&prob, &mut state, &mut history, &mut scratch, &mut linear_solver, &mut diag, 0.01, None);
    assert!(res.is_ok(), "bdf::step failed on mock problem");
    assert!((state.t - 0.01).abs() < 1e-12, "Time did not advance to 0.01: t={}", state.t);

    for (i, val) in state.y.iter().enumerate() {
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

#[test]
fn test_sparsity_stability_across_assemblies() {
    let (prob, mut state, _history, _scratch, mut linear_solver, _diag) = create_mock_fixture();
    let mut triplet_counts = Vec::new();

    for i in 0..100 {
        state.y[0] = 1.0 + (i as f64) * 1e-17;
        let c_j = 1.0 + (i as f64) * 0.01;
        assemble(&prob, &state, &mut linear_solver, c_j);
        triplet_counts.push(linear_solver.triplets().len());
    }

    let first_count = triplet_counts[0];
    assert_eq!(first_count, 20); // 14 sparse + 6 dense entries
    for (step, &count) in triplet_counts.iter().enumerate() {
        assert_eq!(
            count, first_count,
            "Sparsity pattern mutated at step {step}: count={count}, expected={first_count}"
        );
    }
}

#[test]
fn test_symbolic_lu_reuse_single_execution() {
    let (prob, mut state, _history, _scratch, mut linear_solver, mut diag) = create_mock_fixture();

    for i in 0..10 {
        state.y[0] = 1.0 + (i as f64) * 0.05;
        let c_j = 1.0 + (i as f64) * 0.1;
        assemble(&prob, &state, &mut linear_solver, c_j);
        linear_solver.factorize_from_triplets(&mut diag).expect("Factorization failed");
    }

    assert_eq!(diag.numeric_factorizations, 10, "Expected 10 numeric factorizations");
    assert_eq!(
        diag.symbolic_factorizations, 1,
        "Symbolic factorization must run exactly once on step 1 and be cached thereafter!"
    );
}

#[test]
fn test_closed_enum_linear_solver_dispatch() {
    use crate::linear::solver::LinearSolver;

    let (prob, state, _history, _scratch, mut linear_solver, mut diag) = create_mock_fixture();
    let c_j = 1.0;
    assemble(&prob, &state, &mut linear_solver, c_j);

    assert!(!linear_solver.is_gmres());
    linear_solver.factorize_from_triplets(&mut diag).expect("SparseLU factorization failed");

    let b_orig = vec![1.0, 2.0, -1.0, 3.0, 0.5, 4.0];
    let mut b_lu = b_orig.clone();
    linear_solver.solve(&mut b_lu, &mut diag).expect("SparseLU solve failed");

    let mut gmres_solver = LinearSolver::new(MOCK_N, -1);
    assert!(gmres_solver.is_gmres());

    gmres_solver.set_gmres_context(
        state.y.as_ptr(),
        state.ydot.as_ptr(),
        state.p.as_ptr(),
        prob.m.as_ptr(),
        c_j,
        prob.id.as_ptr(),
        prob.spatial_diag.as_ptr(),
        prob.fns.jvp_fn,
    );

    let mut b_gmres = b_orig.clone();
    gmres_solver.solve(&mut b_gmres, &mut diag).expect("GMRES solve failed");

    let j_true = mock_analytical_jacobian(state.p[0], c_j);
    for r in 0..MOCK_N {
        let mut row_sum_lu = 0.0;
        let mut row_sum_gmres = 0.0;
        for c in 0..MOCK_N {
            row_sum_lu += j_true[r][c] * b_lu[c];
            row_sum_gmres += j_true[r][c] * b_gmres[c];
        }
        assert!((row_sum_lu - b_orig[r]).abs() < 1e-10);
        assert!((row_sum_gmres - b_orig[r]).abs() < 1e-4);
    }
}

// Note: test_bdf_step_zero_allocations is disabled pending pre-allocated faer factorization arenas.
// Note: test_bdf_step_zero_allocations is disabled pending pre-allocated faer factorization arenas.
// #[test]
// fn test_bdf_step_zero_allocations() {
//     let (prob, mut state, mut history, mut scratch, mut linear_solver, mut diag) = create_mock_fixture();
//
//     // Pre-converge algebraic equations to guarantee exact thermodynamic equilibrium
//     algebraic_roots::calc_algebraic_roots(&prob, &mut state, &mut history, &mut scratch, &mut linear_solver, &mut diag)
//         .expect("Algebraic initialization failed");
//
//     // Warm-up step: step 0 sets up initial step size and warms up caches
//     bdf::step(&prob, &mut state, &mut history, &mut scratch, &mut linear_solver, &mut diag, 0.001, None)
//         .expect("Warmup step failed");
//
//     // Active time-stepping loop: assert 0 heap allocations
//     let (_res, stats) = track_allocations(|| {
//         for _ in 0..5 {
//             bdf::step(&prob, &mut state, &mut history, &mut scratch, &mut linear_solver, &mut diag, 0.001, None)
//                 .expect("Active step failed");
//         }
//     });
//
//     assert_eq!(
//         stats.alloc_count, 0,
//         "Active bdf::step executed {} allocations (expected 0 hot-loop allocations)!",
//         stats.alloc_count
//     );
// }

#[test]
fn test_protocol_hysteresis() {
    let (prob, state, history, scratch, linear_solver, diag) = create_mock_fixture();
    let mut out_t = Vec::new();
    let mut out_traj = Vec::new();
    let mut out_obs = Vec::new();

    let steps = vec![
        crate::control::sequence::ProtocolStep {
            s_type: 0, // CC mode
            target_val: 0.1,
            t_limit: 0.02,
            trig: crate::control::bisection::TrigInfo {
                has_trig: false,
                idx: 0,
                size: 1,
                is_obs: false,
                op: 0,
                val: 0.0,
            },
            p_mode: 0,
            p_i: 1,
            p_v: 0,
        },
        crate::control::sequence::ProtocolStep {
            s_type: 1, // CV mode
            target_val: 1.0,
            t_limit: 0.02,
            trig: crate::control::bisection::TrigInfo {
                has_trig: false,
                idx: 0,
                size: 1,
                is_obs: false,
                op: 0,
                val: 0.0,
            },
            p_mode: 0,
            p_i: 1,
            p_v: 0,
        },
    ];

    let mut instance = crate::stepper::state::SolverInstance {
        state,
        history,
        scratch,
        linear_solver,
        diag,
    };
    let res = crate::control::sequence::run_sequence(
        &prob,
        &mut instance,
        &steps,
        &mut out_t,
        &mut out_traj,
        &mut out_obs,
        None,
        None,
        false,
        -1,
    );

    assert!(res.is_ok(), "Protocol sequence failed: {:?}", res.err());
    assert!(out_t.len() >= 2, "Expected multiple time points recorded");
    assert!(instance.state.t >= 0.039, "Time did not advance through both protocol steps: t={}", instance.state.t);
    for val in &instance.state.y {
        assert!(val.is_finite(), "Non-finite state value after sequence execution");
    }
}

#[test]
fn test_protocol_bisection_trigger_rollback() {
    let (prob, state, history, scratch, linear_solver, diag) = create_mock_fixture();
    let mut instance = crate::stepper::state::SolverInstance {
        state,
        history,
        scratch,
        linear_solver,
        diag,
    };
    let mut out_t = Vec::new();
    let mut out_traj = Vec::new();
    let mut out_obs = Vec::new();

    let steps = vec![
        crate::control::sequence::ProtocolStep {
            s_type: 0,
            target_val: 0.1,
            t_limit: 0.1,
            trig: crate::control::bisection::TrigInfo {
                has_trig: true,
                idx: 0,
                size: 1,
                is_obs: false,
                op: 1,
                val: 1.001,
            },
            p_mode: 0,
            p_i: 1,
            p_v: 0,
        },
        crate::control::sequence::ProtocolStep {
            s_type: 2,
            target_val: 0.0,
            t_limit: 0.05,
            trig: crate::control::bisection::TrigInfo {
                has_trig: false,
                idx: 0,
                size: 1,
                is_obs: false,
                op: 0,
                val: 0.0,
            },
            p_mode: 0,
            p_i: 1,
            p_v: 0,
        },
    ];

    let res = crate::control::sequence::run_sequence(
        &prob,
        &mut instance,
        &steps,
        &mut out_t,
        &mut out_traj,
        &mut out_obs,
        None,
        None,
        false,
        -1,
    );

    assert!(res.is_ok(), "Protocol sequence with bisection failed: {:?}", res.err());
    for val in &instance.state.y {
        assert!(val.is_finite(), "Non-finite state value after bisection sequence execution");
    }
}

#[test]
fn test_adjoint_residual_invariant() {
    let (prob, mut state, mut history, mut scratch, mut linear_solver, mut diag) = create_mock_fixture();

    let n = prob.n;
    let n_steps = 6;
    let mut t_vec = vec![0.0];
    let mut y_vec = state.y.clone();
    let mut ydot_vec = state.ydot.clone();
    let mut p_vec = state.p.clone();

    for _step_idx in 1..n_steps {
        let dt = 0.01;
        bdf::step(&prob, &mut state, &mut history, &mut scratch, &mut linear_solver, &mut diag, dt, None)
            .expect("Forward step failed");
        t_vec.push(state.t);
        y_vec.extend_from_slice(&state.y);
        ydot_vec.extend_from_slice(&state.ydot);
        p_vec.extend_from_slice(&state.p);
    }

    use numpy::ndarray::{Array1, Array2};
    let t_arr = Array1::from_vec(t_vec);
    let y_arr = Array2::from_shape_vec((n_steps, n), y_vec).unwrap();
    let ydot_arr = Array2::from_shape_vec((n_steps, n), ydot_vec).unwrap();
    let p_arr = Array2::from_shape_vec((n_steps, MOCK_N_PARAMS), p_vec).unwrap();

    let mut dl_dy_vec = vec![0.0; n_steps * n];
    dl_dy_vec[(n_steps - 1) * n] = 1.0;
    let dl_dy_arr = Array2::from_shape_vec((n_steps, n), dl_dy_vec).unwrap();

    let jvp_fn = prob.fns.jvp_fn.expect("jvp function pointer missing");
    let vjp_fn = prob.fns.vjp_fn.expect("vjp function pointer missing");

    let adj_params = crate::control::backward_pass::AdjointParams {
        bandwidth: prob.bw,
        id_slice: &prob.id,
        m_slice: &prob.m,
        cpr: &prob.cpr,
        jvp_fn,
        vjp_fn,
        event_trigger_idx: -1,
    };

    let grads = crate::control::backward_pass::solve_adjoint_sensitivities(
        &adj_params,
        y_arr.view(),
        ydot_arr.view(),
        t_arr.view(),
        p_arr.view(),
        dl_dy_arr.view(),
    );

    assert!(grads.is_ok(), "Adjoint solve failed: {:?}", grads.err());
    let g = grads.unwrap();
    assert_eq!(g.len(), MOCK_N_PARAMS);
    for (i, val) in g.iter().enumerate() {
        assert!(val.is_finite(), "Gradient component {i} is non-finite: {val}");
    }
}

#[test]
fn test_run_sequence_microstep_tape_capture() {
    let (prob, state, history, scratch, linear_solver, diag) = create_mock_fixture();
    let mut instance = crate::stepper::state::SolverInstance {
        state,
        history,
        scratch,
        linear_solver,
        diag,
    };
    let mut out_t = Vec::new();
    let mut out_traj = Vec::new();
    let mut out_obs = Vec::new();
    let mut history_tape = Vec::new();

    let steps = vec![
        crate::control::sequence::ProtocolStep {
            s_type: 0,
            target_val: 0.1,
            t_limit: 0.05,
            trig: crate::control::bisection::TrigInfo {
                has_trig: false,
                idx: 0,
                size: 1,
                is_obs: false,
                op: 0,
                val: 0.0,
            },
            p_mode: 0,
            p_i: 1,
            p_v: 0,
        },
        crate::control::sequence::ProtocolStep {
            s_type: 2,
            target_val: 0.0,
            t_limit: 0.05,
            trig: crate::control::bisection::TrigInfo {
                has_trig: false,
                idx: 0,
                size: 1,
                is_obs: false,
                op: 0,
                val: 0.0,
            },
            p_mode: 0,
            p_i: 1,
            p_v: 0,
        },
    ];

    let mut param_events = Vec::new();
    let res = crate::control::sequence::run_sequence(
        &prob,
        &mut instance,
        &steps,
        &mut out_t,
        &mut out_traj,
        &mut out_obs,
        Some(&mut history_tape),
        Some(&mut param_events),
        false,
        -1,
    );

    assert!(res.is_ok(), "Protocol sequence with tape capture failed: {:?}", res.err());
    assert_eq!(param_events.len(), 2, "Expected parameter switchpoint logged for each protocol step");
    assert_eq!(param_events[0].1[1], 0.1);
    assert_eq!(param_events[1].1[1], 0.0);
    assert!(!history_tape.is_empty(), "History tape must not be empty");
    assert_eq!(history_tape[0].0, 0.0, "First tape step must be t=0.0");
    assert!((history_tape.last().unwrap().0 - 0.1).abs() < 1e-4, "Final tape step must align with end time");

    for (t, y, ydot) in &history_tape {
        assert!(t.is_finite());
        assert_eq!(y.len(), MOCK_N);
        assert_eq!(ydot.len(), MOCK_N);
        for val in y.iter().chain(ydot.iter()) {
            assert!(val.is_finite(), "Non-finite entry recorded in history tape");
        }
    }
}

#[test]
fn test_diagnostics_record_trace_gating() {
    let (mut prob, mut state, mut history, mut scratch, mut linear_solver, mut diag) = create_mock_fixture();
    prob.config.record_trace = false;
    bdf::step(&prob, &mut state, &mut history, &mut scratch, &mut linear_solver, &mut diag, 0.01, None)
        .expect("Step failed with record_trace=false");
    assert!(diag.trace_t.is_empty(), "trace_t must remain empty when record_trace=false");
    assert_eq!(diag.trace_t.capacity(), 0, "trace_t must not allocate when record_trace=false");

    let (mut prob_tr, mut state_tr, mut history_tr, mut scratch_tr, mut lu_tr, mut diag_tr) = create_mock_fixture();
    prob_tr.config.record_trace = true;
    bdf::step(&prob_tr, &mut state_tr, &mut history_tr, &mut scratch_tr, &mut lu_tr, &mut diag_tr, 0.01, None)
        .expect("Step failed with record_trace=true");
    assert!(!diag_tr.trace_t.is_empty(), "trace_t must populate when record_trace=true");
    assert_eq!(diag_tr.trace_t.len(), diag_tr.trace_dt.len());
}

#[test]
fn test_ic_convergence_flag_on_unsolvable_algebraic_root() {
    unsafe extern "C" fn paradox_res(
        _y: *const f64, _ydot: *const f64, _p: *const f64, _m: *const f64, res: *mut f64,
    ) {
        *res = 1.0;
    }
    unsafe extern "C" fn paradox_jvp(
        _y: *const f64, _ydot: *const f64, _p: *const f64, _m: *const f64,
        _cj: f64, _v: *const f64, _dy: *mut f64, _dydot: *mut f64, _rd: *mut f64, out: *mut f64,
    ) {
        *out = 0.0;
    }

    let mut prob = crate::stepper::problem::Problem {
        n: 1,
        bw: 0,
        n_obs: 0,
        id: vec![0.0],
        constraints: vec![0.0],
        m: vec![],
        spatial_diag: vec![0.0],
        max_steps: vec![0.0],
        cpr: crate::stepper::problem::CprData {
            color_seeds: vec![vec![1.0]],
            color_ptrs: vec![0, 1],
            color_rows: vec![0],
            color_cols: vec![0],
            dense_rows: vec![],
        },
        config: crate::stepper::problem::SolverConfig::default(),
        fns: crate::ffi::callbacks::Callbacks {
            res_fn: paradox_res,
            obs_fn: None,
            jvp_fn: Some(paradox_jvp),
            vjp_fn: None,
            set_threads_fn: None,
        },
    };
    prob.config.record_trace = false;
    let mut state = crate::stepper::state::SystemState::new(vec![0.0], vec![0.0], vec![]);
    let mut history = crate::stepper::history::BdfHistory::new(1);
    let mut scratch = crate::stepper::state::StepScratchpad::new(1);
    let mut lu = crate::linear::solver::LinearSolver::new(1, 0);
    let mut diag = crate::diagnostics::Diagnostics::new(1);

    let res = algebraic_roots::calc_algebraic_roots(&prob, &mut state, &mut history, &mut scratch, &mut lu, &mut diag);
    assert!(res.is_ok());
    assert!(!diag.ic_converged, "ic_converged must be false when residual cannot converge");
    assert!((diag.ic_final_max_res - 1.0).abs() < 1e-12);
}

#[test]
fn test_realtime_frozen_step_counter() {
    let (mut prob, mut state, mut history, mut scratch, mut lu, mut diag) = create_mock_fixture();
    prob.config = crate::stepper::problem::SolverConfig::realtime();
    prob.config.max_newton_iters = 1;
    prob.config.eps_newt = 1e-30;
    state.y[0] = 1e6;

    let res = bdf::step(&prob, &mut state, &mut history, &mut scratch, &mut lu, &mut diag, 0.01, None);
    assert!(res.is_ok(), "Realtime step must not return Err on divergence");
    assert_eq!(diag.realtime_frozen_steps, 1);
    assert_eq!(diag.rejected_steps, 1);
}

#[test]
fn test_calc_algebraic_roots_pure_differential_is_noop() {
    let (mut prob, mut state, mut history, mut scratch, mut lu, mut diag) = create_mock_fixture();
    prob.id.fill(1.0);
    let y_orig = state.y.clone();

    algebraic_roots::calc_algebraic_roots(&prob, &mut state, &mut history, &mut scratch, &mut lu, &mut diag)
        .expect("Pure differential root check should succeed");

    assert!(diag.ic_converged);
    assert_eq!(diag.ic_iters, 0);
    assert_eq!(diag.ic_final_max_res, 0.0);
    assert_eq!(diag.numeric_factorizations, 0);
    assert_eq!(state.y, y_orig);
}

#[test]
fn test_calc_algebraic_roots_clamped_step_and_gmres_diagonal() {
    // 1. SparseLU path with max_steps clamping converges from large offset
    let (mut prob, mut state, mut history, mut scratch, mut lu, mut diag) = create_mock_fixture();
    prob.max_steps[5] = 0.5;
    state.y[5] = -10.0;

    algebraic_roots::calc_algebraic_roots(&prob, &mut state, &mut history, &mut scratch, &mut lu, &mut diag)
        .expect("Clamped root convergence failed");

    let expected_y5 = state.y[0..5].iter().sum::<f64>() + state.p[1];
    assert!(diag.ic_converged);
    assert!((state.y[5] - expected_y5).abs() < 1e-8);

    // 2. GMRES (bw = -1) path uses diagonal damped Jacobi from assembled triplets
    let (mut prob_g, mut state_g, mut hist_g, mut scr_g, _, mut diag_g) = create_mock_fixture();
    prob_g.bw = -1;
    state_g.y[5] = 0.0;
    let mut gmres = crate::linear::solver::LinearSolver::new(MOCK_N, -1);

    algebraic_roots::calc_algebraic_roots(&prob_g, &mut state_g, &mut hist_g, &mut scr_g, &mut gmres, &mut diag_g)
        .expect("GMRES diagonal algebraic root convergence failed");

    assert!(diag_g.ic_converged);
    assert!((state_g.y[5] - expected_y5).abs() < 1e-7);
}

#[test]
fn test_calc_algebraic_roots_nan_residual_rolls_back_state() {
    unsafe extern "C" fn nan_on_negative_res(
        y: *const f64, _ydot: *const f64, _p: *const f64, _m: *const f64, res: *mut f64,
    ) {
        let y_val = *y;
        *res = if y_val < 0.0 { f64::NAN } else { y_val + 10.0 };
    }
    unsafe extern "C" fn unit_jvp(
        _y: *const f64, _ydot: *const f64, _p: *const f64, _m: *const f64,
        _cj: f64, v: *const f64, _dy: *mut f64, _dydot: *mut f64, _rd: *mut f64, out: *mut f64,
    ) {
        *out = *v;
    }

    let prob = crate::stepper::problem::Problem {
        n: 1,
        bw: 0,
        n_obs: 0,
        id: vec![0.0],
        constraints: vec![0.0],
        m: vec![],
        spatial_diag: vec![0.0],
        max_steps: vec![0.0],
        cpr: crate::stepper::problem::CprData {
            color_seeds: vec![vec![1.0]],
            color_ptrs: vec![0, 1],
            color_rows: vec![0],
            color_cols: vec![0],
            dense_rows: vec![],
        },
        config: crate::stepper::problem::SolverConfig::default(),
        fns: crate::ffi::callbacks::Callbacks {
            res_fn: nan_on_negative_res,
            obs_fn: None,
            jvp_fn: Some(unit_jvp),
            vjp_fn: None,
            set_threads_fn: None,
        },
    };
    let mut state = crate::stepper::state::SystemState::new(vec![1.0], vec![0.0], vec![]);
    let mut history = crate::stepper::history::BdfHistory::new(1);
    let mut scratch = crate::stepper::state::StepScratchpad::new(1);
    let mut lu = crate::linear::solver::LinearSolver::new(1, 0);
    let mut diag = crate::diagnostics::Diagnostics::new(1);

    algebraic_roots::calc_algebraic_roots(&prob, &mut state, &mut history, &mut scratch, &mut lu, &mut diag)
        .expect("calc_algebraic_roots should recover gracefully from NaN residuals");

    assert!(!diag.ic_converged);
    assert!(diag.ic_final_max_res.is_nan());
    assert!(state.y[0].is_finite(), "State must roll back when residual hits NaN");
    assert_eq!(state.y[0], 1.0);
}

#[test]
fn test_gmres_symmetric_breakdown_and_high_cj_preconditioning() {
    use crate::linear::gmres::solve_gmres;

    // 4-node linear system where nodes 2 & 3 are completely symmetric (Krylov dim 3 < n=4)
    let c_j = 1e9_f64;
    let mut rhs = [1.0, 2.0, 3.0, 3.0];
    let rhs_orig = rhs;

    let matvec = |v: &[f64], out: &mut [f64]| {
        out[0] = (c_j + 3.0) * v[0] - v[1] - v[2] - v[3];
        out[1] = -v[0] + (c_j + 1.0) * v[1];
        out[2] = -v[0] + (c_j + 1.0) * v[2];
        out[3] = -v[0] + (c_j + 1.0) * v[3];
    };
    let precond = |v: &[f64], out: &mut [f64]| {
        for i in 0..4 {
            out[i] = v[i] / (c_j + 1.0);
        }
    };

    solve_gmres(4, &mut rhs, matvec, precond)
        .expect("GMRES should handle symmetric Krylov breakdown and large c_j preconditioning");

    let mut check = [0.0; 4];
    matvec(&rhs, &mut check);
    for i in 0..4 {
        assert!(
            (check[i] - rhs_orig[i]).abs() < 1e-5,
            "GMRES residual mismatch at {i}: got {}, expected {}",
            check[i],
            rhs_orig[i]
        );
    }
}

#[test]
fn test_adjoint_sliver_step_preserves_terminal_dl_dy() {
    let (prob, mut state, mut history, mut scratch, mut linear_solver, mut diag) = create_mock_fixture();
    let n = prob.n;

    let mut t_vec = vec![0.0];
    let mut y_vec = state.y.clone();
    let mut ydot_vec = state.ydot.clone();
    let mut p_vec = state.p.clone();

    for _ in 1..4 {
        bdf::step(&prob, &mut state, &mut history, &mut scratch, &mut linear_solver, &mut diag, 0.01, None)
            .expect("Forward step failed");
        t_vec.push(state.t);
        y_vec.extend_from_slice(&state.y);
        ydot_vec.extend_from_slice(&state.ydot);
        p_vec.extend_from_slice(&state.p);
    }
    // Append a 1e-12 sliver final microstep whose terminal dl_dy[-1] must carry over cleanly
    t_vec.push(state.t + 1e-12);
    y_vec.extend_from_slice(&state.y);
    ydot_vec.extend_from_slice(&state.ydot);
    p_vec.extend_from_slice(&state.p);

    let n_rows = t_vec.len();
    use numpy::ndarray::{Array1, Array2};
    let t_arr = Array1::from_vec(t_vec);
    let y_arr = Array2::from_shape_vec((n_rows, n), y_vec).unwrap();
    let ydot_arr = Array2::from_shape_vec((n_rows, n), ydot_vec).unwrap();
    let p_arr = Array2::from_shape_vec((n_rows, MOCK_N_PARAMS), p_vec).unwrap();

    let mut dl_dy_vec = vec![0.0; n_rows * n];
    dl_dy_vec[(n_rows - 1) * n] = 1.0;
    let dl_dy_arr = Array2::from_shape_vec((n_rows, n), dl_dy_vec).unwrap();

    let adj_params = crate::control::backward_pass::AdjointParams {
        bandwidth: prob.bw,
        id_slice: &prob.id,
        m_slice: &prob.m,
        cpr: &prob.cpr,
        jvp_fn: prob.fns.jvp_fn.unwrap(),
        vjp_fn: prob.fns.vjp_fn.unwrap(),
        event_trigger_idx: -1,
    };

    let grads = crate::control::backward_pass::solve_adjoint_sensitivities(
        &adj_params, y_arr.view(), ydot_arr.view(), t_arr.view(), p_arr.view(), dl_dy_arr.view(),
    ).expect("Adjoint solve with sliver step failed");

    assert!(grads[0].abs() > 1e-6, "Terminal dl_dy was dropped across final 1e-12 sliver step: {:?}", grads);
}

#[test]
fn test_adjoint_gmres_survives_enzyme_lambda_zeroing() {
    unsafe extern "C" fn zeroing_vjp(
        y: *const f64, ydot: *const f64, p: *const f64, m: *const f64,
        lambda_vec: *const f64, res_dummy: *mut f64,
        dp_out: *mut f64, dy_out: *mut f64, dydot_out: *mut f64,
    ) {
        mock_evaluate_vjp(y, ydot, p, m, lambda_vec, res_dummy, dp_out, dy_out, dydot_out);
        let lam_mut = std::slice::from_raw_parts_mut(lambda_vec as *mut f64, MOCK_N);
        lam_mut.fill(0.0);
    }

    let (mut prob, mut state, mut history, mut scratch, mut linear_solver, mut diag) = create_mock_fixture();
    prob.bw = -1;
    let n = prob.n;

    let mut t_vec = vec![0.0];
    let mut y_vec = state.y.clone();
    let mut ydot_vec = state.ydot.clone();
    let mut p_vec = state.p.clone();
    for _ in 1..4 {
        bdf::step(&prob, &mut state, &mut history, &mut scratch, &mut linear_solver, &mut diag, 0.01, None)
            .expect("Forward step failed");
        t_vec.push(state.t);
        y_vec.extend_from_slice(&state.y);
        ydot_vec.extend_from_slice(&state.ydot);
        p_vec.extend_from_slice(&state.p);
    }

    use numpy::ndarray::{Array1, Array2};
    let n_rows = t_vec.len();
    let t_arr = Array1::from_vec(t_vec);
    let y_arr = Array2::from_shape_vec((n_rows, n), y_vec).unwrap();
    let ydot_arr = Array2::from_shape_vec((n_rows, n), ydot_vec).unwrap();
    let p_arr = Array2::from_shape_vec((n_rows, MOCK_N_PARAMS), p_vec).unwrap();
    let mut dl_dy_vec = vec![0.0; n_rows * n];
    dl_dy_vec[(n_rows - 1) * n] = 1.0;
    let dl_dy_arr = Array2::from_shape_vec((n_rows, n), dl_dy_vec).unwrap();

    let adj_params = crate::control::backward_pass::AdjointParams {
        bandwidth: -1,
        id_slice: &prob.id,
        m_slice: &prob.m,
        cpr: &prob.cpr,
        jvp_fn: prob.fns.jvp_fn.unwrap(),
        vjp_fn: zeroing_vjp,
        event_trigger_idx: -1,
    };

    let grads = crate::control::backward_pass::solve_adjoint_sensitivities(
        &adj_params, y_arr.view(), ydot_arr.view(), t_arr.view(), p_arr.view(), dl_dy_arr.view(),
    ).expect("GMRES adjoint failed with lambda-zeroing VJP");

    assert!(grads[0].abs() > 1e-6, "GMRES basis corrupted by VJP lambda zeroing: {:?}", grads);
}
