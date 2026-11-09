use std::io::Write;
use crate::solver::shared::problem::Problem;
use crate::stepper::state::{SystemState, StepScratchpad};
use crate::stepper::history::BdfHistory;
use crate::linear::solver::LinearSolver;
use crate::solver::shared::diagnostics::Diagnostics;
use crate::nonlinear::algebraic_roots;
use crate::stepper::bdf;
use crate::control::bisection::{TrigInfo, check_trigger, execute_bisection};
pub use crate::solver::_0_ffi::SolverError;

#[derive(Clone, Debug)]
pub struct ProtocolStep {
    pub s_type: i32, pub target_val: f64, pub t_limit: f64,
    pub trig: TrigInfo, pub p_mode: usize, pub p_i: usize, pub p_v: usize,
}

pub fn run_sequence(
    prob: &Problem,
    state: &mut SystemState,
    history: &mut BdfHistory,
    scratch: &mut StepScratchpad,
    linear_solver: &mut LinearSolver,
    diag: &mut Diagnostics,
    steps: &[ProtocolStep],
    out_t: &mut Vec<f64>,
    out_traj: &mut Vec<f64>,
    out_obs: &mut Vec<f64>,
    show_progress: bool,
    v_idx: i32
) -> Result<(), SolverError> {
    let mut step_obs = vec![0.0; prob.n_obs];
    
    for (step_idx, step) in steps.iter().enumerate() {
        if step.s_type == 0 { state.p[step.p_mode] = 1.0; state.p[step.p_i] = step.target_val; }
        else if step.s_type == 1 { state.p[step.p_mode] = 0.0; state.p[step.p_v] = step.target_val; }
        else if step.s_type == 2 { state.p[step.p_mode] = 1.0; state.p[step.p_i] = 0.0; }
        
        algebraic_roots::calc_algebraic_roots(prob, state, history, scratch, linear_solver, diag)?;
        
        if step_idx == 0 {
            out_t.push(state.t);
            out_traj.extend_from_slice(&state.y);
            if prob.n_obs > 0 {
                if let Some(obs_fn) = prob.fns.obs_fn {
                    unsafe { obs_fn(state.y.as_ptr(), state.ydot.as_ptr(), state.p.as_ptr(), prob.m.as_ptr(), step_obs.as_mut_ptr()); }
                    out_obs.extend_from_slice(&step_obs);
                }
            }
        }
        
        let mut t_elapsed = 0.0;
        
        while t_elapsed < step.t_limit {
            if step.t_limit == std::f64::INFINITY && !step.trig.has_trig { break; } 
            let dt_step = 1.0_f64.min(step.t_limit - t_elapsed);
            let ckpt = state.clone_tuple();
            let target_t = state.t + dt_step;
            
            bdf::advance_to(prob, state, history, scratch, linear_solver, diag, target_t, None)?;
            
            if check_trigger(prob, state, &step.trig, &mut step_obs) {
                let low = execute_bisection(prob, state, history, scratch, linear_solver, diag, dt_step, &step.trig, ckpt);
                t_elapsed += low;
                out_t.push(state.t);
                out_traj.extend_from_slice(&state.y);
                if prob.n_obs > 0 {
                    if let Some(obs_fn) = prob.fns.obs_fn {
                        unsafe { obs_fn(state.y.as_ptr(), state.ydot.as_ptr(), state.p.as_ptr(), prob.m.as_ptr(), step_obs.as_mut_ptr()); }
                        out_obs.extend_from_slice(&step_obs);
                    }
                }
                if show_progress { print_progress(step.s_type, state.t, 1.0, v_idx, &state.y, true); }
                break;
            }
            
            t_elapsed += dt_step;
            out_t.push(state.t);
            out_traj.extend_from_slice(&state.y);
            if prob.n_obs > 0 {
                if let Some(obs_fn) = prob.fns.obs_fn {
                    unsafe { obs_fn(state.y.as_ptr(), state.ydot.as_ptr(), state.p.as_ptr(), prob.m.as_ptr(), step_obs.as_mut_ptr()); }
                    out_obs.extend_from_slice(&step_obs);
                }
            }
            if show_progress { print_progress(step.s_type, state.t, t_elapsed / step.t_limit, v_idx, &state.y, false); }
        }
        if show_progress && t_elapsed >= step.t_limit && step.t_limit != std::f64::INFINITY {
            print_progress(step.s_type, state.t, 1.0, v_idx, &state.y, true);
        }
    }
    Ok(())
}

fn print_progress(s_type: i32, t: f64, pct: f64, v_idx: i32, y: &[f64], is_final: bool) {
    let name = match s_type { 0 => "CC", 1 => "CV", 2 => "Rest", _ => "Step" };
    let v_str = if v_idx >= 0 { format!(" | V: {:.3}V", y[v_idx as usize]) } else { String::new() };
    
    if is_final {
        print!("\r▶ {:<4} [██████████████████████████████] 100.0% | t: {:.1}s{}   \n", name, t, v_str);
    } else if pct.is_nan() || pct.is_infinite() {
        print!("\r▶ {:<4} ⏳ t: {:.1}s{}   ", name, t, v_str);
    } else {
        let p = pct.clamp(0.0, 1.0);
        let filled = (p * 30.0) as usize;
        let bar: String = std::iter::repeat('█').take(filled).chain(std::iter::repeat('-').take(30 - filled)).collect();
        print!("\r▶ {:<4} [{}] {:.1}% | t: {:.1}s{}   ", name, bar, p * 100.0, t, v_str);
    }
    std::io::stdout().flush().unwrap();
}
