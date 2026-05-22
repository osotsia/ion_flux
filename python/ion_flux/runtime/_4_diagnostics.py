import json
from ion_flux.runtime.manifest import ExecutableManifest

def format_native_crash(original_error: Exception, manifest: ExecutableManifest) -> Exception:
    """
    Parses Native Rust Crash JSONs (embedded in the error message) and maps physical Python AST variable names 
    back to the faulty flat C-arrays that caused the crash.
    """
    try:
        err_str = str(original_error)
        
        # Try to parse the embedded JSON from the Rust panic message
        start_idx = err_str.find('{')
        end_idx = err_str.rfind('}')
        if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
            raise ValueError("No JSON payload found in the Native Rust Error string.")
            
        json_str = err_str[start_idx:end_idx+1]
        
        # Rust's f64 formatter produces 'inf', '-inf', and 'NaN' for floating point extremities. 
        # Python's strict json parser requires 'Infinity', '-Infinity', and 'NaN'.
        json_str = json_str.replace(": inf", ": Infinity")
        json_str = json_str.replace(": -inf", ": -Infinity")
        
        crash_data = json.loads(json_str)
        
        if crash_data.get("status") != "CRASH":
            raise ValueError("Valid JSON found, but it is not a solver crash report.")
            
        idx_to_name = {}
        for name, (offset, size) in manifest.layout.state_offsets.items():
            for i in range(size): 
                idx_to_name[offset + i] = f"{name}[{i}]" if size > 1 else name
        
        for off in crash_data.get("top_offenders", []):
            off["name"] = idx_to_name.get(off.get("index", -1), f"Unknown[{off.get('index', -1)}]")
            
        if "initialization_health" in crash_data:
            idx = crash_data["initialization_health"].get("t0_max_residual_index", -1)
            crash_data["initialization_health"]["t0_max_residual_name"] = idx_to_name.get(idx, f"Unknown[{idx}]")
                
        msg = f"\n{'-'*100}\n🔥 NATIVE SOLVER CRASH\n{'-'*100}\n"
        msg += f"Reason: {crash_data.get('reason', 'Unknown')}\n"
        msg += f"Accepted Steps: {crash_data.get('accepted_steps', 0)}\n"
        
        init_health = crash_data.get("initialization_health", {})
        if init_health.get("t0_max_residual", 0.0) > 1e3:
            msg += f"\n⚠️ INITIALIZATION WARNING: Massive residual at t=0 detected!\n"
            msg += f"   Variable: {init_health.get('t0_max_residual_name')} (Residual: {init_health.get('t0_max_residual'):.3e})\n"
            msg += f"   Check your `initial_conditions` for severe algebraic imbalances.\n"
            
        jac_health = crash_data.get("jacobian_health", {})
        if jac_health.get("condition_warning", False):
            msg += f"\n⚠️ JACOBIAN CONDITION WARNING: Matrix is likely singular or badly scaled.\n"
            
        trace = crash_data.get("newton_thrashing_trace", [])
        if trace:
            msg += f"\nNewton Trace (Last {len(trace)} iterations):\n"
            for t in trace: msg += f"   Iter {t.get('iter')}: Residual Norm = {t.get('residual_norm'):.3e}, Step Norm = {t.get('step_norm'):.3e}\n"
        
        msg += f"\nTop Offenders (Ranked by NaN presence, then Absolute Residual):\n"
        msg += f"{'State Name':<25} | {'Type':<9} | {'Residual':<10} | {'Weight':<9} | {'Step dy':<10} | {'y_val':<10}\n"
        msg += "-" * 100 + "\n"
        
        for off in crash_data.get("top_offenders", []):
            name, rtype = off.get("name", ""), off.get("type", "")
            def fmt(v): return f"{float(v):<10.3e}" if isinstance(v, (float, int)) else f"{v:<10}"
            msg += f"{name[:24]:<25} | {rtype:<9} | {fmt(off.get('residual', 0.0))} | {fmt(off.get('solver_weight', 0.0))} | {fmt(off.get('proposed_step_dy', 0.0))} | {fmt(off.get('y_val', 0.0))}\n"
            
        msg += f"{'-'*100}\n"
        return RuntimeError(msg)
    except Exception as e:
        # Gracefully fallback to the original error if JSON parsing or string processing fails,
        # but append the parsing error for diagnostic context.
        return RuntimeError(f"{str(original_error)}\n\n[Diagnostic Formatting Failed: {e}]")