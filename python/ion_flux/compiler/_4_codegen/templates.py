import jinja2

TEMPLATE = """
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>

#if defined(_OPENMP)
#include <omp.h>
#endif

#define CLAMP(idx, bound) (std::max(0, std::min((int)(idx), (int)(bound) - 1)))

#ifdef ENZYME_ACTIVE
int enzyme_dup = 1;
int enzyme_const = 2;
int enzyme_dupnoneed = 3;
extern void __enzyme_fwddiff(void*, ...);
extern void __enzyme_autodiff(void*, ...);
#endif

extern "C" {

void set_spatial_threads(int num_threads) {
#if defined(_OPENMP)
    omp_set_num_threads(num_threads);
#endif
}

void evaluate_residual(const double* y, const double* ydot, const double* p, const double* m, double* res) {
{{ body | indent(4) }}
}

void evaluate_observables(const double* y, const double* ydot, const double* p, const double* m, double* obs) {
{{ obs_body | indent(4) }}
}

void evaluate_jvp(const double* y, const double* ydot, const double* p, const double* m, double c_j, const double* v, double* jvp_out) {
    int N = {{ n_states }};
#if {{ n_states }} <= 2048
    double jvp_dy[{{ n_states }}];
    double jvp_dydot[{{ n_states }}];
    double res_dummy[{{ n_states }}] = {0.0};
    double* p_dy = jvp_dy;
    double* p_dydot = jvp_dydot;
    double* p_res = res_dummy;
#else
    std::vector<double> jvp_dy(N);
    std::vector<double> jvp_dydot(N);
    std::vector<double> res_dummy(N, 0.0);
    double* p_dy = jvp_dy.data();
    double* p_dydot = jvp_dydot.data();
    double* p_res = res_dummy.data();
#endif

    for (int i = 0; i < N; ++i) {
        p_dy[i] = v[i];
        p_dydot[i] = c_j * v[i];
        jvp_out[i] = 0.0;
    }

#ifdef ENZYME_ACTIVE
    __enzyme_fwddiff((void*)evaluate_residual, 
        enzyme_dup, y, p_dy, 
        enzyme_dup, ydot, p_dydot, 
        enzyme_const, p, 
        enzyme_const, m,
        enzyme_dupnoneed, p_res, jvp_out);
#endif
}

void evaluate_vjp(const double* y, const double* ydot, const double* p, const double* m, const double* lambda_vec, double* dp_out, double* dy_out, double* dydot_out) {
    int N = {{ n_states }};
    int N_P = {{ n_params }};
#if {{ n_states }} <= 2048
    double res_dummy[{{ n_states }}] = {0.0};
    double* p_res = res_dummy;
#else
    std::vector<double> res_dummy(N, 0.0);
    double* p_res = res_dummy.data();
#endif
    
    for(int i = 0; i < N_P; ++i) dp_out[i] = 0.0;
    for(int i = 0; i < N; ++i) { dy_out[i] = 0.0; dydot_out[i] = 0.0; }

#ifdef ENZYME_ACTIVE
    __enzyme_autodiff((void*)evaluate_residual, 
        enzyme_dup, y, dy_out, 
        enzyme_dup, ydot, dydot_out, 
        enzyme_dup, p, dp_out, 
        enzyme_const, m,
        enzyme_dupnoneed, p_res, (double*)lambda_vec);
#endif
}
} // extern "C"
"""

def generate_cpp_skeleton(n_states: int, n_params: int, n_obs: int, body: str, obs_body: str) -> str:
    env = jinja2.Environment()
    template = env.from_string(TEMPLATE)
    return template.render(
        n_states=n_states,
        n_params=n_params,
        n_obs=n_obs,
        body=body,
        obs_body=obs_body
    )