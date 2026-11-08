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

namespace {
    thread_local std::vector<double> tl_jvp_dy;
    thread_local std::vector<double> tl_jvp_dydot;
    thread_local std::vector<double> tl_res_dummy;
}

extern "C" {

void set_spatial_threads(int num_threads) {
#if defined(_OPENMP)
    omp_set_num_threads(num_threads);
#endif
}

void evaluate_residual(const double* __restrict__ y, const double* __restrict__ ydot, const double* __restrict__ p, const double* __restrict__ m, double* __restrict__ res) {
{{ body | indent(4) }}
}

void evaluate_observables(const double* __restrict__ y, const double* __restrict__ ydot, const double* __restrict__ p, const double* __restrict__ m, double* __restrict__ obs) {
{{ obs_body | indent(4) }}
}

void evaluate_jvp(const double* __restrict__ y, const double* __restrict__ ydot, const double* __restrict__ p, const double* __restrict__ m, double c_j, const double* __restrict__ v, double* __restrict__ jvp_out) {
    int N = {{ n_states }};
    if ((int)tl_jvp_dy.size() < N) {
        tl_jvp_dy.resize(N);
        tl_jvp_dydot.resize(N);
        tl_res_dummy.resize(N);
    }

    for (int i = 0; i < N; ++i) {
        tl_jvp_dy[i] = v[i];
        tl_jvp_dydot[i] = c_j * v[i];
        jvp_out[i] = 0.0;
        tl_res_dummy[i] = 0.0;
    }

#ifdef ENZYME_ACTIVE
    __enzyme_fwddiff((void*)evaluate_residual, 
        enzyme_dup, y, tl_jvp_dy.data(), 
        enzyme_dup, ydot, tl_jvp_dydot.data(), 
        enzyme_const, p, 
        enzyme_const, m,
        enzyme_dupnoneed, tl_res_dummy.data(), jvp_out);
#endif
}

void evaluate_vjp(const double* __restrict__ y, const double* __restrict__ ydot, const double* __restrict__ p, const double* __restrict__ m, const double* __restrict__ lambda_vec, double* __restrict__ dp_out, double* __restrict__ dy_out, double* __restrict__ dydot_out) {
    int N = {{ n_states }};
    int N_P = {{ n_params }};
    if ((int)tl_res_dummy.size() < N) {
        tl_res_dummy.resize(N);
    }
    for (int i = 0; i < N; ++i) {
        tl_res_dummy[i] = 0.0;
    }
    
    for(int i=0; i<N_P; ++i) dp_out[i] = 0.0;
    for(int i=0; i<N; ++i) { dy_out[i] = 0.0; dydot_out[i] = 0.0; }

#ifdef ENZYME_ACTIVE
    __enzyme_autodiff((void*)evaluate_residual, 
        enzyme_dup, y, dy_out, 
        enzyme_dup, ydot, dydot_out, 
        enzyme_dup, p, dp_out, 
        enzyme_const, m,
        enzyme_dupnoneed, tl_res_dummy.data(), (double*)lambda_vec);
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