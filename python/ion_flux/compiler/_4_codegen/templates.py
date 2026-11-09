import jinja2

'''
 Enforced `__restrict__` qualifiers across all generated C++ kernel entry points 
 (`evaluate_residual`, `evaluate_jvp`, `evaluate_vjp`, `evaluate_observables`) 
 to emit LLVM `noalias` parameter metadata. This enables Clang loop vectorization 
 and prevents Enzyme AD from generating conservative shadow loads under assumed pointer aliasing.
 '''

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

void evaluate_residual(const double* __restrict__ y, const double* __restrict__ ydot, const double* __restrict__ p, const double* __restrict__ m, double* __restrict__ res) {
{{ body | indent(4) }}
}

void evaluate_observables(const double* __restrict__ y, const double* __restrict__ ydot, const double* __restrict__ p, const double* __restrict__ m, double* __restrict__ obs) {
{{ obs_body | indent(4) }}
}

void evaluate_jvp(
    const double* __restrict__ y,
    const double* __restrict__ ydot,
    const double* __restrict__ p,
    const double* __restrict__ m,
    double c_j,
    const double* __restrict__ v,
    double* __restrict__ jvp_dy,
    double* __restrict__ jvp_dydot,
    double* __restrict__ res_dummy,
    double* __restrict__ jvp_out
) {
    int N = {{ n_states }};
    for (int i = 0; i < N; ++i) {
        jvp_dy[i] = v[i];
        jvp_dydot[i] = c_j * v[i];
        jvp_out[i] = 0.0;
        res_dummy[i] = 0.0;
    }

#ifdef ENZYME_ACTIVE
    __enzyme_fwddiff((void*)evaluate_residual, 
        enzyme_dup, y, jvp_dy, 
        enzyme_dup, ydot, jvp_dydot, 
        enzyme_const, p, 
        enzyme_const, m,
        enzyme_dupnoneed, res_dummy, jvp_out);
#endif
}

void evaluate_vjp(
    const double* __restrict__ y,
    const double* __restrict__ ydot,
    const double* __restrict__ p,
    const double* __restrict__ m,
    const double* __restrict__ lambda_vec,
    double* __restrict__ res_dummy,
    double* __restrict__ dp_out,
    double* __restrict__ dy_out,
    double* __restrict__ dydot_out
) {
    int N = {{ n_states }};
    int N_P = {{ n_params }};
    for (int i = 0; i < N; ++i) {
        res_dummy[i] = 0.0;
        dy_out[i] = 0.0;
        dydot_out[i] = 0.0;
    }
    for (int i = 0; i < N_P; ++i) {
        dp_out[i] = 0.0;
    }

#ifdef ENZYME_ACTIVE
    __enzyme_autodiff((void*)evaluate_residual, 
        enzyme_dup, y, dy_out, 
        enzyme_dup, ydot, dydot_out, 
        enzyme_dup, p, dp_out, 
        enzyme_const, m,
        enzyme_dupnoneed, res_dummy, (double*)lambda_vec);
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