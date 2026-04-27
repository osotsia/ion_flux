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

{{ jac_block_funcs }}

void evaluate_jacobian_sparse(const double* y, const double* ydot, const double* p, const double* m, double c_j, int* out_rows, int* out_cols, double* out_vals, int* out_nnz) {
    int N = {{ n_states }};
    std::vector<double> dy(N, 0.0);
    std::vector<double> dydot(N, 0.0);
    int nnz = 0;
    int max_nnz = N * 50;

{{ jac_assembly_body | indent(4) }}

    *out_nnz = nnz;
}

void evaluate_jvp(const double* y, const double* ydot, const double* p, const double* m, double c_j, const double* v, double* jvp_out) {
    int N = {{ n_states }};
    std::vector<double> dy(N, 0.0);
    std::vector<double> dydot(N, 0.0);
    std::vector<double> res_dummy(N, 0.0);

    for (int i = 0; i < N; ++i) {
        dy[i] = v[i];
        dydot[i] = c_j * v[i];
        jvp_out[i] = 0.0;
    }

#ifdef ENZYME_ACTIVE
    __enzyme_fwddiff((void*)evaluate_residual, 
        enzyme_dup, y, dy.data(), 
        enzyme_dup, ydot, dydot.data(), 
        enzyme_const, p, 
        enzyme_const, m,
        enzyme_dupnoneed, res_dummy.data(), jvp_out);
#endif
}

void evaluate_vjp(const double* y, const double* ydot, const double* p, const double* m, const double* lambda_vec, double* dp_out, double* dy_out, double* dydot_out) {
    int N = {{ n_states }};
    int N_P = {{ n_params }};
    std::vector<double> res_dummy(N, 0.0);
    
    for(int i=0; i<N_P; ++i) dp_out[i] = 0.0;
    for(int i=0; i<N; ++i) { dy_out[i] = 0.0; dydot_out[i] = 0.0; }

#ifdef ENZYME_ACTIVE
    __enzyme_autodiff((void*)evaluate_residual, 
        enzyme_dup, y, dy_out, 
        enzyme_dup, ydot, dydot_out, 
        enzyme_dup, p, dp_out, 
        enzyme_const, m,
        enzyme_dupnoneed, res_dummy.data(), (double*)lambda_vec);
#endif
}
} // extern "C"
"""

def generate_cpp_skeleton(n_states: int, n_params: int, n_obs: int, body: str, obs_body: str, jac_block_funcs: str, jac_assembly_body: str) -> str:
    env = jinja2.Environment()
    template = env.from_string(TEMPLATE)
    return template.render(
        n_states=n_states,
        n_params=n_params,
        n_obs=n_obs,
        body=body,
        obs_body=obs_body,
        jac_block_funcs=jac_block_funcs,
        jac_assembly_body=jac_assembly_body
    )