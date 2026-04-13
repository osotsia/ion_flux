import textwrap

def generate_cpp_skeleton(n_states: int, n_params: int, n_obs: int, body: str, obs_body: str, bandwidth: int) -> str:
    
    if bandwidth > 0:
        jacobian_logic = textwrap.dedent(f"""\
            int bw = {bandwidth};
            int stride = 2 * bw + 1;
            for (int color = 0; color < stride; ++color) {{
                for (int i = 0; i < N; ++i) {{
                    bool active = ((i % stride) == color);
                    dy[i] = active ? 1.0 : 0.0;
                    dydot[i] = active ? c_j : 0.0;
                    dres[i] = 0.0;
                }}
        #ifdef ENZYME_ACTIVE
                __enzyme_fwddiff((void*)evaluate_residual, enzyme_dup, y, dy.data(), enzyme_dup, ydot, dydot.data(), enzyme_const, p, enzyme_const, m, enzyme_dup, res_dummy.data(), dres.data());
        #endif
                for (int row = 0; row < N; ++row) {{
                    int col_base = row - (row % stride) + color;
                    int actual_col = -1;
                    if (std::abs(row - col_base) <= bw) actual_col = col_base;
                    else if (std::abs(row - (col_base - stride)) <= bw) actual_col = col_base - stride;
                    else if (std::abs(row - (col_base + stride)) <= bw) actual_col = col_base + stride;

                    if (actual_col >= 0 && actual_col < N) {{
                        jac_out[actual_col * N + row] = dres[row];
                    }}
                }}
            }}""")
    else:
        jacobian_logic = textwrap.dedent("""\
            for (int col = 0; col < N; ++col) {
                for (int i = 0; i < N; ++i) {
                    dy[i] = (i == col) ? 1.0 : 0.0;
                    dydot[i] = (i == col) ? c_j : 0.0;
                    dres[i] = 0.0;
                }
        #ifdef ENZYME_ACTIVE
                __enzyme_fwddiff((void*)evaluate_residual, enzyme_dup, y, dy.data(), enzyme_dup, ydot, dydot.data(), enzyme_const, p, enzyme_const, m, enzyme_dup, res_dummy.data(), dres.data());
        #endif
                for (int row = 0; row < N; ++row) {
                    jac_out[col * N + row] = dres[row];
                }
            }""")

    return textwrap.dedent(f"""\
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
        extern void __enzyme_fwddiff(void*, ...);
        extern void __enzyme_autodiff(void*, ...);
        #endif

        extern "C" {{

        void set_spatial_threads(int num_threads) {{
        #if defined(_OPENMP)
            omp_set_num_threads(num_threads);
        #endif
        }}

        void evaluate_residual(const double* y, const double* ydot, const double* p, const double* m, double* res) {{
        {body}
        }}

        void evaluate_observables(const double* y, const double* ydot, const double* p, const double* m, double* obs) {{
        {obs_body}
        }}

        void evaluate_jacobian(const double* y, const double* ydot, const double* p, const double* m, double c_j, double* jac_out) {{
            int N = {n_states};
            std::vector<double> dy(N, 0.0);
            std::vector<double> dydot(N, 0.0);
            std::vector<double> res_dummy(N, 0.0);
            std::vector<double> dres(N, 0.0);

        {textwrap.indent(jacobian_logic, '    ')}
        }}

        void evaluate_jvp(const double* y, const double* ydot, const double* p, const double* m, double c_j, const double* v, double* jvp_out) {{
            int N = {n_states};
            std::vector<double> dy(N, 0.0);
            std::vector<double> dydot(N, 0.0);
            std::vector<double> res_dummy(N, 0.0);

            for (int i = 0; i < N; ++i) {{
                dy[i] = v[i];
                dydot[i] = c_j * v[i];
                jvp_out[i] = 0.0;
            }}

        #ifdef ENZYME_ACTIVE
            __enzyme_fwddiff((void*)evaluate_residual, 
                enzyme_dup, y, dy.data(), 
                enzyme_dup, ydot, dydot.data(), 
                enzyme_const, p, 
                enzyme_const, m,
                enzyme_dup, res_dummy.data(), jvp_out);
        #endif
        }}

        void evaluate_vjp(const double* y, const double* ydot, const double* p, const double* m, const double* lambda_vec, double* dp_out, double* dy_out, double* dydot_out) {{
            int N = {n_states};
            int N_P = {n_params};
            std::vector<double> res_dummy(N, 0.0);
            
            for(int i=0; i<N_P; ++i) dp_out[i] = 0.0;
            for(int i=0; i<N; ++i) {{ dy_out[i] = 0.0; dydot_out[i] = 0.0; }}

        #ifdef ENZYME_ACTIVE
            __enzyme_autodiff((void*)evaluate_residual, 
                enzyme_dup, y, dy_out, 
                enzyme_dup, ydot, dydot_out, 
                enzyme_dup, p, dp_out, 
                enzyme_const, m,
                enzyme_dup, res_dummy.data(), (double*)lambda_vec);
        #endif
        }}
        }} // extern "C"
        """)