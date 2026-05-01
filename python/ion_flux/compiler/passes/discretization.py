from typing import Any
from .ir import Expr, Literal, BinaryOp, FuncCall

class Discretizer:
    """
    Provides pure functional mathematical discretization for the Finite Volume Method (FVM).
    Isolates coordinate system geometry (Cartesian, Cylindrical, Spherical) from AST traversal.
    """

    @staticmethod
    def divergence_normalized(r_flux: Expr, l_flux: Expr, A_R: Expr, A_L: Expr, V_i: Expr, L_phys: Expr) -> Expr:
        V_scaled = BinaryOp("*", V_i, L_phys)
        V_safe = FuncCall("std::max", [Literal("1e-30"), V_scaled])
        net_flux = BinaryOp("-", BinaryOp("*", A_R, r_flux), BinaryOp("*", A_L, l_flux))
        return BinaryOp("/", net_flux, V_safe)

    @staticmethod
    def unstructured_divergence_code(rp: int, ci: int, w: int, s_off: int, idx_cpp: str) -> str:
        return (
            f"[&]() {{\n    double sum = 0.0;\n"
            f"    for(int k = (int)m[{rp} + {idx_cpp}]; k < (int)m[{rp} + {idx_cpp} + 1]; ++k) {{\n"
            f"        sum += m[{w} + k] * (y[{s_off} + (int)m[{ci} + k]] - y[{s_off} + {idx_cpp}]);\n"
            f"    }}\n    return sum;\n}}()"
        )

    @staticmethod
    def integral_volume_code_normalized(coord_sys: str, int_var: str, start: int, b_axis: str, layout: Any = None) -> str:
        if coord_sys == "unstructured":
            if layout and b_axis in layout.mesh_offsets and "volumes" in layout.mesh_offsets[b_axis]:
                vol_off = layout.mesh_offsets[b_axis]["volumes"]
                return f"        vol *= m[{vol_off} + {int_var}];\n"
            else:
                return f"        vol *= 1.0;\n"
                
        dim_exp = 3.0 if coord_sys == "spherical" else (2.0 if coord_sys == "cylindrical" else 1.0)
        vol_off = layout.mesh_offsets[b_axis]["w_V_nodes"]
        return (
            f"        double L_scale_{b_axis} = std::pow(L_phys_{b_axis}, {dim_exp});\n"
            f"        vol *= m[{vol_off} + {start} + {int_var}] * L_scale_{b_axis};\n"
        )

    @staticmethod
    def ale_dimension_multiplier(coord_sys: str) -> float:
        if coord_sys == "spherical": return 3.0
        if coord_sys == "cylindrical": return 2.0
        return 1.0