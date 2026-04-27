from typing import Any
from .ir import Expr, Literal, BinaryOp, FuncCall

class Discretizer:
    """
    Provides pure functional mathematical discretization for the Finite Volume Method (FVM).
    Isolates coordinate system geometry (Cartesian, Cylindrical, Spherical) from AST traversal.
    """

    @staticmethod
    def cell_geometry(coord_sys: str, i_R: Expr, i_L: Expr, dx_ir: Expr) -> tuple[Expr, Expr, Expr]:
        if coord_sys == "spherical":
            A_R = BinaryOp("*", i_R, BinaryOp("*", i_R, BinaryOp("*", dx_ir, dx_ir)))
            A_L = BinaryOp("*", i_L, BinaryOp("*", i_L, BinaryOp("*", dx_ir, dx_ir)))
            V = BinaryOp("/", BinaryOp("*", BinaryOp("-", BinaryOp("*", A_R, i_R), BinaryOp("*", A_L, i_L)), dx_ir), Literal(3.0))
            
        elif coord_sys == "cylindrical":
            A_R = BinaryOp("*", i_R, dx_ir)
            A_L = BinaryOp("*", i_L, dx_ir)
            V = BinaryOp("/", BinaryOp("*", BinaryOp("-", BinaryOp("*", A_R, i_R), BinaryOp("*", A_L, i_L)), dx_ir), Literal(2.0))
            
        else:
            A_R = Literal(1.0)
            A_L = Literal(1.0)
            V = BinaryOp("*", BinaryOp("-", i_R, i_L), dx_ir)
            
        return A_R, A_L, V

    @staticmethod
    def divergence(r_flux: Expr, l_flux: Expr, A_R: Expr, A_L: Expr, V: Expr) -> Expr:
        V_safe = FuncCall("std::max", [Literal("1e-30"), V])
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
    def integral_volume_code(coord_sys: str, int_var: str, res: int, b_axis: str, layout: Any = None) -> str:
        if coord_sys == "spherical":
            return (
                f"        double r_R = ({int_var} == {res}-1) ? ({int_var} * dx_{b_axis}) : ({int_var} * dx_{b_axis} + 0.5 * dx_{b_axis});\n"
                f"        double r_L = ({int_var} == 0) ? 0.0 : ({int_var} * dx_{b_axis} - 0.5 * dx_{b_axis});\n"
                f"        vol *= (4.0/3.0) * 3.141592653589793 * (std::pow(r_R, 3.0) - std::pow(r_L, 3.0));\n"
            )
        elif coord_sys == "unstructured":
            if layout and b_axis in layout.mesh_offsets and "volumes" in layout.mesh_offsets[b_axis]:
                vol_off = layout.mesh_offsets[b_axis]["volumes"]
                return f"        vol *= m[{vol_off} + {int_var}];\n"
            else:
                return f"        vol *= 1.0;\n"
        else:
            return f"        vol *= ({int_var} == 0 || {int_var} == {res}-1) ? 0.5 * dx_{b_axis} : dx_{b_axis};\n"

    @staticmethod
    def ale_dimension_multiplier(coord_sys: str) -> float:
        if coord_sys == "spherical": return 3.0
        if coord_sys == "cylindrical": return 2.0
        return 1.0