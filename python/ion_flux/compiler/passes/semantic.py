from typing import Dict, Any, Optional

class SemanticContext:
    def __init__(self, ast_payload: Dict[str, Any]):
        self.payload = ast_payload
        self.neumann_bcs: Dict[str, Dict[str, Any]] = {}
        self.dynamic_domains: Dict[str, Dict[str, Any]] = {}
        self.dirichlet_bcs: Dict[str, Dict[str, Any]] = {}
        
        self._resolve_boundaries()

    def _resolve_boundaries(self):
        """Pre-processes the boundary bucket into O(1) lookup tables."""
        for bc in self.payload.get("boundaries", []):
            if bc["type"] == "neumann":
                node_id = bc["node_id"]
                if node_id not in self.neumann_bcs:
                    self.neumann_bcs[node_id] = {}
                for side, ast_node in bc["bcs"].items():
                    self.neumann_bcs[node_id][side] = {"ast": ast_node, "domain": bc.get("domain")}
            elif bc["type"] == "moving_domain":
                d_name = bc["domain"]
                for side, rhs_ast in bc["bcs"].items():
                    self.dynamic_domains[d_name] = {"side": side, "rhs": rhs_ast}
            elif bc["type"] == "dirichlet":
                self.dirichlet_bcs[bc["state"]] = bc["bcs"]

    def get_neumann_bc(self, node_id: Optional[str], face: str) -> Optional[Dict[str, Any]]:
        """Returns the AST dict and domain info of the boundary condition if one applies to this face."""
        if not node_id: return None
        bcs = self.neumann_bcs.get(node_id)
        if bcs and face in bcs:
            return bcs[face]
        return None
        
    def get_dirichlet_bc(self, state_name: str) -> Optional[Dict[str, Any]]:
        return self.dirichlet_bcs.get(state_name)