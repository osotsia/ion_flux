"""
Pass 1: Semantic Resolver
Separates "What the math means" from "How it is executed".
Resolves Neumann boundary tags and ALE dynamic domains into a clean, queryable context.
"""
from typing import Dict, Any, Optional

class SemanticContext:
    def __init__(self, ast_payload: Dict[str, Any]):
        self.payload = ast_payload
        self.neumann_bcs: Dict[str, Dict[str, Any]] = {}
        self.dynamic_domains: Dict[str, Dict[str, Any]] = {}
        
        self._resolve_boundaries()

    def _resolve_boundaries(self):
        """Pre-processes the boundary bucket into O(1) lookup tables."""
        for bc in self.payload.get("boundaries", []):
            if bc["type"] == "neumann":
                self.neumann_bcs[bc["node_id"]] = bc["bcs"]
            elif bc["type"] == "moving_domain":
                d_name = bc["domain"]
                for side, rhs_ast in bc["bcs"].items():
                    # We store the math driving the boundary movement (Stefan problems)
                    self.dynamic_domains[d_name] = {"side": side, "rhs": rhs_ast}

    def get_neumann_bc(self, node_id: Optional[str], face: str) -> Optional[Dict[str, Any]]:
        """Returns the AST dict of the boundary condition if one applies to this face."""
        if not node_id: return None
        bcs = self.neumann_bcs.get(node_id)
        if bcs and face in bcs:
            return bcs[face]
        return None