from typing import Dict, Any

def validate_ast(ast_payload: Dict[str, Any]) -> bool:
    """
    Validates the structure of a generated semantic AST before dispatching to the compiler.
    Ensures structural integrity to prevent fatal unwraps in the Rust backend.
    """
    if not isinstance(ast_payload, dict):
        raise TypeError("AST payload must be a dictionary of semantic equation buckets.")
        
    valid_buckets = {"regions", "global", "boundaries"}
    
    for bucket, content in ast_payload.items():
        if bucket not in valid_buckets:
            raise ValueError(f"Invalid equation bucket: '{bucket}'. Allowed buckets: {valid_buckets}")
            
        if bucket == "regions":
            if not isinstance(content, dict):
                raise ValueError("'regions' must be a dictionary mapping domains to lists of equations.")
            for dom, eqs in content.items():
                for eq in eqs:
                    if "lhs" not in eq or "rhs" not in eq:
                        raise ValueError(f"Every AST equation in 'regions' must contain a 'lhs' and 'rhs'.")
                    if "type" not in eq["lhs"] or "type" not in eq["rhs"]:
                        raise ValueError(f"All AST nodes in 'regions' must declare a 'type'.")
        else:
            for eq in content:
                if "lhs" not in eq or "rhs" not in eq:
                    raise ValueError(f"Every AST equation in '{bucket}' must contain a 'lhs' and 'rhs'.")
                if "type" not in eq["lhs"] or "type" not in eq["rhs"]:
                    raise ValueError(f"All AST nodes in '{bucket}' must declare a 'type'.")
            
    return True