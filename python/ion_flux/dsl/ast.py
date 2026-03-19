from typing import Dict, Any, List

def validate_ast(ast_payload: List[Dict[str, Any]]) -> bool:
    """
    Validates the structure of a generated AST before dispatching to the Rust compiler.
    Ensures structural integrity to prevent fatal unwraps in the Rust backend.
    """
    if not isinstance(ast_payload, list):
        raise TypeError("AST payload must be a list of equations.")
        
    for eq in ast_payload:
        if "lhs" not in eq or "rhs" not in eq:
            raise ValueError("Every AST equation must contain a 'lhs' and 'rhs'.")
            
        if "type" not in eq["lhs"] or "type" not in eq["rhs"]:
            raise ValueError("All AST nodes must declare a 'type'.")
            
    return True
