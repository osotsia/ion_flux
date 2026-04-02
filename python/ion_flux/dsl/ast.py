from typing import Dict, Any

def validate_ast(ast_payload: Dict[str, Any]) -> bool:
    """
    Validates the structure of a generated semantic AST before dispatching to the compiler.
    Ensures structural integrity to prevent fatal unwraps downstream.
    """
    if not isinstance(ast_payload, dict):
        raise TypeError("AST payload must be a dictionary.")
        
    valid_buckets = {"equations", "boundaries", "initial_conditions", "domains"}
    
    for bucket, content in ast_payload.items():
        if bucket not in valid_buckets:
            raise ValueError(f"Invalid equation bucket: '{bucket}'. Allowed buckets: {valid_buckets}")
            
        if bucket == "equations":
            for eq in content:
                if "state" not in eq or "type" not in eq:
                    raise ValueError("Equations must explicitly define 'state' and 'type'.")
                if eq["type"] == "piecewise" and "regions" not in eq:
                    raise ValueError("Piecewise equations must declare a list of 'regions'.")
                if eq["type"] == "standard" and "eq" not in eq:
                    raise ValueError("Standard equations must contain an 'eq' declaration.")
                    
        elif bucket == "boundaries":
            for bc in content:
                if "type" not in bc or "bcs" not in bc:
                    raise ValueError("Boundaries must explicitly define 'type' and 'bcs'.")
                    
        elif bucket == "initial_conditions":
            for ic in content:
                if "state" not in ic or "value" not in ic:
                    raise ValueError("Initial conditions must define 'state' and 'value'.")
                    
    return True