import yaml
import os
from pathlib import Path

def get_registry_provenance() -> dict[str, str]:
    """
    Load the parameter registry and extract a flattened dictionary of citations.
    Returns a mapping of logical parameter names to their source justification.
    """
    registry_path = Path(__file__).parent / "parameters.yaml"
    
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)
        
    provenance = {}
    
    # Flatten species parameters
    for species_name, params in registry.get("species", {}).items():
        for param_name, data in params.items():
            if "source" in data:
                provenance[f"{species_name}_{param_name}"] = data["source"]
                
    # Flatten geometry parameters
    for case_name, params in registry.get("geometry", {}).items():
        for param_name, data in params.items():
            if "source" in data:
                provenance[f"{param_name}"] = data["source"]
                
    # Flatten fluid parameters
    for fluid_name, params in registry.get("fluid", {}).items():
        for param_name, data in params.items():
            if "source" in data:
                provenance[f"{fluid_name}_{param_name}"] = data["source"]
                
    # Flatten gravity parameters
    for param_name, data in registry.get("gravity", {}).items():
        if "source" in data:
            provenance[f"{param_name}"] = data["source"]
            
    # Flatten mass_transfer parameters
    for param_group, params in registry.get("mass_transfer", {}).items():
        if param_group == "membrane_permeability_1g":
            for species_name, data in params.items():
                if "source" in data:
                    provenance[f"membrane_permeability_1g_{species_name}"] = data["source"]
        
    return provenance
