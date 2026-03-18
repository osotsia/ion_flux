def Chen2020() -> dict:
    """Returns a flat dictionary of the Chen2020 parameter set."""
    return {
        "neg_elec.porosity": 0.33,
        "pos_elec.porosity": 0.32,
        "electrolyte.initial_concentration": 1000.0,
    }

def Marquis2019() -> dict:
    """Returns a flat dictionary of the Marquis2019 parameter set."""
    return {
        "neg_elec.porosity": 0.3,
        "pos_elec.porosity": 0.3,
        "electrolyte.initial_concentration": 1000.0,
    }