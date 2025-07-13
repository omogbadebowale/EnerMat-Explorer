def vegard_bowing(prop_A, prop_B, frac_B, bow):
    """
    Quadratic Vegard + bowing interpolation.
    Works for Eg, Eox_e, Ehull… any scalar.
    """
    return prop_A * (1 - frac_B) + prop_B * frac_B - bow * frac_B * (1 - frac_B)
