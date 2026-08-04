import numpy as np

# Defining necessary constants
Na = 6.022e23 # mol-1
alpha = 1/137.035999139
re = 2.817940322 * 1e-13 # cm
Rad_const = 4 * alpha * (re**2) * Na

def x_mass(L, rho):
    """Calculates the mass per unit of area"""
    return np.asarray(L) * rho

def f_z(Z):
    """Calculates the Coulomb correction function."""
    a2 = (alpha * Z)**2
    poly = 0.20206 + a2 * (-0.0369 + a2 * (0.0083 - 0.002 * a2))
    return a2 * (1 / (1 + a2) + poly)

def L_rad(Z):
    return np.log(184.15 * Z**(-1/3))

def L_rad_prime(Z):
    return np.log(1194 * Z**(-2/3))

def rad_length(A, Z):
    """Calculates radiation length (X0) for a single material."""
    x0_inv = Rad_const * (Z**2 * (L_rad(Z) - f_z(Z)) + Z * L_rad_prime(Z)) / A
    return 1.0 / x0_inv

def rad_length_mix(A_list, Z_list, x_list):
    """
    Calculates effective radiation length (X0) for a composite path of N materials.
    A_list: list of atomic masses
    Z_list: list of atomic numbers
    x_list: list of traversed mass thicknesses for each material array
    """
    total_x = np.sum(x_list, axis=0)
    inv_x0_eff = np.zeros_like(total_x, dtype=float)

    for A, Z, x in zip(A_list, Z_list, x_list):
        # Calculate inverse radiation length for this specific material
        x_inv = Rad_const * (Z**2 * (L_rad(Z) - f_z(Z)) + Z * L_rad_prime(Z)) / A
        
        # Weight by mass fraction (using safe division to avoid NaN if total_x is 0)
        w = np.divide(x, total_x, out=np.zeros_like(x, dtype=float), where=(total_x != 0))
        inv_x0_eff += w * x_inv

    # Return effective X0 (safe division) and the total thickness
    X0_eff = np.divide(1.0, inv_x0_eff, out=np.full_like(total_x, np.inf), where=(inv_x0_eff != 0))
    return X0_eff, total_x

def MC_scattering(p, x_list, A_list, Z_list, m=105.66, z=1):
    """
    Returns the gaussian sigma of the angular distribution.
    Accepts arbitrary numbers of materials via lists.
    """
    p = np.asarray(p)
    p_mev = p * 1e3
    e = np.sqrt(p_mev**2 + m**2)
    beta = p_mev / e

    # Calculate effective X0 and total thickness dynamically
    X0_eff, total_x = rad_length_mix(A_list, Z_list, x_list)
    
    # Safe division for thickness in radiation lengths
    thickness = np.divide(total_x, X0_eff, out=np.zeros_like(total_x, dtype=float), where=(X0_eff != np.inf))

    # Highland formula
    prefactor = (13.6 / (beta * p_mev)) * z
    
    # Avoid log(0) warnings for muons that didn't hit any material
    safe_thickness = np.where(thickness <= 0, 1e-9, thickness)
    log_term = 1 + 0.038 * np.log(safe_thickness * (z / beta)**2)
    
    # Zero out scattering for particles with zero thickness
    sigma = prefactor * np.sqrt(safe_thickness) * log_term
    return np.where(thickness > 0, sigma, 0.0)
    