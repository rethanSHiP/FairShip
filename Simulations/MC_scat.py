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
    # Using general PDG high-Z/low-Z formulation
    x0_inv = Rad_const * (Z**2 * (L_rad(Z) - f_z(Z)) + Z * L_rad_prime(Z)) / A
    return 1.0 / x0_inv

def rad_length_mix(A1, Z1, A2, Z2, x1, x2):
    """Calculates effective radiation length (X0) for a composite/mixed path."""
    x1_inv = Rad_const * (Z1**2 * (L_rad(Z1) - f_z(Z1)) + Z1 * L_rad_prime(Z1)) / A1
    x2_inv = Rad_const * (Z2**2 * (L_rad(Z2) - f_z(Z2)) + Z2 * L_rad_prime(Z2)) / A2

    total_x = x1 + x2
    w1, w2 = x1 / total_x, x2 / total_x
    
    # PDG Harmonic mean rule for mixtures
    inv_x0_eff = w1 * x1_inv + w2 * x2_inv
    return 1.0 / inv_x0_eff

def MC_scattering(p, x_total, A=4.0026, m=105.66, z=1, Z=2, SBT=None, x_He=None, x_SBT=None, A_SBT=12.0, Z_SBT=6.0):
    """Returns the gaussian sigma of the angular distribution"""
    p = np.asarray(p)

    p_mev = p * 1e3
    e = np.sqrt(p_mev**2 + m**2)
    beta = p_mev / e

    if SBT is None:
        X0 = rad_length(A, Z)
        thickness = x_total / X0
    else: 
        # Mixed path (e.g., Helium + SBT)
        # Material 1: Helium (A, Z), thickness x_He
        # Material 2: SBT/Scintillator (A_SBT, Z_SBT), thickness x_SBT
        X0_eff = rad_length_mix(A1=A, Z1=Z, A2=A_SBT, Z2=Z_SBT, x1=x_He, x2=x_SBT)
        thickness = x_total / X0_eff # where x_total should be x_He + x_SBT

    prefactor = (13.6 / (beta * p_mev)) * z
    log_term = 1 + 0.038 * np.log(thickness * (z / beta)**2)
    return prefactor * np.sqrt(thickness) * log_term
    