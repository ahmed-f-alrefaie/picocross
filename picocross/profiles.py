import numpy as np
from astropy import constants as const
from astropy import units as u


def doppler_hwhm(
    transition: u.Quantity, temperature: u.Quantity, molecular_mass: u.Quantity
) -> u.Quantity:
    """Compute doppler HWHM.

    .. math::

        \\alpha_D = \\sqrt{\\frac{2k_BT \ln 2}{m}}\\frac{v_{if}}{c}

    Args:
        transition: Transition frequencies
        temperature: Temperature to compute HWHW
        molecular_mass: Mass per mole of molecule

    """

    transition_fixed = transition
    if transition.unit != u.k:
        transition_fixed = transition.to(u.k, u.spectral())

    left_factor = np.sqrt(2 * const.k_B * np.log(2) * temperature / molecular_mass)

    return left_factor * transition_fixed / const.c
