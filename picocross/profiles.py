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


def doppler_profile(
    bins: u.Quantity, transitions: u.Quantity, doppler_hwhm: u.Quantity
) -> u.Quantity:
    """Compute doppler profile using the form:

    .. math::
        f = g(x)

    Where :math:`g(x)` is the doppler profile.

    Args:
        bins: Frequency bins to compute the profile
        transitions: Transitions to compute the profile
        doppler_hwhm: Doppler HWHM (Compute from :func:`~doppler_hwhm`)

    Returns:
        Computed doppler profile

    """

    v_squared = (bins[None, :] - transitions[:, None]) ** 2

    ln2 = np.log(2)

    doppler_fixed = doppler_hwhm[:, None]

    left_side = np.sqrt(ln2 / np.pi) / doppler_fixed

    return left_side * np.exp(-v_squared * ln2 / doppler_fixed**2)
