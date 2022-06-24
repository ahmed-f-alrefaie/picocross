import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u


@pytest.mark.parametrize(
    "unit",
    [
        u.k,
        u.um,
        u.Hz,
    ],
)
def test_doppler_hwhm(unit):
    from picocross.profiles import doppler_hwhm

    trans = np.linspace(100, 1000, 10) * unit

    mass = 12 * u.g / u.mol / const.N_A

    doppler_hwhm(trans, 1000 * u.K, mass).to(1 / u.m)
