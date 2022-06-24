import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u

from picocross.profiles import doppler_hwhm


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


def test_doppler_profile():
    from picocross.profiles import doppler_profile

    trans = np.linspace(100, 1000, 10) * u.k

    doppler_hwhm = np.linspace(2, 10, 10) / u.cm

    bins = np.linspace(300, 600, 20) * u.k

    result = doppler_profile(bins, trans, doppler_hwhm)

    assert result.shape == (10, 20)

    result.to(u.cm)
