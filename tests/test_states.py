from multiprocessing.sharedctypes import Value
import pytest
from astropy import units as u
import numpy as np

def test_partition_function():
    from picocross.states import partition_function

    result = partition_function(100 * u.k, 10.0, 1000 * u.K)

    assert result.unit == u.dimensionless_unscaled

    E = np.linspace(0, 1000, 10) * u.k
    g = np.linspace(0, 100, 10)

    result = partition_function(E, g, 1000* u.K)

    assert result.unit == u.dimensionless_unscaled


def test_partition_exception():
    from picocross.states import partition_function

    with pytest.raises(ValueError):
        E = np.linspace(0, 1000, 9) * u.k
        g = np.linspace(0, 100, 10)
        partition_function(E, g, 1000* u.K)

    with pytest.raises(ValueError):
        partition_function(100 * u.k, 10, 0 * u.K)

@pytest.mark.parametrize(
    "energy",
    [
        1000*u.k,
        1000/u.cm,
        1000*u.Hz,
        1000*u.um,
        1000*u.m,
    ]
)
def test_partition_units(energy):
    from picocross.states import partition_function

    assert partition_function(energy, 10.0, 1000* u.K).unit == u.dimensionless_unscaled







