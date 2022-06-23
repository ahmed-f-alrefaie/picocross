from multiprocessing.sharedctypes import Value
import pytest
from astropy import units as u
import numpy as np


@pytest.fixture
def statefile(tmp_path_factory):
    fn = tmp_path_factory.mktemp('data') / 'test.states'
    fn.write_text(
"""           1     0.000000      1       0         Inf +  1          1 p A1  0  0  0  0  0
           2  1594.873096      1       0  4.1203e-02 +  1          2 p A1  0  1  0  0  0
           3  3151.677108      1       0  2.0601e-02 +  1          3 p A1  0  2  0  0  0
           4  3657.155752      1       0  1.4963e-01 +  1          4 p A1  1  0  0  0  0
           5  4666.724999      1       0  1.3749e-02 +  1          5 p A1  0  3  0  0  0
"""


    )

    return fn


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



def test_load_exomol_states(statefile):
    from picocross.states import load_exomol_states

    df = load_exomol_states(statefile)

    assert len(df.columns) == 4
    assert 'ID' in df.columns
    assert 'Energy' in df.columns
    assert 'g_total' in df.columns
    assert 'J' in df.columns
    assert len(df) == 5


    assert df['Energy'][0] == 0.0
    assert df['Energy'][2] == 3151.677108
    assert df['Energy'][4] == 4666.724999






