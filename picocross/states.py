import typing as t

import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u
from attr import has


def partition_function(
    energy: u.Quantity,
    g_total: t.Union[
        u.Quantity,
        np.ndarray,
    ],
    temperature: u.Quantity,
) -> u.Quantity:
    """Computes partition function for a given temperature.

    The partition function is computed using the equation.

    .. math::
        Q(T) = \\sum_i g_i e^{-c_2 E_i / T}

    You can use it like this:

    >>> E = np.linspace(0, 100, 100) * u.k
    >>> g_total = np.ones(100)
    >>> temperature = 1000 * u.K
    >>> Q = partition_function(E, g_total, temperature)



    Args:
        energy      : Energy of states
        g_total     : degeneracy of states
        temperature : Temperature to compute partion function.

    Returns:
        Quantity: Partition function at temperature.

    """
    if not energy.shape == ():
        if len(energy) != len(g_total):
            raise ValueError("Energy and gtotal do not match in shape")

    if temperature <= 0 * u.K:
        raise ValueError(f"Temperature is not physical {temperature}")
    energy_fixed = energy.to(u.k, u.spectral())

    c2 = const.h * const.c / const.k_B

    partition = g_total * np.exp(-c2 * energy_fixed / temperature)

    return np.sum(partition)


def load_exomol_states(filename: str) -> pd.DataFrame:
    return pd.read_csv(
        filename,
        delim_whitespace=True,
        usecols=[0, 1, 2, 3],
        names=["ID", "Energy", "g_total", "J"],
    )


class ExomolStates:
    """Represents a state for a single linelist"""

    def __init__(self, filename: str):
        """Load in a state file.

        Args:
            filename: filename to `.state` file

        """
        self.df = load_exomol_states(filename)

    def Q(self, temperature: u.K) -> u.Quantity:
        return partition_function(self.energy, self.degeneracy, temperature)

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.df

    @property
    def energy(self) -> u.Quantity:
        return self.df["Energy"].values << u.k

    @property
    def degeneracy(self):
        return self.df["g_total"].values
