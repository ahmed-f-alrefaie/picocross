from astropy import units as u
from astropy import constants as const
from attr import has
import numpy as np


def partition_function(energy, g_total, temperature):


    if not energy.shape == ():
        if len(energy) != len(g_total):
            raise ValueError('Energy and gtotal do not match in shape')

    if temperature <= 0 * u.K:
        raise ValueError(f'Temperature is not physical {temperature}')
    energy_fixed = energy.to(u.k, u.spectral())

    c2 = const.h*const.c/const.k_B

    partition = g_total * np.exp(-c2 * energy_fixed / temperature)

    return np.sum(partition)
