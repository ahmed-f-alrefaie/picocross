import pandas as pd
from .states import ExomolStates
from astropy import units as u
import typing as t
import numpy as np
from astropy import constants as const

def load_exomol_transition(filename: str, chunksize: bool = None) -> pd.DataFrame:
    """Load in Exomol transition file
    
    Args:
        filename: Path to .trans file
        chunksize: How many transitions to read at a time. If None then read everything.
    
    Returns:
        Dataframe or TextFileRead: dataframe containing transitions.
            Is a TextFileReader if chunksize is not None
    
    """
    return pd.read_csv(filename, 
                       delim_whitespace=True, 
                       usecols=[0, 1, 2], 
                       names=['upper', 'lower', 'Afi'], chunksize=chunksize)


def combine_states_transitions(states: ExomolStates, 
            transitions: pd.DataFrame) -> t.Union[
                u.Quantity,
                u.Quantity,
                u.Quantity,
                np.ndarray
            ]:
    new_df = transitions.merge(
        states.dataframe, 
        how='left', 
        left_on='upper', 
        right_on='ID'
    ).merge(states.dataframe, 
            how='left',
            left_on='lower',
            right_on='ID', suffixes=('_upper', '_lower'))
    g_f = new_df['g_total_upper'].values
    energy_lower = new_df['Energy_lower'].values
    afi = new_df['Afi'].values
    energy_upper = new_df['Energy_upper'].values
    
    transitions = energy_upper - energy_lower

    return (
        transitions << u.k,
        afi << 1/u.s,
        energy_lower << u.k,
        g_f,
    )

def intensity(trans_freq: u.Quantity, Afi: u.Quantity,
              Ei: u.Quantity, g_f: np.ndarray, 
              temperature: u.Quantity,
              Q: u.Quantity) -> u.Quantity:

    c2 = const.h * const.c / const.k_B

    trans_exp = (1 - np.exp(-c2 * trans_freq / temperature))
    energy_exp = np.exp(-c2*Ei/temperature)

    left_factor = g_f * Afi / (8 * np.pi * const.c * trans_freq ** 2)

    return (left_factor * energy_exp * trans_exp / Q).to(u.cm)


