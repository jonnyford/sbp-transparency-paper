import numpy as np
from scipy.signal import hilbert

def envelope(a, n=100):
    """Return the envelope of an n-dimensional array in the first dimension.
    Pad with n samples to stop edge effects.
    """
    dims = len(a.shape)
    pad = ((n, n),) + ((0, 0),) * (dims - 1)
    return np.abs(hilbert(np.pad(a, pad, mode='edge'), axis=0))[n:-n]