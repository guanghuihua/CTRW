import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb

nb.config.NUMBA_DEFAULT_NUM_THREADS = 8

@nb.njit(fastmath = True)