import numpy as np
import matplotlib.pyplot as plt
import time
import numba as nb

nb.config.NUMBA_DEFAULT_NUM_THREADS = 8

@nb.njit(fastmath=True)
def mu(x: float) -> float:
    return -(x ** 3)
    
@nb.njit(fastmath=True)
def SSA_cubic(
    low: float,
    span: float,
    n: int,
    sample_size: int,
) -> np.ndarray:
    """
    Stochastic Simulation Algorithm for cubic potential function

    Parameters:
    ----------  
    - low : float
        The lower bound of the spatial domain.
    - span : float
        The length of the spatial domain.
    - n : int
        The number of spatial points.
    - sample_size : int
        The number of samples to generate.
    - sample_loops : int
        The number of times to repeat the sampling.
    Returns:
    -------
    np.array
        The generated samples trajectories and density estimations.
    2D array of shape (sample_loops, sample_size)
    ------------------------------------------------------------------------------
    1. Initialize parameters and arrays.
    2. For each sampling loop:
        a. For each sample:
            i. Generate a random initial position within the spatial domain.
            ii. Determine the corresponding spatial bin.
            iii. Simulate the stochastic process until the desired number of valid samples is reached.
            iv. Record the sample position.
    3. Return the array of samples.
    ------------------------------------------------------------------------------
    2D array of shape (sample_loops, sample_size)
        The generated samples trajectories and density estimations.
    """

    density = np.zeros(n, dtype=np.float64)
    h = span / n
    x0 = low + h / 2.0 + np.random.random() * span
    m = max(2 - abs(mu(x0)) * h, 0.0) / 2.0
    valid_count = 0
    trajectory_point = low + h / 2.0 + round((x0 - low - h / 2.0) / h ) * h
    while valid_count < sample_size:
        q0 = max(mu(trajectory_point), 0.0) / h + m / ( h ** 2)
        q1 = - min( mu(trajectory_point), 0.0) / h + m / ( h ** 2)
        lam = q0 + q1
        r1 = np.random.random()
        r2 = np.random.random()
        tau = -np.log(1.0 - r2) / lam
        if q0 >= r1 * lam:
            trajectory_point += h
        else:
            trajectory_point -= h
        x_n = round((trajectory_point - low - h / 2.0) / h)
        if 0 <= x_n < n:
            density[x_n] += 1
            valid_count += 1
        else:
            raise ValueError("Trajectory point out of bounds.")
    density /= (sample_size * h)
    return density


@nb.njit(fastmath=True, parallel=True)
def run_ensemble(low: float, span: float, n: int, sample_size: int, sample_loops: int) -> np.ndarray:
    """
    Run multiple independent SSA simulations in parallel and return the mean density.

    Parameters:
    ----------
    - low : float
        The lower bound of the spatial domain.
    - span : float
        The length of the spatial domain.
    - n : int
        The number of spatial points.
    - sample_size : int
        The number of samples per trajectory.
    - sample_loops : int
        The number of independent trajectories to average.

    Returns:
    -------
    np.ndarray
        Mean density estimate over `sample_loops` simulations.
    """
    density_ensemble = np.zeros((sample_loops, n), dtype=np.float64)
    for loop in nb.prange(sample_loops):
        density_ensemble[loop, :] = SSA_cubic(low, span, n, sample_size)
    density_mean = np.zeros(n, dtype=np.float64)
    for j in range(n):
        s = 0.0
        for i in range(sample_loops):
            s += density_ensemble[i, j]
        density_mean[j] = s / sample_loops
    return density_mean


def main():
    low = -3.0
    span = 6.0
    sample_size = int(1e7)
    sample_loops = 5
    n = 120
    start_time = time.perf_counter()
    density_mean = run_ensemble(low, span, n, sample_size, sample_loops)
    end_time = time.perf_counter()
    print(f"Execution Time: {end_time - start_time:.4f} seconds")   
    x = np.linspace(low + span / (2 * n), low + span - span / (2 * n), n)
    plt.plot(x, density_mean, label='SSA Density')
    plt.xlabel('Position')
    plt.ylabel('Density')
    plt.title('Density Estimation using SSA')
    plt.legend()
    plt.show()      

if __name__ == "__main__":
    main()
