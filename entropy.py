import numpy as np

# E(x)
def calculate_entropy(x, N, Ix):
    interval_range, x_min, x_max = calculate_range(x, Ix)

    cross_entropy = 0
    for i in range(Ix):
        lower_bound = x_min + interval_range * i
        upper_bound = lower_bound + interval_range

        # Se obtienen la cantidad de muestras dentro del intervalo actual
        ni = np.where(np.logical_and(x >= lower_bound, x < upper_bound))[0].shape[0]

        if ni != 0:
          pi = ni / N

          entropy = pi * np.log2(pi)
          cross_entropy += entropy

    return -cross_entropy

def calculate_range(x, Ix, x_min=0.01, x_max=1):
    range = (x_max - x_min) / Ix
    return range, x_min, x_max

