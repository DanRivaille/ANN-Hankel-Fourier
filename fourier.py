import numpy as np
from prep import load_data
import time


x = load_data(3)[0, 0][:20_000]
N = x.shape[0]
j = 1j

"""
for i in range(450):
    F = np.empty((N), dtype='complex_')
    n = np.arange(N)
    for k in range(N):
        F[k] = np.sum(x * np.exp(-j * (2 * np.pi / N) * k * n ))
    F = F / N
"""
initial_time_f = time.time()
F = np.empty((N), dtype='complex_')
n = np.arange(N)
for k in range(N):
    F[k] = np.sum(x * np.exp(-j * (2 * np.pi / N) * k * n ))
F = F
print(F[:5])
print(time.time() - initial_time_f)

def compute_fourier(X):
    return np.fft.fft(X)

initial_time = time.time()
print(compute_fourier(x)[:5])
print(time.time() - initial_time)



