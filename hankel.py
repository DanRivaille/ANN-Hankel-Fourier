import numpy as np

def extract_component(H):
  first_row = H[0]
  last_colu = H[:, -1][1:]
  component = np.concatenate((first_row, last_colu))
  return component

def extract_dyadic_component(H):
  a = np.concatenate((h[0], h[1, -1:]))
  b = np.concatenate((h[0, :1], h[1]))
  c = (a + b) / 2
  return c

def create_hankel_matrix(signal, L):
  N = signal.shape[0]
  K = N - L + 1
  hankel = np.empty((L, K))
  for i in range(L):
    hankel[i] = signal[i:i + K]

  return hankel

def get_H(U, S, Vt, i):
  Hi = S[i] * (U[:, i].reshape(-1, 1) @ Vt[:, i].reshape(1, -1))
  return Hi

X = np.array([3.5186, 3.2710, 1.0429, 2.3774, 0.0901, 1.7010, 1.2509, 0.6459])

L = 3
H = create_hankel_matrix(X, L)
print(X)
print(H)

u, s, vt = np.linalg.svd(H)
print(u)
print(np.diag(s))
print(vt.T)

print(u.shape)
print(np.diag(s).shape)
print(vt.T.shape)

print(get_H(u, s, vt, 0))
