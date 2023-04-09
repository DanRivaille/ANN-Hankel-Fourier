import numpy as np

def extract_dyadic_component(H):
  a = np.concatenate((H[0], H[1, -1:]))
  b = np.concatenate((H[0, :1], H[1]))
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
  Hi = S[i] * (U[:, i].reshape(-1, 1) @ Vt[i, :].reshape(1, -1))
  return Hi


def decompose_matrix(U, S, Vt):
  H1 = get_H(U, S, Vt, 0)
  H2 = get_H(U, S, Vt, 1)
  return H1, H2


def svd(H):
  U, S, Vt = np.linalg.svd(H, full_matrices=False)
  return U, S, Vt


def compute_next_level(H, max_level, current_level, C, S_components):
  U, S, Vt = svd(H)
  H1, H2 = decompose_matrix(U, S, Vt)
  
  if max_level == current_level:
    C1 = extract_dyadic_component(H1)
    C2 = extract_dyadic_component(H2)

    C.append(C1)
    C.append(C2)
    S_components.append(S[0])
    S_components.append(S[1])
  else:
    compute_next_level(H1, max_level, current_level + 1, C, S_components)
    compute_next_level(H2, max_level, current_level + 1, C, S_components)

def compute_singular_decomposition(frame, j):
  C = []
  Sc = []
  H_matrix = create_hankel_matrix(frame, 2)
  compute_next_level(H_matrix, j, 1, C, Sc)

  return C, Sc
