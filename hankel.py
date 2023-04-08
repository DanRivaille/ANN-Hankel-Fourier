import numpy as np

def extract_component(H):
  first_row = H[0]
  last_colu = H[:, -1][1:]
  component = np.concatenate((first_row, last_colu))
  return component


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
  print(H1, H2)
  
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


X = np.array([3.5186, 3.2710, 1.0429, 2.3774, 0.0901, 1.7010, 1.2509, 0.6459])

L = 3
H = create_hankel_matrix(X, L)

X = np.array(list(range(1, 9)), dtype=float)

L = 2
H = create_hankel_matrix(X, 2)

C = []
S_c = []
max_level = 1
initial_level = 1
compute_next_level(H, max_level, initial_level, C, S_c)

print(H)

print(len(C))
print(C)
