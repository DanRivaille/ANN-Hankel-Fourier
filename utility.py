# My Utility : auxiliars functions

import numpy  as np
  
#load parameters to train the SNN
def load_cnf():
  FILE_CNF = 'cnf.csv'
  param = dict()
  cnf_list = np.loadtxt(FILE_CNF, dtype=float)

  param['n_classes'] = int(cnf_list[0])
  param['n_frame'] = int(cnf_list[1])
  param['l_frame'] = int(cnf_list[2])
  param['j_desc'] = int(cnf_list[3])

  hidden_nodes_layer_1 = int(cnf_list[4])
  hidden_nodes_layer_2 = int(cnf_list[5])

  param['g_fun'] = int(cnf_list[6])
  param['train_factor'] = cnf_list[7]
  param['M_batch'] = int(cnf_list[8])
  param['mu'] = cnf_list[9]
  param['beta'] = cnf_list[10]
  param['max_iter'] = int(cnf_list[11])

  if hidden_nodes_layer_2 != 0:
    param['hidden_nodes'] = [hidden_nodes_layer_1, hidden_nodes_layer_2]
  else:
    param['hidden_nodes'] = [hidden_nodes_layer_1]

  return param


def get_one_hot(y):
  K = np.unique(y).shape[0]
  res = np.eye(K)[(y-1).reshape(-1)]
  return res.reshape(list(y.shape)+[K]).astype(int)


# Load data from filename
def load_data(file_x, file_y):
  X = np.loadtxt(file_x, delimiter=',', dtype=float).T
  y = get_one_hot(np.loadtxt(file_y, dtype=int)).T
  return X, y


# Initialize weights for SNN-SGDM
def iniWs(W, L, d, m, n_nodes):    
  W[1] = iniW(n_nodes[0], d)
  W[L] = iniW(m, n_nodes[-1])

  for i in range(L - 2):
    W[i + 2] = iniW(n_nodes[i + 1], n_nodes[i])

  return W


# Initialize weights for one-layer    
def iniW(next, prev):
  r = np.sqrt(6 / (next + prev))
  w = np.random.rand(next, prev)
  w = w * 2 * r - r    
  return w

# Create a dictionary with the ann info
def create_ann(hidden_nodes):
  n_layers = len(hidden_nodes) + 1
  W = [None] * (n_layers + 1)   # Weights matrixes
  a = [None] * (n_layers + 1)   # activation matrixes
  z = [None] * (n_layers + 1)   # transfer matrixes

  ann = {'W': W, 'a': a, 'z': z, 'L': n_layers, 'hidden_nodes': hidden_nodes}
  return ann


# Feed-forward of SNN
def forward(ann, param, X):
  for l in range(1, ann['L'] + 1):
    if ann['L'] == l:
      num_funct = 5
    else:
      num_funct = param['g_fun']

    ann['z'][l] = np.matmaul(ann['W'][l], ann['a'][l - 1])
    ann['a'][l] = act_function(num_funct, X)

  return ann['a'][-1]


#Activation function
def act_function(num_function, x):
  if 1 == num_function:	# Relu
    return np.maximum(0, x)
  if 2 == num_function:	# L-Relu
    return np.maximum(0.01 * x, x)
  if 3 == num_function: 	# ELU
    return np.maximum(_alpha_elu * (np.exp(x) - 1), x)
  if 4 == num_function:	# SELU
    return np.maximum(_lambda * _alpha_selu * (np.exp(x) - 1), _lambda * x)
  if 5 == num_function:	# Sigmoide
    return sigmoid(x)
  else:
    return None

# Derivatives of the activation funciton
def deriva_act(num_function, x):
  if 1 == num_function:	# Relu
    return np.greater(x, 0).astype(float)
  if 2 == num_function:	# L-Relu
    return np.piecewise(x, [x <= 0, x > 0], [lambda e: 0.01, lambda e: 1])
  if 3 == num_function: 	# ELU
    return np.piecewise(x, [x <= 0, x > 0], [lambda e: 0.1 * np.exp(e), lambda e: 1])
  if 4 == num_function:	# SELU
    return np.piecewise(x, [x <= 0, x > 0], [lambda e: _lambda * _alpha_selu * np.exp(e), lambda e: _lambda])
  if 5 == num_function:	# Sigmoide
    return dev_sigmoid(x)
  else:
    return None

def sigmoid(x):
  f_x = 1 / (1 + np.exp(-x))
  return f_x

def dev_sigmoid(x):
  f_x = sigmoid(x)
  return f_x * (1 - f_x)


#Feed-Backward of SNN
def gradW():
  pass


# Update W and V
def updWV_sgdm():
  pass


# Measure
def metricas(a, y):
  pass
    

#Confusion matrix
def confusion_matrix(a, y):
  pass

