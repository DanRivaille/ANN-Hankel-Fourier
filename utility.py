# My Utility : auxiliars functions

import numpy  as np
  
#load parameters to train the SNN
def load_cnf():
  pass


def get_one_hot(y):
  K = np.unique(y).shape[0]
  res = np.eye(K)[(y-1).reshape(-1)]
  return res.reshape(list(y.shape)+[K]).astype(int)


# Load data from filename
def load_data(file_x, file_y):
  X = np.loadtxt(file_x, delimiter=',', dtype=float)
  y = get_one_hot(np.loadtxt(file_y, dtype=int))
  return X, y


# Initialize weights for SNN-SGDM
def iniWs(param):    
  pass


# Initialize weights for one-layer    
def iniW(next, prev):
  r = np.sqrt(6 / (next + prev))
  w = np.random.rand(next, prev)
  w = w * 2 * r - r    
  return w

# Feed-forward of SNN
def forward():
  pass


#Activation function
def act_function():
  pass


# Derivatives of the activation funciton
def deriva_act():
  pass


#Feed-Backward of SNN
def gradW():
  pass


# Update W and V
def updWV_sgdm():
  pass


# Measure
def metricas(x, y):
  pass
    

#Confusion matrix
def confusion_matrix(z, y):
  pass

