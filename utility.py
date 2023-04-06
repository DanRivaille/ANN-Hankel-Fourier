# My Utility : auxiliars functions

import pandas as pd
import numpy  as np
  
#load parameters to train the SNN
def load_cnf():
  pass


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

