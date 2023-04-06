# SNN's Training :

import numpy      as np
import utility    as ut

#Save weights and MSE  of the SNN
def save_w_mse():
  pass


#gets Index for n-th miniBatch
def get_Idx_n_Batch(n, x, N):
  pass


#miniBatch-SGDM's Training 
def trn_minibatch(x, y, param):    
  W,V   = iniWs()
  pass


#SNN's Training 
def train(x, y, param):    
  W, V = iniWs()
  pass
  #return(W, Costo)


# Load data to train the SNN
def load_data_trn():
  pass


# Beginning ...
def main():
  param = ut.load_cnf()            
  xe, ye = load_data_trn()   
  W, Cost = train(xe, ye, param)             
  save_w_cost(W, Cost)


if __name__ == '__main__':   
	 main()

