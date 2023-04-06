# SNN's Training :

import numpy      as np
import utility    as ut

#Save weights and MSE  of the SNN
def save_w_mse(W, ann_MSE):
  #np.savez('w_snn.npz', *W)
  #np.savetxt("costo.csv", np.array(ann_MSE))
  pass


#gets Index for n-th miniBatch
def get_Idx_n_Batch(n, x, N):
  pass


#miniBatch-SGDM's Training 
def trn_minibatch(x, y, ann, param, V):
  pass


#SNN's Training 
def train(x, y, param):
  ann = init_ann(param, x)

  return ann['W']#, Costo


def init_ann(param, x):
  ann = ut.create_ann(param['hidden_nodes'], x)
  d = x.shape[0]
  ann['W'] = ut.iniWs(ann['W'], ann['L'], d, param['n_classes'], param['hidden_nodes'])

  return ann


# Load data to train the SNN
def load_data_trn():
  FILE_X = 'dtrn.csv'
  FILE_Y = 'etrn.csv'
  X_train, y_train = ut.load_data(FILE_X, FILE_Y)
  return X_train, y_train


# Beginning ...
def main():
  param = ut.load_cnf()            
  xe, ye = load_data_trn()
  W = train(xe, ye, param)
  #W, Cost = train(xe, ye, param)             
  #save_w_cost(W, Cost)
  np.savez('w_snn.npz', *W)


if __name__ == '__main__':   
	 main()

