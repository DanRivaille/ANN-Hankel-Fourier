# SNN's Training :

import numpy      as np
import utility    as ut

#Save weights and MSE  of the SNN
def save_w_mse(W, ann_MSE):
  np.savez('w_snn.npz', *W)
  np.savetxt("costo.csv", np.array(ann_MSE))


def create_momentum(W, L):
  V = [None]*(L+1)
  for i in range(1, L+1 ):
    V[i] = np.zeros_like(W[i])
  return V

#miniBatch-SGDM's Training 
def trn_minibatch(x, y, ann, param):
  N = len(x[0])
  M = param['M_batch']
  V = create_momentum(ann['W'], ann['L'])
  nBatch = N//M
  ann_mse = []

  for n in range(0, nBatch):
    xe = x[:,n*M:(n+1)*M]
    ye = y[:,n*M:(n+1)*M]
    act = ut.forward(ann, param, xe) 
    #act = ut.get_one_hot(np.argmax(ut.forward(ann, param, xe), axis=0) + 1, param['n_classes']).T
    #print(act)
    e = act - ye
    cost = ut.get_mse(act, ye)
    ann_mse.append(cost)
    de_dw = ut.gradW(ann, param, e)
    ann['W'], V = ut.updWV_sgdm(ann, param, de_dw, V)

  return ann_mse, ann

#SNN's Training 
def train(x, y, param):
  ann = init_ann(param, x)
  mse = []

  for i in range(1, param['max_iter'] + 1): 
    #X, Y = ut.sort_data_random(x,y)
    ann_mse, ann = trn_minibatch(x, y, ann, param)
    mse.append(np.mean(ann_mse))

    if np.mod(i,10) == 0:
      print('\n Iterar-SGD: ', i, mse[i-1])

  return ann['W'], mse


def init_ann(param, x):
  ann = ut.create_ann(param['hidden_nodes'], x)
  d = x.shape[0]
  ann['W'] = ut.iniWs(ann['W'], ann['L'], d, param['n_classes'], param['hidden_nodes'])

  return ann


# Load data to train the SNN
def load_data_trn(param):
  FILE_X = 'dtrn.csv'
  FILE_Y = 'etrn.csv'
  X_train, y_train = ut.load_data(FILE_X, FILE_Y, param['n_classes'])
  return X_train, y_train


# Beginning ...
def main():
  param = ut.load_cnf()            
  xe, ye = load_data_trn(param)

  W, Cost = train(xe, ye, param)

  save_w_mse(W, Cost)
  #np.savez('w_snn.npz', *W)


if __name__ == '__main__':   
	 main()

