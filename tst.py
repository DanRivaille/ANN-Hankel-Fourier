import numpy as np
import utility as ut

def save_measure(cm, Fsc):
  np.savetxt("cmatriz.csv", np.array(cm), fmt='%i')
  np.savetxt("fscores.csv", np.array(Fsc))


def load_w(L):
  pesos = np.load('w_snn.npz', allow_pickle=True)
  W = [None] * (L + 1)

  for i in range(1, L + 1):
    W[i] = pesos[f'arr_{i}']
  
  for i in range(1, L + 1):
    print(W[i].shape)
  return W


def load_data_test():
  #FILE_X = 'dtst.csv'
  #FILE_Y = 'etst.csv'
  FILE_X = 'dtrn.csv'
  FILE_Y = 'etrn.csv'
  X_test, y_test = ut.load_data(FILE_X, FILE_Y)
  return X_test, y_test
    

# Beginning ...
def main():			
  param = ut.load_cnf()
  xv, yv  = load_data_test()
  ann = ut.create_ann(param['hidden_nodes'], xv)
  ann['W'] = load_w(ann['L'])
  aL = ut.forward(ann, param, xv)

  #cm, Fsc = ut.metricas(aL, yv) 	
  #save_measure(cm, Fsc)
		

if __name__ == '__main__':   
	 main()

