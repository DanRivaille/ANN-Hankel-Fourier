import numpy as np
import utility as ut

def save_measure(cm, Fsc):
  np.savetxt("cmatriz.csv", np.array(cm), fmt='%i')
  np.savetxt("fscores.csv", np.array(Fsc))


def load_w():
  W = np.load('w_snn.npz')
  return W


def load_data_test():
  FILE_X = 'dtst.csv'
  FILE_Y = 'etst.csv'
  X_test, y_test = ut.load_data(FILE_X, FILE_Y)
  return X_test, y_test
    

# Beginning ...
def main():			
  param = ut.load_cnf()
  xv, yv  = load_data_test()
  ann = ut.create_ann(param, xv)
  ann['W'] = load_w()
  aL = ut.forward(ann, param, xv)      		
  cm, Fsc = ut.metricas(aL, yv) 	
  save_measure(cm, Fsc)
		

if __name__ == '__main__':   
	 main()

