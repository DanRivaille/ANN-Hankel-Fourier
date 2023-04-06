import numpy      as np
import utility    as ut

# Save Data from  Hankel's features
def save_data(X,Y):
  pass


# normalize data 
def data_norm():
  pass


# Binary Label
def binary_label():
  pass


# Fourier spectral entropy
def entropy_spectral():
  pass


# Hankel-SVD
def hankel_svd():
  pass


# Hankel's features 
def hankel_features(X,Param):
  pass


# Obtain j-th variables of the i-th class
def data_class(x,j,i):
  pass


# Create Features from Data
def create_features(X,Param):
  pass


# Load data from ClassXX.csv
def load_data(n_classes):
  list_classes = []
  for i in range(n_classes):
    file_data = f'data/Data{n_classes}/class{i + 1}.csv'
    x = np.loadtxt(file_data, delimiter=',')
    list_classes.append(x)

  raw_data = np.array(list_classes)
  return raw_data


# Beginning ...
def main():        
  param = ut.load_cnf()	
  print(param)

  data = load_data(param['n_classes'])	
  print(data.shape)
  #input_dat, out_dat = create_features(data, param)
  #input_dat = data_norm(input_dat)
  #save_data(input_dat, out_dat)


if __name__ == '__main__':   
	 main()

