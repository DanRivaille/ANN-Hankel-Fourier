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
def load_data():
  pass


# Beginning ...
def main():        
  params = ut.load_cnf()	
  print(params)

  #data = load_data()	
  #input_dat, out_dat = create_features(data, params)
  #input_dat = data_norm(input_dat)
  #save_data(input_dat, out_dat)


if __name__ == '__main__':   
	 main()

