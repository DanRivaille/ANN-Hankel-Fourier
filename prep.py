import numpy      as np
import utility    as ut

# Save Data from  Hankel's features
def save_data(X, Y, param):
  x_train, y_train, x_test, y_test = create_dtrn_dtst(X, Y, param['p_train'])

  np.savetxt('dtrn.csv', x_train)
  np.savetxt('etrn.csv', y_train)

  np.savetxt('dtst.csv', x_test)
  np.savetxt('etst.csv', y_test)


def create_dtrn_dtst(input, output, p_train):
  # TODO: Shuffle data
  N = output.shape[0]
  index_cut = int(N * p_train)

  x_train = input[:, :index_cut]
  y_train = output[:index_cut]

  x_test = input[:, index_cut:]
  y_test = output[index_cut:]

  return x_train, y_train, x_test, y_test


# normalize data 
def data_norm(X):
  a = 0.01
  b = 0.99
  n_classes = X.shape[0]
  D = X.shape[1]
  for i in range(n_classes):
    for j in range(D):
      X[i][j] = normalize_var(X[i][j], a, b)

  return X

def normalize_var(x, a=0.01, b=0.99):
  x_min = x.min()
  x_max = x.max()
  if x_max > x_min:
    x = ((x - x_min) / (x_max - x_min)) * (b - a) + a
  else:
    x = a
  return x


# Binary Label
def binary_label(class_i, N):
  labels_class = np.repeat(class_i, N)
  return labels_class


# Fourier spectral entropy
def entropy_spectral():
  pass


# Hankel-SVD
def hankel_svd():
  pass


# Hankel's features 
def hankel_features(x, param):
  # TODO: Program this function
  return x[:100]


# Obtain j-th variables of the i-th class
def data_class(x, j, i):
  return x[i, j]


# Create Features from Data
# TODO: Finish this function
def create_features(data, param):
  nbrClass = param['n_classes']
  nbrVariables = data.shape[0]
  N = data.shape[-1]

  Y = np.array([])
  X = np.array([])

  for i in range(nbrClass):
    datF = np.array([])
    for j in range(nbrVariables):
      x = data_class(data, j, i)
      F = hankel_features(x, param)
      datF = np.concatenate((datF, F))
    
    label = binary_label(i + 1, N)

  return X, Y


# Load data from ClassXX.csv
def load_data(n_classes):
  list_classes = []
  for i in range(n_classes):
    file_data = f'data/Data{n_classes}/class{i + 1}.csv'
    x = np.loadtxt(file_data, delimiter=',').T
    list_classes.append(x)

  raw_data = np.array(list_classes)
  return raw_data


# Beginning ...
def main():        
  param = ut.load_cnf()
  data = load_data(param['n_classes'])
  print(data.shape)
  print(data_class(data, 0, 1).shape)
  print(data[0][0][:10])
  X, Y = create_features(data, param)
  #data = data_norm(data)
  print(Y.shape)
  print(X.shape)
  #save_data(X, Y, param)


if __name__ == '__main__':   
	 main()

