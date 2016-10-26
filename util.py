## CMU 10-807 HW2
## helper functions for reading data and saving results
## 
## @author Lingxue Zhu
import numpy as np

def get_data(filename, dim_feature, cutoff=0.5):
	"""
	Load data from a file and convert to binary using the given cutoff,
	and only keep the first dim_feature columns.
	(for MNIST data, the last column is label and should be ignored)
	"""
	data = np.loadtxt(filename, dtype=float, delimiter=",")
	data_labels = data[:, dim_feature]
	data = data[:, xrange(dim_feature)]
	## convert to binary
	data = np.greater_equal(data, cutoff).astype(int)
	return (data_labels, data)


def save_model(model, save_dir="./", prefix=""):
	"""
	Save the parameters in model (from class RBM or Autoencoder) 
	under given directory.
	"""
	np.savetxt(save_dir+prefix+"W.csv", model.W, delimiter=",")
	np.savetxt(save_dir+prefix+"bias_input.csv", model.bias_input, delimiter=",")
	np.savetxt(save_dir+prefix+"bias_hidden.csv", model.bias_hidden, delimiter=",")

