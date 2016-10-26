## CMU 10-807 HW2
## Train an RBM model.
## Input: MNIST dataset, with feature dimension 784 (from a 28x28 image)
##
## @author Lingxue Zhu

from RBM import *
from util import *
import numpy as np
import os, sys

#####################
## train RBM
#####################
if __name__ == "__main__":

	if (len(sys.argv) != 3):
		print "Usage: python trainRBM.py <dim_hidden> <k_CD>"
		sys.exit(0)

	## parameters
	dim_hidden = int(sys.argv[1]) ## hidden layer size
	k_CD = int(sys.argv[2]) ## steps for contrastive divergence
	dim_feature = 784 ## input feature dimension
	max_epoch = 100
	rate = 0.01 ## learning rate in gradient descent
	seed = 77 ## seed for initialize weights

	## load data and convert to binary
	(trainLabel, trainData) = get_data("data/digitstrain.txt", 
									dim_feature=dim_feature)
	(valLabel, valData) = get_data("data/digitsvalid.txt", 
									dim_feature=dim_feature)

	## RBM
	RBMtrain = RBM(dim_feature, dim_hidden, k_CD=k_CD, k_eval=1, seed=seed)
	(train_loss, val_loss) = RBMtrain.train(
					trainData=trainData, valData=valData,
					max_epoch=max_epoch, rate=rate)

	## save results
	save_dir = "RBMresults/"
	prefix = "RBM" + "_h" + str(dim_hidden) + "_" + str(max_epoch) + \
			"epoch_seed" + str(seed) + "_k" + str(k_CD) + "_"

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	save_model(RBMtrain, save_dir=save_dir, prefix=prefix)
	np.savetxt(save_dir+prefix+"loss.csv",
		np.column_stack((np.array(train_loss), np.array(val_loss))), 
		delimiter=",")

