## CMU 10-807 HW2
## Train an Autoencoder.
## Input: MNIST dataset, with feature dimension 784 (from a 28x28 image)
##
## @author Lingxue Zhu

from Autoencoder import *
from util import *
import numpy as np
import os, sys

#####################
## train Autoencoder
#####################
if __name__ == "__main__":
	
	if (len(sys.argv) != 3):
		print "Usage: python trainAutoencoder.py <dim_hidden> <noise_probability>"
		sys.exit(0)

	## parameters
	dim_hidden = int(sys.argv[1]) ## hidden layer size
	noise_prob = float(sys.argv[2]) ## probability of adding noise

	dim_feature = 784 ## input feature dimension
	max_epoch = 100 
	is_noisy = True ## use denoising Autoencoder	
	rate = 0.05 ## learning rate in gradient descent
	seed = 77 ## seed for initialize weights

	## load data and convert to binary
	(trainLabel, trainData) = get_data("data/digitstrain.txt", 
									dim_feature=dim_feature)
	(valLabel, valData) = get_data("data/digitsvalid.txt", 
									dim_feature=dim_feature)
	
	## Autoencoder
	Auto_train = Autoencoder(dim_feature, dim_hidden, seed=seed)
	(train_loss, val_loss) = Auto_train.train(
					trainData=trainData, valData=valData,
					is_noisy=is_noisy, noise_prob=noise_prob,
					max_epoch=max_epoch, rate=rate)

	## save results
	save_dir="Autoresults/"
	prefix = "Auto_h_" + str(dim_hidden) + \
			str(max_epoch) + "epoch_seed" + str(seed) +\
			"_noise" + str(noise_prob) + "_"

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	save_model(Auto_train, save_dir=save_dir, prefix=prefix)
	np.savetxt(save_dir + prefix + "loss.csv",
		np.column_stack((np.array(train_loss), np.array(val_loss))), 
		delimiter=",")
