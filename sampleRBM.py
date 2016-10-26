## CMU 10-807 HW2
## Generate new samples from a trained RBM model
##
## @author Lingxue Zhu
from RBM import *
import numpy as np
import os

if __name__ == "__main__":

	###########################
	## load trained results
	##########################
	seed = 55
	k = 1
	res_dir="RBMresults/"
	prefix="RBM_h100_100epoch_seed" + str(seed) + "_k" + str(k) + "_"

	W = np.loadtxt(res_dir+prefix+"W.csv", delimiter=",")
	bias_input = np.loadtxt(res_dir+prefix+"bias_input.csv", delimiter=",")
	bias_hidden = np.loadtxt(res_dir+prefix+"bias_hidden.csv", delimiter=",")


	####################################
	## sample image
	####################################
	## An RBM with trained parameters
	(dim_hidden, dim_feature) = W.shape
	RBMsample= RBM(dim_feature, dim_hidden, W=W, 
		bias_input=bias_input[:, np.newaxis], 
		bias_hidden=bias_hidden[:, np.newaxis])

	## Generate gibbs chains starting from randomly initialized inputs
	num_image = 100
	k_sample = 1000
	sample_images = np.zeros((num_image, dim_feature))
	for n in xrange(num_image):
		## gibbs sampling
		(h, x) = RBMsample.gibbs_chain(k_CD=k_sample)
		sample_images[n, :] = x.flatten()


	############################
	## save results
	############################
	save_dir = "sample_images/"
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	np.savetxt(save_dir + "sample_images_seed" + str(seed) + \
		 "_k" + str(k) + ".csv", sample_images, delimiter=",")


