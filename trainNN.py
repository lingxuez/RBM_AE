## 10-807 HW2
## @author Lingxue Zhu
## @version 2016/10/24
##
## Lear an one-layer Neural Network using pre-training
## given by RBM or (denoising) Autoencoder

from NN import *
from util import *
import numpy as np
import os, sys
try:
   import cPickle as pickle
except:
   import pickle


if __name__ == "__main__":

	if len(sys.argv) != 3:
		print "Usage: python trainNN.py <dim_hidden> <method>"

	## model parameters
	dim_hidden = int(sys.argv[1]) ## hidden layer size
	pretrain = sys.argv[2] ## pretrained weights
	dim_feature = 784 ## input feature dimension
	nclass = 10 ## number of classes

	## NN optimization parameters
	momentum=0.5
	nepoch = 150 ## number of epochs to train in NN
	rate = 0.01 ## learning rate in gradient descent
	seed=44 ## seed to initialize the top layer

	## pre-trained weights
	if pretrain == "RBM":
		W = np.loadtxt("RBMresults/RBM_h" + str(dim_hidden) + "_100epoch_seed55_k1_W.csv", 
			delimiter=",")
	elif pretrain == "Auto":
		W = np.loadtxt("Autoresults/Auto_h" + str(dim_hidden) + "_100epoch_seed77_noise0.0_W.csv", 
						delimiter=",")
	elif pretrain == "DenoAuto":
		W = np.loadtxt("Autoresults/Auto_h" + str(dim_hidden) + "_100epoch_seed111_noise0.25_W.csv", 
						delimiter=",")
	elif pretrain == "Random":
		W = None
	else:
		print "<method> must be one of RBM, Auto, DenoAuto, Random."
		sys.exit(0)

	## load data and convert to binary
	(trainLabels, trainData) = get_data("data/digitstrain.txt", 
									dim_feature=dim_feature)
	(valLabels, valData) = get_data("data/digitsvalid.txt", 
									dim_feature=dim_feature)

	## output path to save results
	outdir = "NNout/"
	if not os.path.exists(outdir):
		os.makedirs(outdir)	
	outfile = (outdir + pretrain + "pretrain_NN_h" + str(dim_hidden))

	## NN training
	myNN = NN()
	myNN.fullTrainProcedure(trainFeatures=trainData, trainLabels=trainLabels, 
				valFeatures=valData, valLabels=valLabels, outfile=outfile,
				nclass=nclass, dim_feature=dim_feature, 
				## one layer network, only the top layer is randomly initialized 
				hiddenSizes=[dim_hidden], seeds=[42, seed], bottom_initW=W,
				nepoch=nepoch, rate=rate, momentum=momentum)

	## save model
	with open(outfile + "_model.pkl", 'wb') as fout:
		pickle.dump(myNN, fout)




