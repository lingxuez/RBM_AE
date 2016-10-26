## 10-807 HW2
## @author Lingxue Zhu
## @version 2016/10/24
##
## A Neural Network

from NNnode import *
import numpy as np
import math, os, sys
from scipy import misc

class NN(object):

	def __init__(self):
		self.initTree(sizes=[1], seeds=[]) ## an empty tree with 1 node

	def initTree(self, sizes, seeds, bottom_initW=None, wDecay=0):
		"""
		Initialize the tree structure of neuron network.
		-- sizes: sizes of layers bottom to top. 
		       Bottom size = feature size; top size = class number
		-- seeds: seeds to use for W from bottom to top. 
				len(seeds) = len(sizes)-1
		-- bottom_initW: shape (1st hidden layer, input dimension)
		"""
		## root: top a
		height = len(sizes)-1
		node = aNode(shape=(sizes[height], 1), name="a_"+str(height))
		self.root = node
		self.predict()

		for layer in xrange(height, 0, -1):
			## node "a"
			node.setChildren({
			"w" : wNode(shape=(sizes[layer], sizes[layer-1]), 
						parent=node, name="w_"+str(layer),
						seed=seeds[layer-1], 
						wDecay=wDecay),
			"b" : bNode(shape=(sizes[layer], 1), parent=node,
						name="b_"+str(layer)),
			"h" : hNode(shape=(sizes[layer-1], 1), parent=node,
						name="h_"+str(layer-1))
			})

			## pre-trained W for bottom layer
			if (layer == 1) and (bottom_initW is not None):
				node.getChild("w").setValue(bottom_initW)

			## node "h"
			node = node.getChild("h")
			if layer > 1:
				node.setChildren({
					"a" : aNode(shape=(sizes[layer-1], 1), parent=node,
								name="a_"+str(layer-1))
					})
				node = node.getChild("a")
			else: ## last layer
				self.data = node


	def predict(self):
		"""
		predict class probabilities given top values of a
		use log probability to avoid overflow
		"""
		self.outLog = NNnode.outActivate(self.root.getValue(), logScale=True)
		

	def getPredict(self):
		self.predict()
		self.out = np.exp(self.outLog)
		return self.out

	def forwardProp(self, node=None, dataValue=None, dropout=0, 
					useRandom=True):
		"""
		Forward propogate from node & its ancestors: 
		Given features from one sample, 
		update values of (a, h) using current parameters (W, b).
		Note that (W, b) values remain unchanged
		"""
		## default: from bottom layer
		if node is None:
			node = self.data
			node.fprop(dataValue, dropout=dropout, useRandom=useRandom)
		else:
			node.fprop(dropout = dropout, useRandom=useRandom)

		## forward propogation
		while(node.hasParent()):
			node = node.getParent()
			node.fprop(dropout=dropout, useRandom=useRandom)

		## update prediction
		self.predict()


	def backProp(self, node=None, yLabel=None):
		"""
		Backward propogate from node & its descendants: 
		Given current values of (a,h,W,b),
		update gradient w.r.t. loss function on one sample
		"""
		## default: start from root
		if node is None: 
			node = self.root
			node.bprop(yLabel=yLabel)
		else:
			node.bprop()

		if node.hasChildren():
			for child in node.getChildren().values():
				self.backProp(node=child)


 	def updatePara(self, node=None, rate=0.1, momentum=0):
		"""
		Update the parameters (W, b) in node & its descendants.
		-- rate: learning rate
		-- node: starting node.
		"""
		if node is None: ## default: start from root
			node = self.root

		node.update(rate=rate, momentum=momentum)
		if node.hasChildren():
			for child in node.getChildren().values():
				self.updatePara(node=child, rate=rate, momentum=momentum)


	def entropyLoss(self, yLabel):
		"""
		Cross-entropy loss for current data point.
		"""
		## avoid the numerical issue by directly using log scale
		return (-self.outLog[yLabel])	

	def mceLoss(self, yLabel):
		"""
		Mean classification error for current data point.
		"""
		predLabel = int(np.argmax(self.outLog, axis=0))
		return int(yLabel != predLabel)	


	def trainOneSample(self, dataValue, yLabel, rate=0.1, momentum=0, dropout=0):	
		"""
		One iteration of training using one data point.
		Use random dropout in training phase
		"""	
		self.forwardProp(dataValue=dataValue, dropout=dropout, useRandom=True)
		self.backProp(yLabel=yLabel)
		self.updatePara(rate=rate, momentum=momentum)
		return (self.entropyLoss(yLabel), self.mceLoss(yLabel))


	def train(self, trainFeatures, trainLabels, valFeatures, valLabels,
					nepoch=200, rate=0.1, momentum=0, dropout=0):
		"""
		Train neuron network using given training samples
		-- dataFeatures: each row ~ 1 observation; 
	           	number of columns = size of bottom layer
	    -- dataLabels: vector of class labels [0, C-1];
	    		number of classes C = size of top layer
	    -- nepoch: number of epochs to run
	    -- rate: learning rate
		"""
		numSample = trainFeatures.shape[0]
		## record training and validating losses at every epoch
		trainLoss = {"entropy":np.zeros(shape=(nepoch+1)), 
						"mce":np.zeros(shape=(nepoch+1))}
		valLoss = {"entropy":np.zeros(shape=(nepoch+1)), 
						"mce":np.zeros(shape=(nepoch+1))} 

		## starting loss
		lossTRN = self.computeDataLoss(trainFeatures, trainLabels, dropout)
		trainLoss["entropy"][0] = lossTRN[0]
		trainLoss["mce"][0] = lossTRN[1]

		## calculate loss on validating sample
		lossVAL = self.computeDataLoss(valFeatures, valLabels, dropout)
		valLoss["entropy"][0] = lossVAL[0]
		valLoss["mce"][0] = lossVAL[1]

		for epoch in xrange(nepoch):
			## 1 epoch = 1 pass through training samples
			for n in xrange(numSample):
				self.trainOneSample(
								dataValue=trainFeatures[n, :, np.newaxis], 
								yLabel=int(trainLabels[n]), 
								rate=rate, momentum=momentum, dropout=dropout)
			## update loss on training sample
			lossTRN = self.computeDataLoss(trainFeatures, trainLabels, dropout)
			trainLoss["entropy"][epoch+1] = lossTRN[0]
			trainLoss["mce"][epoch+1] = lossTRN[1]

			## calculate loss on validating sample
			lossVAL = self.computeDataLoss(valFeatures, valLabels, dropout)
			valLoss["entropy"][epoch+1] = lossVAL[0]
			valLoss["mce"][epoch+1] = lossVAL[1]

		return(trainLoss, valLoss)


	def computeDataLoss(self, valFeatures, valLabels, dropout=0):
		"""
		Given data, calculate the (entropyLoss, mceLoss) using current coefficients
		With dropout, now use expected values instead of random dropouts
		"""
		numSample = valFeatures.shape[0]
		(entropy, mce) = (0, 0) ## (entropyLoss, mceLoss)

		for n in xrange(numSample):
			yLabel = int(valLabels[n])
			self.forwardProp(dataValue=valFeatures[n, :, np.newaxis], 
							dropout=dropout, useRandom=False)
			entropy += self.entropyLoss(yLabel)/float(numSample)
			mce += self.mceLoss(yLabel)/float(numSample)

		return (entropy, mce)


	def fullTrainProcedure(self, trainFeatures, trainLabels, 
				valFeatures, valLabels, outfile=None,
				nclass=10, dim_feature=784, 
				hiddenSizes=[100], seeds=[23, 42], bottom_initW=None,
				nepoch=50, rate=0.1, momentum=0, dropout=0, wDecay=0):
		"""
		Complete training procedure of the neuron network.
		-- hiddenSizes: sizes of hidden layers, from bottom to top. 
		-- seeds: seeds to use for W from bottom to top. 
				len(seeds) = len(hiddenSizes) + 1
		"""

		## NN: bottom = features; top = nclass
		sizes = [dim_feature] + hiddenSizes + [nclass]
		self.initTree(sizes, seeds=seeds, bottom_initW=bottom_initW, wDecay=wDecay)

		## training and validating
		(trainLoss, valLoss) = self.train(
						trainFeatures=trainFeatures, trainLabels=trainLabels, 
						valFeatures=valFeatures, valLabels=valLabels,
						nepoch=nepoch, rate=rate, momentum=momentum, 
						dropout=dropout)

		self.writeLoss(trainLoss, valLoss, outfile+"_loss.csv")
		return(trainLoss, valLoss)

	def writeLoss(self, trainLoss, valLoss, outfile):
		"""
		Write the loss to file.
		"""
		nepoch = len(trainLoss["entropy"])
		with open(outfile, mode="wt") as fout:
			fout.write("epoch,train_entropy,train_mce,val_entropy,val_mce\n")
			for epoch in xrange(nepoch):
				fout.write("%d,%f,%f,%f,%f\n" % (epoch, 
					trainLoss["entropy"][epoch], trainLoss["mce"][epoch],
					valLoss["entropy"][epoch], valLoss["mce"][epoch]))

