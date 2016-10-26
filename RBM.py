## CMU 10-807 HW2
## A Restricted Boltzman Machine (RBM)
## using Contrastive Divergence for learning 
## 
## @author Lingxue Zhu
import numpy as np
import math, sys, os
from scipy.special import expit

class RBM(object):
	"""
	A restricted boltzman machine with binary input and hidden layer.
	Inherits from the class BaseOnelayer.
	"""

	def __init__(self, dim_input, dim_hidden,
					W=None, bias_input=None, bias_hidden=None, seed=42,
					k_CD=1, k_eval=1):
		"""
		Constructor.
		"""
		(self.k_CD, self.k_eval) = (k_CD, k_eval)
		(self.dim_input, self.dim_hidden) = (dim_input, dim_hidden)

		## weight
		if W is None:
			## uniform (-bound, bound)
			bound = 4 * math.sqrt(6) / math.sqrt(dim_input+dim_hidden)
			np.random.seed(seed)
			self.W = np.random.rand(dim_hidden, dim_input) * 2 * bound - bound
		elif W.shape == (dim_hidden, dim_input):
			self.W = W
		else:
			raise ValueError('shape of W should match (dim_hidden, dim_input)')

		## bias terms
		if bias_input is None:
			self.bias_input = np.zeros((dim_input, 1))
		elif bias_input.shape == (dim_input, 1):
			self.bias_input = bias_input
		else:
			raise ValueError('shape of bias_input should match (dim_input, 1).')

		if bias_hidden is None:
			self.bias_hidden = np.zeros((dim_hidden, 1))
		elif bias_hidden.shape == (dim_hidden, 1):
			self.bias_hidden = bias_hidden
		else:
			raise ValueError('shape of bias_hidden should match (dim_hidden, 1).')

	#######################
	## Main functions
	#######################

	def train(self, trainData, valData, max_epoch=100, tol=1e-3, rate=0.01):
		"""
		Train the model using training data, 
		and keep track of cross-entropy loss on training and validating data.
		:param trainData: binary training data, shape (nsample, dim_input)
		:param valData: binary validating data, shape (nsample2, dim_input)
		"""
		(nsample, nfeature) = trainData.shape
		if nfeature != self.dim_input:
			raise ValueError('column dimension of trainData must match dim_input.')

		## training
		(train_loss, val_loss) = ([], [])
		(diff_loss, nEpoch) = (tol+1, 0)	
		while (nEpoch < max_epoch) and (diff_loss > tol):
			## scan through training sample using random order
			scan_order = np.random.permutation(nsample)
			for n in scan_order:
				## update gradient using one sample
				data_x = trainData[n, :, np.newaxis]
				self.gradientDescent(data_x=data_x, rate=rate)

			## track cross entropy loss
			train_loss += [self.cross_entropy_loss(data=trainData)]
			val_loss += [self.cross_entropy_loss(data=valData)]

			## stopping criteria
			if (nEpoch > 1):
				diff_loss = abs(train_loss[nEpoch-1] - train_loss[nEpoch])
			nEpoch += 1

		return (train_loss, val_loss)


	def cross_entropy_loss(self, data):
		"""
		Compute the cross entropy loss using current parameters 
		on given data set.
		:param data: binary data, shape (nsample, dim_input)
		"""
		loss = 0
		(nsample, nfeature) = data.shape
		if nfeature != self.dim_input:
			raise ValueError('column dimension of data must match dim_input.')

		for n in xrange(nsample):
			data_x = data[n, :, np.newaxis]

			## predicted mean of x (probability)
			(h, x_mean) = self.predict_x_mean(data_x)

			## loss = - (x log(p) + (1-x) log (1-p))
			loss -= np.sum(np.multiply(data_x, np.log(x_mean)))/nsample
			loss -= np.sum(np.multiply((1-data_x), np.log(1-x_mean)))/nsample

		return loss

	#######################
	## Helper functions
	#######################

	def gradientDescent(self, data_x, rate):
		"""
		Perform gradient update using contrastive divergence on one data point.
		Destructively modify self.W, self.bias_input and self.bias_hidden.
		:param data_x: input sample with shape (dim_input, 1)
		:param rate: learning rate
		"""
		## positive sample
		grad_pos = self.energy_gradient(data_x)

		## negative sample from Gibbs 
		(neg_h, neg_x) = self.gibbs_chain(init_x=data_x, k_CD=self.k_CD)
		grad_neg = self.energy_gradient(neg_x)

		## gradient update
		self.W -= rate * (grad_pos[0] - grad_neg[0])
		self.bias_input -= rate * (grad_pos[1] - grad_neg[1])
		self.bias_hidden -= rate * (grad_pos[2] - grad_neg[2])


	def energy_gradient(self, x):
		"""
		Calculate the (estimated) gradient of energy E(h, x) at given x,
		with respect to W, bias_input, bias_hidden at current values.
		:param x: input vector with shape (dim_input, 1)
		"""
		h_mean = expit(self.bias_hidden + self.W.dot(x))
		grad_W = -h_mean.dot(x.transpose())
		grad_bias_input = -x
		grad_bias_hidden = -h_mean
		
		return (grad_W, grad_bias_input, grad_bias_hidden)


	def gibbs_chain(self, init_x=None, k_CD=1):
		"""
		Peform Gibbs sampling for k_CD steps.
		:param init_x: starting point of gibbs chain, shape (dim_input, 1)
		"""
		## initialization
		if init_x is None:
			x = np.random.binomial(n=1, p=0.5, size=(self.dim_input, 1))
		else:
			x = init_x 

		## gibbs sampling with step k_CD
		for k in xrange(k_CD):
			h = self.gibbs_sample_h(x)
			x = self.gibbs_sample_x(h)

		return (h, x)


	def gibbs_sample_h(self, x):
		"""
		Sample a new h from p(h|x) using current parameters.
		:param x: shape (dim_input, 1)
		:return: shape (dim_hidden, 1)
		"""
		h_mean = expit(self.bias_hidden + self.W.dot(x))
		return self.bernoulli(h_mean)


	def gibbs_sample_x(self, h):
		"""
		Sample a new x from p(x|h) using current parameters.
		:param h: shape (dim_hidden, 1)
		:return: shape (dim_input, 1)
		"""
		x_mean = expit(self.bias_input + self.W.transpose().dot(h))
		return self.bernoulli(x_mean)


	def bernoulli(self, means):
		"""
		Generate a vector of binary bernoulli random variables.
		:param means: vector of probability (shape=(p, 1))
		:return rv: random variables (shape=(p, 1))
		"""
		rv = np.array([np.random.binomial(n=1, p=mean, size=1) for mean in means])
		return rv


	def predict_x_mean(self, data_x):
		"""
		Calculate the predicted mean given input data_x,
		for evaluating cross-entropy loss.
		:param data_x: binary input with dimension (dim_input, 1)
		"""
		## obtain hidden layer from gibbs sampling
		x = data_x 
		for k in xrange(self.k_eval):
			h = self.gibbs_sample_h(x)
			x = self.gibbs_sample_x(h)

		## predicted x
		x_mean = expit(self.bias_input + self.W.transpose().dot(h))

		return (h, x_mean)



