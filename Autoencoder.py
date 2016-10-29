## CMU 10-807 HW2
## Autoencoder and Denoising Autoencoder
## 
## @author Lingxue Zhu
import numpy as np
import math, sys, os
from scipy.special import expit

class Autoencoder(object):
	"""
	Autoencoder and Denoising Autoencoder.
	"""

	def __init__(self, dim_input, dim_hidden,
					W=None, bias_input=None, bias_hidden=None, seed=42):
		"""
		Constructor.
		:param dim_input: the dimension of input features
		:param dim_hidden: the dimension of hidden layer
		:param W: the weight matrix with size (dim_hidden, dim_input); 
					if None, then weights are randomly initialized
		:param bias_input: bias for input feature with size (dim_input, 1);
					if None, then is initialized at zeros
		:param bias_hidden: bias term for hidden layer with size (dim_hidden, 1);
					if None, then is initialized at zeros.
		:param seed: seed to randomly initialize W; ignored if W is not None.
		"""
		(self.dim_input, self.dim_hidden) = (dim_input, dim_hidden)

		## weight
		if W is None:
			## uniform (-bound, bound)
			bound = math.sqrt(6) / math.sqrt(dim_input+dim_hidden)
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


	def train(self, trainData, valData, is_noisy=False, noise_prob=0.25,
							 max_epoch=100, tol=1e-3, rate=0.01):
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
			## construct a noisy input
			if is_noisy:
				noisy_trainData = self.add_noise(trainData, noise_prob)
			else:
				noisy_trainData = trainData

			## scan through training sample using random order
			scan_order = np.random.permutation(nsample)
			for n in scan_order:
				## update gradient using one sample
				data_x = trainData[n, :, np.newaxis]
				noisy_data_x = noisy_trainData[n, :, np.newaxis]

				self.gradientDescent(data_x=data_x, noisy_data_x=noisy_data_x,
										rate=rate, noise_prob=noise_prob)

			## track cross entropy loss
			train_loss += [self.cross_entropy_loss(data=trainData,
											noisy_data=trainData, noise_prob=noise_prob)]
			val_loss += [self.cross_entropy_loss(data=valData,
											noisy_data=valData, noise_prob=noise_prob)]

			## stopping criteria
			if (nEpoch > 1):
				diff_loss = abs(train_loss[nEpoch-1] - train_loss[nEpoch])
			nEpoch += 1

		return (train_loss, val_loss)


	def cross_entropy_loss(self, data, noisy_data, noise_prob=0):
		"""
		Compute the cross entropy loss using current parameters 
		on given data set.
		:param data: binary data, shape (nsample, dim_input)
		:param data: perturbed binary data, shape (nsample, dim_input)
		"""
		loss = 0
		(nsample, nfeature) = data.shape
		if nfeature != self.dim_input:
			raise ValueError('column dimension of data must match dim_input.')

		for n in xrange(nsample):
			data_x = data[n, :, np.newaxis]
			noisy_data_x = noisy_data[n, :, np.newaxis]

			## predicted mean of x (probability)
			(h, x_mean) = self.predict_x_mean(noisy_data_x)

			## loss = - (x log(p) + (1-x) log (1-p))
			## use masked array to avoid log(0), which will be replaced by 0 in the sum
			loss -= np.sum(np.multiply(data_x, np.ma.log(x_mean)).filled(0))/nsample
			loss -= np.sum(np.multiply((1-data_x), np.ma.log(1-x_mean)).filled(0))/nsample

		return loss



	###################
	## Helper Functions
	####################

	def add_noise(self, trainData, noise_prob=0):
		"""
		Randomly choose noise_prob proportion of entries to flip.
		"""
		is_noise = np.random.binomial(n=1, p=noise_prob, size=trainData.shape)
		noisy_trainData = np.multiply(is_noise, 1-trainData) + \
							np.multiply(1-is_noise, trainData)
		return noisy_trainData


	def predict_x_mean(self, noisy_data_x, noise_prob=0):
		"""
		Calculate the predicted mean given input data_x.
		:param data_x: binary input with dimension (dim_input, 1)
		"""
		## hidden layer
		h = expit(self.bias_hidden + self.W.dot(noisy_data_x))
		## predicted x
		x_mean = expit(self.bias_input + self.W.transpose().dot(h))

		return (h, x_mean)


	def gradientDescent(self, data_x, noisy_data_x, rate, noise_prob=0):
		"""
		Perform gradient update using one data point.
		Destructively modify self.W, self.bias_input and self.bias_hidden.
		:param data_x: input sample with shape (dim_input, 1)
		:param noisy_data_x: perturbed input sample with shape (dim_input, 1)
		:param rate: learning rate
		"""
		## gradient w.r.t pre-activation a=bias_input + t(W).dot(h)
		(h, pred_x) = self.predict_x_mean(noisy_data_x)
		grad_a = pred_x - data_x

		## gradients for parameters
		h_hcomp = np.multiply(h, 1-h)
		grad_W = np.multiply(self.W.dot(grad_a), h_hcomp)\
						.dot(noisy_data_x.transpose()) + \
						h.dot(grad_a.transpose())
		grad_bias_input = grad_a
		grad_bias_hidden = h_hcomp * (np.sum(self.W, axis=0).dot(grad_a))

		## update gradients
		self.W -= rate * grad_W
		self.bias_input -= rate * grad_bias_input
		self.bias_hidden -= rate * grad_bias_hidden


