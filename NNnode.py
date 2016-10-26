## 10-807 HW1
## @author Lingxue Zhu
## @version 2016/09/23
##
## Define the classes for nodes in Neuron Networks

import math
import numpy as np
import warnings
from scipy import misc
from scipy import special

class NNnode(object):

	## constructor
	def __init__(self, shape, parent=None, seed=None, 
					name="NNnode", wDecay=0):
		self.shape = shape
		self.parent = parent
		self.name = name
		self.wDecay=wDecay

		self.children = dict()
		self.gradient = np.zeros(shape=shape, dtype=float)
		self.initValue(seed)

	def initValue(self, seed):
		"""
		initialize values and gradients; default: empty matrices
		"""
		self.value = np.zeros(shape=self.shape, dtype=float)

	## gradient at current value
	def setGradient(self, gradient):
		if (gradient.shape != self.shape):
			warnings.warn("The new gradient shape" + str(gradient.shape)
					+ "does not equal to the previous shape" + str(self.shape))
		self.gradient = np.copy(gradient)

	def getGradient(self):
		return self.gradient

	## current value
	def setValue(self, value):
		if (value.shape != self.shape):
			warnings.warn("The newValue shape" + str(value.shape)
					+ "does not equal to the previous shape" + str(self.shape))
		self.value = np.copy(value)

	def getValue(self):
		return self.value

	## children
	def setChildren(self, children):
		if not isinstance(children, dict):
			raise TypeError("children must be a dict().")
		self.children = children

	def getChildren(self):
		return self.children

	
	def getChild(self, name):
		"""
		get specific children; raises a KeyError if name is not in children
		"""
		return self.children[name]

	def hasChildren(self):
		return len(self.children)>0

	## parent: only one parent
	def hasParent(self):
		return self.parent is not None

	def getParent(self):
		return self.parent

	## update parameter: default is to do nothing
	def update(self, rate=0.1, momentum=0):
		return

	def bprop(self):
		"""
		back propogation: calculate gradient
		store old gradient for momentum
		"""
		self.oldGradient = self.getGradient()
		self.setGradient(self.newGradient())

	def newGradient(self):
		return np.zeros(self.shape)

	@staticmethod
	def activateGradient(hValue):
		"""
		gradient of activation function
		sigmoid: g'(a) = g(a)*(1-g(a)) = h * (1-h); same dmension as hValue
		"""	
		return np.multiply(hValue, 1-hValue)


	@staticmethod
	def activate(aValue):
		"""
		activate function: sigmoid
		g(a) = 1/(1+exp(-a)); same dimension as aValue
		"""
		return special.expit(aValue)


	@staticmethod
	def outActivate(aValue, logScale=False):
		"""
		out activate function: softmax; same dimension as aValue
		"""
		if logScale:
			return aValue - misc.logsumexp(aValue)
		else:
			return np.exp(aValue) / np.sum(np.exp(aValue), axis=0)


## node a: pre-activation
class aNode(NNnode):

	def bprop(self, yLabel=None):
		"""
		Overwrites back propogation
		"""
		self.oldGradient = self.getGradient()
		self.setGradient(self.newGradient(yLabel))
		

	def newGradient(self, y=None):
		if self.hasParent():
			hGradient = self.parent.getGradient()
			hValue = self.parent.getValue()
			gradient = np.multiply(hGradient, NNnode.activateGradient(hValue))
		else: ## top layer
			indicator = np.zeros(self.shape)
			indicator[y] = 1
			## f(x)
			predict = NNnode.outActivate(self.value, logScale=True)
			predict = np.exp(predict)
			gradient = predict - indicator
		return gradient

	def fprop(self, dropout=0, useRandom=True):
		"""
		forward propogation: calculate value
		"""
		wValue = self.getChild("w").getValue()
		hValue = self.getChild("h").getValue()
		bValue = self.getChild("b").getValue()
		self.setValue(wValue.dot(hValue) + bValue)


## node h: hidden unit; h=g(a)
class hNode(NNnode):

	def newGradient(self):
		wValue = self.parent.getChild("w").getValue()
		aGradient = self.parent.getGradient()
		gradient = wValue.transpose().dot(aGradient)
		return gradient


	def fprop(self, dataValue=None, dropout=0, useRandom=False):
		"""
		dataValue: same dimension as hNode.value
		dropout: probability of dropout
		useRandom: whether to use random dropout, or to use expectation
		"""
		if self.hasChildren(): ## hidden unit
			aValue = self.getChild("a").getValue()
			hValue = NNnode.activate(aValue)
			if useRandom:
				mask = np.random.binomial(1, 1-dropout, self.shape)
				self.setValue(np.multiply(hValue, mask))
			else:
				self.setValue(hValue / (1-dropout))
		else: ## last layer: data
			self.setValue(dataValue)


## node W: weight parameter
class wNode(NNnode):

	def initValue(self, seed):
		bound = math.sqrt(6) / math.sqrt(sum(self.shape))
		## uniform random values from [-bound, bound]
		np.random.seed(seed)
		self.value = np.random.rand(self.shape[0], self.shape[1]) * 2*bound - bound
		self.oldupdate = np.zeros(shape=self.shape, dtype=float)

	def newGradient(self):
		aGradient = self.parent.getGradient()
		hValue = self.parent.getChild("h").getValue()
		gradient = aGradient.dot(hValue.transpose())
		## weight decay
		gradient += self.wDecay * self.value 
		return gradient

	def update(self, rate=0.1, momentum=0):
		newvalue = self.value - rate * self.gradient + momentum * self.oldupdate
		## update
		self.oldupdate = newvalue - self.value
		self.setValue(newvalue)


## node b: bias parameter
class bNode(NNnode):
	def initValue(self, seed):
		"""
		initialize values and gradients; default: empty matrices
		"""
		self.value = np.zeros(shape=self.shape, dtype=float)
		self.oldupdate = np.zeros(shape=self.shape, dtype=float)

	def newGradient(self):
		gradient = self.parent.getGradient()
		return gradient


	def update(self, rate=0.1, momentum=0):
		newvalue = self.value - rate * self.gradient + momentum * self.oldupdate
		## update
		self.oldupdate = newvalue - self.value
		self.setValue(newvalue)
		


