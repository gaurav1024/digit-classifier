import numpy as np

class Network:
	def __init__(self, sizes):
		self.sizes = sizes
		self.num_layers = len(sizes)
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x) for y, x in zip(sizes[1:], sizes[:-1])]

	def feedforward(self, a):
		for w, b in zip(self.weights, self.biases):
			a = np.dot(w, a)+b
		return a
		
def sigmoid(z):
	return 1.0/(1.0 + np.exp(-z))