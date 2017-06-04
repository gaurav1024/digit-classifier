import numpy as np

class Network:
	def __init__(self, sizes):
		self.sizes = sizes
		self.num_layers = len(sizes)
		self.biases = [np.random.random(y, 1) for y in sizes[1:]]
		self.weights = [np.random.random(y, x) for y, x in zip(sizes[1:], sizes[:-1])]
