import math, random, string
random.seed(0)
def rand(a, b):
	return (b-a) * random.random() + a

def makeMatrix(I, J, fill = 0.0):
	m = []
	for i in range(I):
		m.append([fill]*J)
	return m

def sigmoid(x):
	return math.tanh(x)

def dsigmoid(y):
	return 1.0 - y**2

class NeuralNetwork:
	def __init__(self, inputNodes, hiddenNodes, outputNodes):
		self.inputNodes = inputNodes + 1
		self.hiddenNodes = hiddenNodes
		self.outputNodes = outputNodes

		self.inputActivation = [1.0] * self.inputNodes
		self.hiddenActivation = [1.0] * self.hiddenNodes
		self.outputActivation = [1.0] * self.outputNodes

		self.inputWeight = makeMatrix(self.inputNodes, self.hiddenNodes)
		self.outputWeight = makeMatrix(self.hiddenNodes, self.outputNodes)

		for i in range(self.inputNodes):
			for j in range(self.hiddenNodes):
				self.inputWeight[i][j] = rand(-0.2, 0.2)
		for k in range(self.hiddenNodes):
			for l in range(self.outputNodes):
				self.outputWeight[k][l] = rand(-2.0, 2.0)

		self.ci = makeMatrix(self.inputNodes, self.hiddenNodes)
		self.co = makeMatrix(self.hiddenNodes, self.outputNodes)

	def update(self, inputs):
		if len(inputs) != self.inputNodes - 1:
			raise ValueError('Incorrect number of inputs')

		for i in range(self.inputNodes - 1):
			self.inputActivation[i] = inputs[i]

		for j in range(self.hiddenNodes):
			sum = 0.0
			for i in range(self.inputNodes):
				sum += self.inputActivation[i] * self.inputWeight[i][j]
			self.hiddenActivation[j] = sigmoid(sum)

		for k in range(self.outputNodes):
			sum = 0.0
			for j in range(self.hiddenNodes):
				sum += self.hiddenActivation[j] * self.outputWeight[j][k]
			self.outputActivation[k] = sigmoid(sum)

		return self.outputActivation[:]

	def backPropegate(self, targets, N, M):
		if len(targets) != self.outputNodes:
			raise ValueError('wrong number of target values')

		output_deltas = [0.0] * self.outputNodes
		for k in range(self.outputNodes):
			error = targets[k]-self.outputActivation[k]
			output_deltas[k] = dsigmoid(self.outputActivation[k]) * error

		hidden_deltas = [0.0] * self.hiddenNodes
		for j in range(self.hiddenNodes):
			error = 0.0
			for k in range(self.outputNodes):
				error = error + output_deltas[k]*self.outputWeight[j][k]
			hidden_deltas[j] = dsigmoid(self.hiddenActivation[j]) * error

		for j in range(self.hiddenNodes):
			for k in range(self.outputNodes):
				change = output_deltas[k]*self.hiddenActivation[j]
				self.outputWeight[j][k] = self.outputWeight[j][k] + N*change + M*self.co[j][k]
				self.co[j][k] = change

		for i in range(self.inputNodes):
			for j in range(self.hiddenNodes):
				change = hidden_deltas[j]*self.inputActivation[i]
				self.inputWeight[i][j] = self.inputWeight[i][j] + N*change + M*self.ci[i][j]
				self.ci[i][j] = change

		error = 0.0
		for k in range(len(targets)):
			error = error + 0.5*(targets[k] - self.outputActivation[k])**2
			
		return error

	def test(self, inputNodes):
		print(inputNodes, '->', self.update(inputNodes))
		return self.update(inputNodes)[0]

	def weights(self):
		print("Input Weights:")
		for i in range(self.inputNodes):
			print(self.inputWeight[i])
		print("\nOutput Weights:")
		for j in range(self.hiddenNodes):
			print(self.outputWeight[j])

	def train(self, patterns, iter = 1000, N = 0.7, M = 0.2):
		for i in range(iter):
			error = 0.0
			for p in patterns:
				inputs = p[0]
				targets = p[1]
				self.update(inputs)
				error += self.backPropegate(targets, N, M)
			if i % 100 == 0:
				print("error %-.7f" % error)