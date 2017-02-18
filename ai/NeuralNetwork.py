import numpy as np
from random import shuffle

sigmoid = lambda z: 1 / (1 + np.exp(-z))
sigmoidprime = lambda z: sigmoid(z) * (1 - sigmoid(z)) # derivate of sigmoid function

# the geometric mean is used a lot in finance because it determines the average percent change
prod = lambda iterable: reduce(lambda a, b: a * b, iterable, 1)
geomean = lambda data: prod(data) ** (1 / len(data))
arthmean = lambda data: sum(data) / len(data)

def chunk(array, n):
    chunks = []
    for i in xrange(0, len(array), n):
        chunks.append(array[i:i+n])
    return chunks

class NeuralNetwork(object):
    # opens is a list of opening prices in chronological order for the given stock
    def __init__(self, opens):
        self.opens = opens

        sizes = [5, 5, 2] # size of each layer in the network
        self.sizes = sizes
        self.biases = [np.random.randn(y) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        self.train(100, 1)

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            # because w is a matrix, np.dot(w, a) is equivilant to matrix multiplication
            a = sigmoid(np.dot(w, a) + b)
        return a

    # applies gradient descent on the entire data set for that stock at once because it's small
    # epochs is the number of times to apply gradient descent
    def train(self, epochs, learning_rate):
        for _ in xrange(epochs): # for epochs number of times
            nabla_b = [np.zeros(b.shape) for b in self.biases] # grad. of cost w/ respect to bias
            nabla_w = [np.zeros(w.shape) for w in self.weights] # grad. of cost w/ respect to weight

            # calculate y using the weekly percent changes
            # if the weekly_percent_change was positive
            opens = self.opens
            weeks = chunk(opens, 5)
            for i, week in enumerate(weeks[:-1]):
                next_week = weeks[i+1]
                change_in_average_value = arthmean(next_week) / arthmean(week) - 1
                x = week
                if change_in_average_value > 0:
                    y = (1, 0)
                elif change_in_average_value < 0:
                    y = (0, 1)
                else:
                    y = (0, 0)

                # calculate and add the necessary changes to bias and weight
                delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

    def backprop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        activation = x # activation is input
        activations = [x] # store the activations layer-wise
        zs = [] # matrix for z values

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b # find the new weighed input
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # now go back

        # dCda(output, y) is the gradient of the cost function with respect to the output
        # a = sigmoid(z)
        # dC/da * da/dz = dC/da * d/dz(sigmoid(z)) = dC/da * sigmoid'(z) = dC/dz = delta
        # delta is a matrix in which delta[l][j] is the partial derivative of cost with respect to
        # the weighed input, z, at the jth neuron of the lth layer
        output = activations[-1]
        delta = self.dCda(output, y) * sigmoidprime(zs[-1])
        nabla_b[-1] = delta # BP3
        nabla_w[-1] = np.dot(delta, output) # BP4

        for l in xrange(2, len(self.sizes)):
            z = zs[-l]
            zp = sigmoidprime(z) # z'
            delta = np.dot(np.transpose(self.weights[-l+1]), delta) * zp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, np.transpose(activations[-l-1]))
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def dCda(self, output_activations, y):
        return output_activations - y
