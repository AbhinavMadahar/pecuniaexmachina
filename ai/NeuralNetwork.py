import numpy as np
from random import shuffle
from math import sqrt

sigmoid = lambda z: 1 / (1 + np.exp(-z))
sigmoidprime = lambda z: sigmoid(z) * (1 - sigmoid(z)) # derivate of sigmoid function

# the geometric mean is used a lot in finance because it determines the average percent change
prod = lambda iterable: reduce(lambda a, b: a * b, iterable, 1)
geomean = lambda data: prod(data) ** (1 / len(data))
arthmean = lambda data: sum(data) / len(data)

distance_between = lambda a, b: sqrt(sum((a[j] - b[j]) ** 2 for j in xrange(len(a))))

def chunk(array, n):
    chunks = []
    for i in xrange(0, len(array), n):
        chunks.append(array[i:i+n])
    return chunks

class NeuralNetwork(object):
    # prices is a list of priceing prices in chronological order for the given stock
    # sizes is the size of each layer in the network
    def __init__(self, prices, sizes):
        self.prices = prices

        self.sizes = sizes
        self.biases = [np.random.randn(y) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            # because w is a matrix, np.dot(w, a) is equivilant to matrix multiplication
            a = sigmoid(np.dot(w, a) + b)
        return a

    # applies gradient descent on the entire data set for that stock at once because it's small
    # epochs is the number of times to apply gradient descent
    def train(self, epochs, learning_rate):
        learning_rate = float(learning_rate) # sorry for the type-casting lol
        for n in xrange(epochs): # for epochs number of times
            nabla_b = [np.zeros(b.shape) for b in self.biases] # grad. of cost w/ respect to bias
            nabla_w = [np.zeros(w.shape) for w in self.weights] # grad. of cost w/ respect to weight

            # calculate y using the weekly percent changes
            # if the weekly_percent_change was positive
            prices = self.prices
            weeks = chunk(prices, self.sizes[0])
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
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

            self.weights = [w - (learning_rate / len(self.prices)) * nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b - (learning_rate / len(self.prices)) * nb for b, nb in zip(self.biases, nabla_b)]

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

        # let's do this literately
        # backpropagation is all about delta, the partial derivate of C with respect to z[l][j]
        # z[l][j] is the weighed input (w[l][j] dot a + b) of the jth neuron of the lth layer
        # a[l][j] = sigmoid(w[l][j] dot a + b)
        # a[l][j] = sigmoid(z[l][j])
        # we will now use (BP1) to find delta
        # we will refer to the partial derivate of C with respect to a[l][j] as dC/da
        # dC/dz = dC/da * da/dz
        # delta = dC/da * d/dz(sigmoid(z))
        # delta = dC/da * sigmoid'(z)
        # we can calculate dC/da with self.dCda and we have the sigmoidprime function already
        # now, we can calculate delta
        dCda = self.dCda(activations[-1], y)
        dadz = sigmoidprime(zs[-1])
        delta = dCda * dadz

        # if b is the bias of a given neuron with weighed input z,
        # dz/db = 1
        # dC/db = dC/dz * dz/db
        # dC/db = dC/dz * 1
        # dC/db = dC/dz
        # dC/db = delta (i.e. BP3)
        # apply this for all the neurons in a given layer to get nabla_b, the gradient of cost with
        # respect to the biases of the neurons in the current layer, the output layer
        nabla_b[-1] = delta

        # we can now calculate the gradient of cost with respect to the weights of the output layer
        # self.weights[l][j][k] is the weight of an input from k to j, but error (delta) propagates
        # backwards through the neural network, so we want the weight from j to k, which we can
        # find by transposing self.weights[l] (i.e. np.transpose(self.weights[l]))
        # because we're trying to found the gradient of the cost with respect to the weight of the
        # connections going to the output layer, -1, we need to find nabla_w[-1]
        # nabla_w[l][j][k] = a[l-1][k] x delta[l][j]
        # where a[l-1][k] is the vector of activations of the neurons in the previous layer
        nabla_w[-1] = np.dot([[d] for d in delta], [activations[-2]])

        for l in xrange(2, len(self.sizes)):
            z = zs[-l]
            zp = sigmoidprime(z) # z'
            delta = np.dot(np.transpose(self.weights[-l+1]), delta) * zp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot([[d] for d in delta], [activations[-l-1]])

        return (nabla_b, nabla_w)

        #/literate python

    def decision(self, inputs):
        output = self.feedforward(inputs)
        return 0 if output[0] > output[1] else 1

    def evaluate(self, test_data):
        cost = 0
        for x, y in test_data:
            output = self.feedforward(x)
            decision = (output[1] - output[0] + 1) / 2
            cost += (decision - y) ** 2
        return cost / len(test_data)

    def dCda(self, output_activations, y):
        return output_activations - y
