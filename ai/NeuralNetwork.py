import numpy as np
from collections import namedtuple
import random

TradingDay = namedtuple("TradingDay", ["high", "low", "open", "close", "date"])

dtanh = lambda z: 1.0 - np.tanh(z)**2

prod = lambda iterable: reduce(lambda a, b: a * b, iterable, 1)
geomean = lambda data: prod(data) ** (1 / len(data))

class NeuralNetwork(object):
    # stock_tick is something like "APPL" that is used to look up stocks
    # trading_days is an array of TradingDay objects
    # sizes is an array whose indeces are the layers and values at each index is the length of the
    # layer in terms of neurons
    def __init__(self, stock_tick, trading_days, sizes):
        self.stock_tick = stock_tick
        self.trading_days = trading_days

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        greatest_weekly_percent_change = -float("inf")
        for k in xrange(len(trading_days) / 5):
            daily_percent_changes = [trading_days[k+i+1]["open"] / trading_days[k+i]["open"] - 1 for i in xrange(0, 5)]
            weekly_percent_change = geomean(daily_percent_changes)
            greatest_weekly_percent_change = max([weekly_percent_change, greatest_weekly_percent_change])

        least_weekly_percent_change = float("inf")
        for k in xrange(len(trading_days) / 5):
            daily_percent_changes = [trading_days[k+i+1]["open"] / trading_days[k+i]["open"] - 1 for i in xrange(0, 5)]
            weekly_percent_change = geomean(daily_percent_changes)
            least_weekly_percent_change = min([weekly_percent_change, least_weekly_percent_change])

        self.greatest_weekly_percent_change = greatest_weekly_percent_change
        self.least_weekly_percent_change = least_weekly_percent_change

    def fit_to_range(self, weekly_percent_change):
        increase = weekly_percent_change / self.greatest_weekly_percent_change
        return increase + self.least_weekly_percent_change

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = np.tanh(np.dot(w, a) + b)
        return a

    def SGD(self, epochs, mini_batch_size, learning_rate):
        n = len(self.trading_days)
        for _ in xrange(epochs): # for epochs number of times
            random.shuffle(self.trading_days)

            # group the training data into batches of length
            mini_batches = [self.trading_days[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.learn_from_mini_batch(mini_batch)

    # mini_batch is an array of arrays of openning values for a given stock
    def learn_from_mini_batch(self, mini_batch):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        opens = [day["open"] for day in mini_batch]
        daily_percent_changes = [opens[k+1] / day - 1 for k, day in enumerate(opens[:-1])]
        weekly_percent_change = geomean(daily_percent_changes)
        y = self.fit_to_range(weekly_percent_change)

        delta_nabla_b, delta_nabla_w = self.backprop(opens, y)
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
            activation = np.tanh(z)
            activations.append(activation)

        # now go back
        error = self.dCdx(activations[-1], y)
        delta = dtanh(zs[-1]) * error

        nabla_b[-1] = delta # BP3
        nabla_w[-1] = np.dot(delta, np.transpose(activations[-2])) # BP2

        for l in xrange(1, self.num_layers - 1):
            z = zs[-l]
            zp = dtanh(z) # z'
            trans = np.transpose(self.weights[-l+1])
            assert trans.shape == (5,5)
            #delta = np.dot(np.transpose(self.weights[-l+1]), delta) * zp
            delta = np.dot(trans, delta) * zp
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

    def dCdx(self, output_activations, y):
        return [a - y for a in output_activations]