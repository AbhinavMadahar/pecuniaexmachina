from NeuralNetwork import NeuralNetwork, chunk, arthmean
import json
import time
import datetime

def recommendation(historical_highs, recent, hidden_layer_sizes, epochs, learning_rate):
    input_size = len(recent)
    testing_data = []
    chunks = chunk(historical_highs, input_size)
    for i, week in enumerate(weeks[:-1]):
        y = 0 if max(weeks[i+1]) > max(week) else 1
        testing_data.append((week, y))

    net = NeuralNetwork(highs, [input_size] + hidden_layer_sizes + [2])

    while sum(int(net.decision(x) == y) for x, y in testing_data) / float(len(testing_data)) < 0.5:
        net.train(epochs, learning_rate)

    return net.decision(query_highs)
