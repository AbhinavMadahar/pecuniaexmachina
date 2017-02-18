from NeuralNetwork import NeuralNetwork
import json

with open('stocks.json') as data_file:
    data = json.load(data_file)

weekly_changes = {}
for tick in data:
    weekly_changes[tick] = []
    days = data[tick]
    net = NeuralNetwork(tick, days, [5, 5, 1])
    net.SGD(30, 5, 2)
