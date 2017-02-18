from NeuralNetwork import NeuralNetwork
import json

with open('stocks.json') as data_file:
	data = json.load(data_file)

days = data["YHOO"]
opens = [day["open"] for day in days]
net = NeuralNetwork(opens)
print(net.feedforward(opens[0:5]))
