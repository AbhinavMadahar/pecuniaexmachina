from NeuralNetwork import NeuralNetwork, chunk, arthmean
import json

with open('stocks.json') as data_file:
	data = json.load(data_file)

days = data["PYPL"]
opens = [day["high"] for day in days]

input_size = 10
testing_data = []
weeks = chunk(opens, input_size)
for i, week in enumerate(weeks[:-1]):
    y = 0 if max(weeks[i+1]) > max(week) else 1
    testing_data.append((week, y))

net = NeuralNetwork(opens, [input_size, 5, 2])

while sum(int(net.decision(x) == y) for x, y in testing_data) / float(len(testing_data)) < 0.5:
    net.train(10, 0.1)

p_correct = sum(int(net.decision(x) == y) for x, y in testing_data) / float(len(testing_data))
print p_correct
