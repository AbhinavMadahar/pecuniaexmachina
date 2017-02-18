from NeuralNetwork import NeuralNetwork
import json
from pprint import pprint

with open('stocks.json') as data_file:
    data = json.load(data_file)

weekly_changes = {}
for tick in data:
    weekly_changes[tick] = []
    for days in 
    days = data[tick]
    net = NeuralNetwork(tick, days, [len(days), 5, 1])
    net.SGD(30, len(days) / 100, 2)

def geomean(nums):
    return (reduce(lambda x, y: x*y, nums))**(1.0/len(nums))