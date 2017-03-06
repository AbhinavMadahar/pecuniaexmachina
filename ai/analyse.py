# determine how accurate the neural network is

import json
from multiprocessing import Process
import time
from NeuralNetwork import NeuralNetwork, chunk

# these can be configured to change the way the neural network runs
input_size = 10
hidden_layers_sizes = [20]
epochs = 10
learning_rate = 0.1

max_processing_time = 20 # seconds

# load stock data from the past year of the S&P 500 companies
print "Reading HSD file..."
with open("hsd.json") as historical_stock_data_file:
    sp500 = json.load(historical_stock_data_file)["data"]
print "Done"

print "Testing stocks"
for stock in sp500:
    def process():
        ticker = stock[0]["Symbol"]
        print "Testing", ticker
    
        historical_highs = [float(day["High"]) for day in stock]
    
        # generate the training data on the past year excluding the past input_size days
        testing_data = []
        chunks = chunk(historical_highs[:-2 * input_size], input_size)
        for i, highs in enumerate(chunks[:-1]):
            y = 0 if max(chunks[i+1]) > max(highs) else 1
            testing_data.append((highs, y))
    
        net = NeuralNetwork(historical_highs[:-input_size], [input_size] + hidden_layers_sizes + [2])
    
        while sum(int(net.decision(x) == y) for x, y in testing_data) / float(len(testing_data)) < 0.5:
            net.train(epochs, learning_rate)
    
        decision = net.decision(historical_highs[-2*input_size:-input_size])
        correct_answer = int(max(historical_highs[-2*input_size:-input_size]) > max(historical_highs[-input_size:]))
        
        print ticker, "was decided", "correctly" if decision == correct_answer else "incorrectly"

    processor = Process(name='process', target=process)
    processor.start()
    # a process that takes longer than max_processing_time seconds is probably broken
    processor.join(timeout=max_processing_time)
    processor.terminate()
