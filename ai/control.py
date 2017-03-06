# determine how accurate a random selection is to act as a control

import json
from random import randint
from NeuralNetwork import chunk

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
    ticker = stock[0]["Symbol"]
    print "Testing", ticker

    historical_highs = [float(day["High"]) for day in stock]

    decision = randint(0, 1)
    correct_answer = int(max(historical_highs[-2*input_size:-input_size]) > max(historical_highs[-input_size:]))
    
    print ticker, "was decided", "correctly" if decision == correct_answer else "incorrectly"
