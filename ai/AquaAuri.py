import json, urllib2, time
from datetime import datetime
from time import mktime

from NN import NeuralNetwork

def normalizePrice(price, minimum, maximum):
	return ((2*price - (maximum + minimum)) / (maximum - minimum))

def denormalizePrice(price, minimum, maximum):
	return (((price*(maximum-minimum))/2) + (maximum + minimum))/2

def rollingWindow(seq, windowSize):
	it = iter(seq)
	win = [it.next() for cnt in xrange(windowSize)]
	yield win
	for e in it:
		win[:-1] = win[1:]
		win[-1] = e
		yield win

def getMovingAverage(values, windowSize):
	movingAverages = []
	
	for w in rollingWindow(values, windowSize):
		movingAverages.append(sum(w)/len(w))

	return movingAverages

def getMinimums(values, windowSize):
	minimums = []

	for w in rollingWindow(values, windowSize):
		minimums.append(min(w))
			
	return minimums

def getMaximums(values, windowSize):
	maximums = []

	for w in rollingWindow(values, windowSize):
		maximums.append(max(w))

	return maximums

def getTimeSeriesValues(values, window):
	movingAverages = getMovingAverage(values, window)
	minimums = getMinimums(values, window)
	maximums = getMaximums(values, window)

	returnData = []

	for i in range(0, len(movingAverages)):
		inputNode = [movingAverages[i], minimums[i], maximums[i]]
		price = normalizePrice(values[len(movingAverages) - (i + 1)], minimums[i], maximums[i])
		outputNode = [price]
		tempItem = [inputNode, outputNode]
		returnData.append(tempItem)

	return returnData

def getHistoricalData(stockSymbol):
	historicalPrices = []
	
	urllib2.urlopen("http://api.kibot.com/?action=login&user=guest&password=guest")

	url = "http://api.kibot.com/?action=history&symbol=" + stockSymbol + "&interval=daily&period=15&unadjusted=1&regularsession=1"
	apiData = urllib2.urlopen(url).read().split("\n")
	for line in apiData:
		if(len(line) > 0):
			tempLine = line.split(',')
			price = float(tempLine[1])
			historicalPrices.append(price)

	return historicalPrices

def getTrainingData(stockSymbol):
	historicalData = getHistoricalData(stockSymbol)

	#historicalData.reverse()
	del historicalData[10:]

	trainingData = getTimeSeriesValues(historicalData, 5)

	return trainingData

def getPredictionData(stockSymbol, historicalData):
	historicalData.reverse()
	del historicalData[5:]

	predictionData = getTimeSeriesValues(historicalData, 5)
	predictionData = predictionData[0][0]

	return predictionData

def analyzeSymbol(stockSymbol):
	historicalData = getHistoricalData(stockSymbol)
	startTime = time.time()
	
	trainingData = getTrainingData(stockSymbol)
	
	network = NeuralNetwork(inputNodes = 3, hiddenNodes = 10, outputNodes = 1)
	i = 0
	while (i <= 5):
		count = 0
		network.train(trainingData)
		i += 1
	while (count <= 5):
		predictionData = getPredictionData(stockSymbol, historicalData)
		if count < 5:
			historicalData += predictionData
		count += 1

	returnPrice = network.test(predictionData)

	predictedStockPrice = denormalizePrice(returnPrice, predictionData[1], predictionData[2])

	returnData = {}
	returnData['price'] = predictedStockPrice
	returnData['time'] = time.time() - startTime

	return returnData

if __name__ == "__main__":
	print analyzeSymbol("AAPL")