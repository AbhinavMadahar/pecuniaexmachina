from NeuralNetwork import NeuralNetwork, chunk, arthmean
import json
import time
import datetime

from firebase import firebase
firebase = firebase.FirebaseApplication('https://pecuniaexmachina.firebaseio.com', None)

def recommendation(tick, query_highs):
    def timestamp(day, _):
        date = day[0].split("-")
        return int(time.mktime(datetime.date(int(date[0]), int(date[1]), int(date[2])).timetuple()))

    data = firebase.get('/stock/' + tick, None)
    highs = [(date, float(data[date])) for date in data]
    highs = map(lambda day: day[1], sorted(highs, timestamp))
    print highs

    input_size = 10
    testing_data = []
    weeks = chunk(highs, input_size)
    for i, week in enumerate(weeks[:-1]):
        y = 0 if max(weeks[i+1]) > max(week) else 1
        testing_data.append((week, y))

    net = NeuralNetwork(highs, [input_size, 5, 2])

    while sum(int(net.decision(x) == y) for x, y in testing_data) / float(len(testing_data)) < 0.5:
        net.train(10, 0.1)

    return net.decision(query_highs)
