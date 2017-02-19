# Pecuniaex Machina AI

This neural network analyzes the opening values of a stock over time to determine whether to buy, sell, or take no action to make the most money by next week.

## Design

The neural network is a simple feed-forward network with 1 hidden layer.

The input layer is a 10-vector in which each component is the opening price of the stock at that day of the week (i.e. v_1 is the opening value for Monday, ..., v_10 is the opening value for next Friday).

The output layer is a 2-vector, with the first component being the recommendation from 0 to 1 to buy (1 being a definite buy) and the second being the recommendation from 0 to 1 to sell (1 being a definite sell).

The neural network also has a `.decision` method that can take the 10-vector and return a 0 for buy and a 1 for sell.

## Training

The neural network is trained by being given the opening values for a given stock over a long duration.

## API

The NeuralNetwork accepts the opening values as its only constructor argument. It makes decisions based on 5 opening values, returning a list where the first value is a value in [0, 1] that recommends (1) or doesn't recommend (0) buying, and the second is the same but for selling.

```python
from NeuralNetwork import NeuralNetwork

net = NeuralNetwork([100, 102, 99.25, 104, ...])
past10days = [33, 34, 35, 36, 37, 33, 32, 31, 37, 34]

print net.feedforward(past10days)
print net.decision(past10days)
```

Alternatively, use the `recommendation` function from `main.py` to simplify the process:

```python
from main import recommendation

print recommendation("GOOG", [33, 34, 35, 36, 37, 33, 32, 31, 37, 34])
```
