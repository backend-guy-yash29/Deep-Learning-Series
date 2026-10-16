import numpy as np
import pandas as pd

class Perceptron:
  def __init__(self, dataset):
    self.inp = dataset.drop('Target', axis=1).copy()
    self.inp['bias'] = np.ones(len(self.inp), dtype='int')
    self.out = dataset['Target']
    self.weights = []

  def initialize_weights(self):
    self.weights = np.array([-1.0] * len(self.inp.columns))

  def step_fun(self, value):
    return 1 if value >= 0 else 0

  def fit(self, epochs=1000, lr=0.01):
    self.initialize_weights()
    n = len(self.inp)

    for epoch in range(epochs):
      for i in range(n):
        x_i = self.inp.iloc[i].values
        y_true = self.out.iloc[i]
        y_pred = self.step_fun(self.weights @ x_i)
        self.weights = self.weights + (y_true - y_pred) * lr * x_i

    return self.weights
