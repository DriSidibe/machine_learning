from perceptron import Perceptron
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples=100, n_features=5, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

# Perceptron test for Iris dataset
p = Perceptron(x, y)
iteration_count = 100

for i in range(iteration_count):
    p.computeLogloss()
    p.computeW_b()
    
print("Learning finish!")
print(p.computeAverage())

p.plotLoss()
#p.plotDatasetAndDecisionBound()