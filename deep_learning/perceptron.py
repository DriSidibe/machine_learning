# This script is the base code for a perceptron made by Drissa Sidibe

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class Perceptron:
  # Base class for a perceptron
    
  def __init__(self, x: np.ndarray, y: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, learning_rate=0.1, epsilon=0, compute_with_test=False):
    self.w = np.random.randn(x.shape[1], 1)
    self.b = np.random.randn(1)
    self.x = x
    self.y = y
    self.x_test = x_test
    self.y_test = y_test
    self.Z = self.computeZ()
    self.A = self.sigmoid()
    self.logLoss = []
    self.alpha = learning_rate
    self.epsilon = epsilon
    self.train_accuracy = []
    self.test_accuracy = []
    self.compute_with_test = compute_with_test
  
  def computeZ(self, p: np.ndarray = None):
    if p is not None:
      self.Z = (p.dot(self.w) + self.b)
      return self.Z
    self.Z = (self.x.dot(self.w) + self.b)
    return self.Z
    
  def sigmoid(self):
    A = 1 / (1 + np.exp(-self.computeZ()))
    return A
  
  def computeLogloss(self):
    log_loss = -(1/self.y.size) * np.sum(self.y * np.log(self.sigmoid() + self.epsilon) + (1-self.y) * np.log(1-self.sigmoid() + self.epsilon))
    self.logLoss.append(log_loss)
    return log_loss
  
  def computeW_b(self):
    w_gradient = (1 / self.y.size) * self.x.transpose().dot(self.sigmoid() - self.y)
    b_gradient = (1 / self.y.size) * np.sum(self.sigmoid() - self.y)
    _w = self.w - self.alpha * w_gradient
    _b = self.b - self.alpha * b_gradient
    self.w, self.b = _w, _b
    self.computeAverage()
    return (_w, _b)
  
  def computeAverage(self, x=None, y=None):
    if x is not None and y is not None:
      prediction_res = self.predict(x)
      acc = accuracy_score(y, prediction_res)
      self.test_accuracy.append(acc)
      return acc
    else:
      prediction_res = self.predict(self.x)
      acc = accuracy_score(self.y, prediction_res)
      self.train_accuracy.append(acc)
      return acc
  
  def predict(self, x):
    A = self.computeZ(x)
    return A >= 0.5
  
  def plotLoss(self):
    plt.plot(self.logLoss[:])
    plt.show()
    
  def plotTrainTestAcurracy(self):
    plt.plot(self.train_accuracy, c='r')
    plt.plot(self.test_accuracy, c='b')
    plt.show()
    
  def plotDatasetAndDecisionBound(self):
    x = np.linspace(-1, 4, 100)
    y = (-(self.w[0]*x) - self.b)/self.w[1]
    plt.scatter(self.x[:, 0], self.x[:, 1], c=self.y)
    plt.scatter(2, 1, c='r')
    plt.plot(x ,y)
    plt.show()

