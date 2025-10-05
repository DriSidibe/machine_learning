from perceptron import Perceptron
import numpy as np
import matplotlib.pyplot as plt
import utilities
from PIL import Image
from tqdm import tqdm

X_train, y_train, X_test, y_test = utilities.load_data()
#image_array = np.array(X_train[0])
#plt.imshow(image_array)

X_flatten = np.zeros((X_train.shape[0], X_train[0].flatten().size))
X_test_flatten = np.zeros((X_test.shape[0], X_test[0].flatten().size))

j = 0
for i in X_train:
    X_flatten[j] = i.flatten()
    j += 1
j = 0
for i in X_test:
    X_test_flatten[j] = i.flatten()
    j += 1

X_flatten = X_flatten / 255
X_test_flatten = X_test_flatten / 255

# Perceptron test for Chien chat dataset
p = Perceptron(X_flatten, y_train, X_test, y_test, epsilon=1e-15, learning_rate=0.01)
iteration_count = 6000

for i in tqdm(range(iteration_count)):
    p.computeLogloss()
    p.computeW_b()
    
print("Learning finish!")

p.plotLoss()
p.plotTrainTestAcurracy()