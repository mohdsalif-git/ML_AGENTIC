import numpy as np

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

raw_scores = np.array([[  62.94, -122.32,   80.64,  -50.84],
                       [ 156.38, -236.53,  132.62, -121.82]])

print("Raw Inputs:          ", raw_scores)
print("ReLU Output (max(0, z)):", relu(raw_scores))
print("Sigmoid Output (0 to 1):", sigmoid(raw_scores).round(3))

