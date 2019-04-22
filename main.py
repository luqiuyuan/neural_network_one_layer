import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob

DIMENSION_INPUT = 49152
DIMENSION_HIDDEN_LAYER = 500
NUM_ITERATIONS = 100
m = 20 # Number of samples
alpha = 0.0001 # Learning rate
SIZE_TESTING = 10 # Size of testing dataset

# Read inputs
def readInputs(path):
    X = np.zeros((DIMENSION_INPUT, 0))
    for filename in glob.glob(path):
        img = mpimg.imread(filename)
        X = np.hstack((X, img.reshape((DIMENSION_INPUT, 1))))
    # Normalize X so that it is in range [0, 1]
    X = X / 256
    return X

# Read ground truth
def readGroundTruth(path):
    Y = np.zeros((1, 0))
    file = open(path, "r")
    for line in file:
        Y = np.hstack((Y, np.array([[int(line)]])))
    file.close()
    return Y

# Initialize parameters to all zeros
def initializeParametersZeros(dimension_input, dimension_hidden_layer):
    return (np.zeros((dimension_input, dimension_hidden_layer)), np.zeros((dimension_hidden_layer, 1)), np.zeros((dimension_hidden_layer, 1)), 0)

# Initialize parameters to small random numbers
def initializeParametersRandom(dimension_input, dimension_hidden_layer):
    return (np.random.random((dimension_input, dimension_hidden_layer)) * 0.001, np.zeros((dimension_hidden_layer, 1)),(np.random.random((dimension_hidden_layer, 1)) - 0.5) * 0.0001, 0)

# Activation function sigma
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Activation function ReLU
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    y = np.copy(x)
    y[x<0] = 0
    return y

# Forward propagation
def calculateForwardPropagatation(X, W1, B1, W2, B2):
    Z1 = W1.T.dot(X) + B1
    A1 = relu(Z1)
    Z2 = W2.T.dot(A1) + B2
    A2 = sigmoid(Z2)
    # print(Z2)
    return (Z1, A1, Z2, A2)

# Backward propagation
def calculateBackwardPropagation(X, Y, Z1, A1, A2, W2):
    dZ2 = A2 - Y
    dW2 = A1.dot(dZ2.T) / m
    dB2 = (np.sum(dZ2, axis=1) / m)[0]
    dZ1 = W2.dot(dZ2) * relu_derivative(Z1)
    dW1 = X.dot(dZ1.T) / m
    dB1 = (np.sum(dZ1, axis=1, keepdims=True) / m)
    return (dW2, dB2, dW1, dB1)

# Cost of a training iteration
def calculateCost(Y, A):
    return -(Y.dot(np.log(A.T)) + (1 - Y).dot(np.log(1 - A.T))) / m

######################
# Training phase
######################

# Read inputs and ground truth for training phase
X = readInputs("./dataset/training/*.jpg")
Y = readGroundTruth("./dataset/training/data.txt")

# (W1, B1, W2, B2) = initializeParametersZeros(DIMENSION_INPUT, DIMENSION_HIDDEN_LAYER)
(W1, B1, W2, B2) = initializeParametersRandom(DIMENSION_INPUT, DIMENSION_HIDDEN_LAYER)

# Training iteration
Js = np.zeros((0))
for i in range(0, NUM_ITERATIONS):
    (Z1, A1, Z2, A2) = calculateForwardPropagatation(X, W1, B1, W2, B2)
    J = calculateCost(Y, A2)[0, 0]
    Js = np.hstack((Js, np.array([J])))
    (dW2, dB2, dW1, dB1) = calculateBackwardPropagation(X, Y, Z1, A1, A2, W2)
    # print(dW1)
    print(J)
    W1 -= alpha * dW1
    B1 -= alpha * dB1
    W2 -= alpha * dW2
    B2 -= alpha * dB2
