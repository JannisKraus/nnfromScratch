import numpy as np
import matplotlib.pyplot as plt

# Z (linear hypothesis) - Z = W * X + b
# W - weight matrix, b - bias vector, X - Input

def sigmoid(Z):
    A = 1/(1 + np.exp(np.dot(-1, Z)))
    cache = (Z)

    return A, cache


def initParams(layerDims):
    np.random.seed(3)
    params = {}
    L = len(layerDims)

    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(layerDims[l], layerDims[l - 1]) * 0.01
        params['b' + str(l)] = np.zeros((layerDims[l], 1))
    
    return params

def forwardProp(X, params):
    A = X # input to first layer i.e. training data
    caches = []
    L = len(params) // 2
    for l in range(1, L+1):
        APrev = A

        # Linear Hypothesis
        Z = np.dot(params['W' + str(l)], APrev) + params['b' + str(l)]

        # Storing linear cache
        linearCache = (APrev, params['W' + str(l)], params['b' + str(l)])

        # Applying sigmoid on linear hypothesis
        A, activationCache = sigmoid(Z)

        # storing both linear and activation cache
        cache = (linearCache, activationCache)
        caches.append(cache)
    
    return A, caches

def costFunction(A, Y):
    m = Y.shape[1]

    cost = (-1 / m) * (np.dot(np.log(A), Y.T) + np.dot(np.log(1-A), 1 - Y.T))

    return cost

def oneLayerBackward(dA, cache):
    linearCache, activationCache = cache

    Z = activationCache
    dZ = dA * sigmoid(Z) * (1 - sigmoid(Z)) # The derivative of the sigmoid function

    APrev, W, b = linearCache
    m = APrev.shape[1]

    dW = (1 / m) * np.dot(dZ, APrev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dAPrev = np.dot(W.T, dZ)

    return dAPrev, dW, db

def backProp(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    currentCache = caches[L - 1]
    grads['dA' + str(L - 1)], grads['dW' + str(L - 1)], grads['db' + str(L - 1)] = oneLayerBackward(dAL, currentCache)

    for l in reversed(range(L - 1)):
        currentCache = caches[l]
        dAPrevTemp, dWTemp, dbTemp = oneLayerBackward(grads['dA' + str(l+1)], currentCache)
        grads["dA" + str(l)] = dAPrevTemp
        grads["dW" + str(l + 1)] = dWTemp
        grads["db" + str(l + 1)] = dbTemp

    return grads

def updateParameters(parameters, grads, learningRate):
    L = len(parameters) // 2

    for l in range(L):
        parameters['W' + str(l + 1)] = parameters['W' + str(l+1)] - learningRate * grads['W' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l+1)] -  learningRate * grads['b' + str(l + 1)]
    
    return parameters

def train(X, Y, layerDims, epochs, lr):
    params = initParams(layerDims)
    costHistory = []

    for i in range(epochs):
        YHat, caches = forwardProp(X, params)
        cost = costFunction(YHat, Y)
        costHistory.append(cost)
        grads = backProp(YHat, Y, caches)

        params = updateParameters(params, grads, lr)

    return params, costHistory