import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class neuralNetwork:

    def __init__(self, trainPath, testPath, layerSizes):

        self.layerSizes = layerSizes

        #get training data
        trainData = pd.read_csv(trainPath)
        trainData = np.array(trainData)
        rows, columns = trainData.shape
        np.random.shuffle(trainData)

        testData = trainData[0:2000].T
        self.testLabels = testData[0]
        testPixels = testData[1:columns + 1]
        self.testPixels = testPixels / 255

        trainData = trainData[2000:rows].T
        self.trainLabels = trainData[0]
        trainPixels = trainData[1:columns + 1]
        self.trainPixels = trainPixels / 255

        #get testing data
        testData = pd.read_csv(testPath)
        testPixels = np.array(testData).T
        self.testDataPixels = testPixels / 255

        #initialsie network in random state
        self.params = {}
        numberOfLayers = len(layerSizes) - 1
        for l in range(1, numberOfLayers + 1):
            self.params[f"W{l}"] = np.random.rand(layerSizes[l], layerSizes[l - 1]) - 0.5
            self.params[f"b{l}"] = np.random.rand(layerSizes[l], 1) - 0.5
        self.params['layerSizes'] = layerSizes

    def gradientDescent(self, alpha, iterations):
        '''
        function that trains the network
        also prints out and saves progress values as network is trained
        '''
        numberOfLayers = len(self.layerSizes) - 1
        params = self.params

        X = self.trainPixels
        Y = self.trainLabels

        iteration = []
        accuracy = []

        for i in range(iterations):
            cache = forwardProp(X, params, numberOfLayers)
            grads = backwardProp(params, cache,X, Y, numberOfLayers)
            params = updateParams(params, grads, alpha, numberOfLayers)

            if i % 10 == 0:
                AL = cache[f"A{numberOfLayers}"]
                predictions = getPredictions(AL)
                print(f"Iteration: {i}, Accuracy: {getAccuracy(predictions, Y)}")
                iteration.append(i)
                accuracy.append(getAccuracy(predictions, Y))

        self.params = params
        return params, iteration, accuracy

    def testPrediction(self, index):
        '''
        gets prediction for chosen image
        also displays the image
        '''
        current_image = self.trainPixels[:, index, None]
        prediction = makePredictions(self.trainPixels[:, index, None], self.params, self.layerSizes)
        label = self.trainLabels[index]
        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.title(f'Prediction : {prediction} , Label : {label}')
        plt.show()


'''
some basic functions that will be used in forward and backward propagation
'''
def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def ReluDeriv(Z):
    return Z > 0

def desiredOutputs(Y):
    desiredOut = np.zeros((Y.size, Y.max() + 1))
    desiredOut[np.arange(Y.size), Y] = 1
    desiredOut = desiredOut.T
    return desiredOut

def forwardProp(pixels, params, numberOfLayers):
    '''
    forward propagation algorithm
    '''
    cache = {"A0": pixels}
    A_prev = pixels

    for l in range(1, numberOfLayers):
        Z = params[f"W{l}"].dot(A_prev) + params[f"b{l}"]
        A = ReLU(Z)
        cache[f"Z{l}"] = Z
        cache[f"A{l}"] = A
        A_prev = A

    ZL = params[f"W{numberOfLayers}"].dot(A_prev) + params[f"b{numberOfLayers}"]
    AL = softmax(ZL)
    cache[f"Z{numberOfLayers}"] = ZL
    cache[f"A{numberOfLayers}"] = AL
    return cache

def backwardProp(params, cache, pixels, labels, numberOfLayers):
    '''
    backward propagation algorithm
    '''
    grads = {}
    desiredOut = desiredOutputs(labels)
    AL = cache[f"A{numberOfLayers}"]

    dZL = AL - desiredOut
    grads[f"dW{numberOfLayers}"] = 1 / pixels.shape[1] * dZL.dot(cache[f"A{numberOfLayers - 1}"].T)
    grads[f"db{numberOfLayers}"] = 1 / pixels.shape[1] * np.sum(dZL, axis=1, keepdims=True)

    dA_prev = params[f"W{numberOfLayers}"].T.dot(dZL)

    for l in range(numberOfLayers - 1, 0, -1):
        dZ = dA_prev * ReluDeriv(cache[f"Z{l}"])
        grads[f"dW{l}"] = 1 / pixels.shape[1] * dZ.dot(cache[f"A{l - 1}"].T)
        grads[f"db{l}"] = 1 / pixels.shape[1] * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = params[f"W{l}"].T.dot(dZ)

    return grads

def updateParams(params, grads, alpha, numberOfLayers):
    '''
    updates parameters dependant on user chosen alpha value
    '''
    for l in range(1, numberOfLayers + 1):
        params[f"W{l}"] -= alpha * grads[f"dW{l}"]
        params[f"b{l}"] -= alpha * grads[f"db{l}"]
    return params

'''
below is a final set of functions that can be used to test the network
'''
def getPredictions(AL):
    '''
    returns index of largest output of final layer
    '''
    return np.argmax(AL, 0)

def getAccuracy(predictions, Y):
    '''
    returns what percentage of correct predictions were made
    '''
    return np.sum(predictions == Y) / Y.size

def makePredictions(pixels, params, layerSizes):
    '''
    gets predictions for single image
    also works for a matrix of stacked images
    '''
    numberOfLayers = len(layerSizes) - 1
    cache = forwardProp(pixels, params, numberOfLayers)
    predictions = getPredictions(cache[f"A{numberOfLayers}"])
    return predictions




