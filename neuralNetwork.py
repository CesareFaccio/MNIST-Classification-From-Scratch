import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class initialise:

    def getDataset(filePath):
        '''
        gets data for training and splits a part for testing
        '''
        trainData = pd.read_csv(filePath)
        trainData = np.array(trainData)
        rows, columns = trainData.shape
        np.random.shuffle(trainData)

        testData = trainData[0:2000].T
        testLabels = testData[0]
        testPixels = testData[1:columns + 1]
        testPixels = testPixels / 255

        trainData = trainData[2000:rows].T
        trainLabels = trainData[0]
        trainPixels = trainData[1:columns + 1]
        trainPixels = trainPixels / 255

        return trainLabels, trainPixels, testLabels, testPixels

    def getTestDataset(filePath):
        '''
        gets data for creating a submission for kaggle competition
        '''
        testData = pd.read_csv(filePath)
        testPixels = np.array(testData).T
        testPixels = testPixels / 255

        return testPixels

    def initParams(layerSizes):
        '''
        initialises the network in a random state
        '''
        params = {}
        numberOfLayers = len(layerSizes) - 1
        for l in range(1, numberOfLayers + 1):
            params[f"W{l}"] = np.random.rand(layerSizes[l], layerSizes[l - 1]) - 0.5
            params[f"b{l}"] = np.random.rand(layerSizes[l], 1) - 0.5
        return params

class functions:

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

class propagation:

    def forwardProp(pixels, params, numberOfLayers):
        '''
        forward propagation algorithm
        '''
        cache = {"A0": pixels}
        A_prev = pixels

        for l in range(1, numberOfLayers):
            Z = params[f"W{l}"].dot(A_prev) + params[f"b{l}"]
            A = functions.ReLU(Z)
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A
            A_prev = A

        ZL = params[f"W{numberOfLayers}"].dot(A_prev) + params[f"b{numberOfLayers}"]
        AL = functions.softmax(ZL)
        cache[f"Z{numberOfLayers}"] = ZL
        cache[f"A{numberOfLayers}"] = AL
        return cache

    def backwardProp(params, cache, pixels, labels, numberOfLayers):
        '''
        backward propagation algorithm
        '''
        grads = {}
        desiredOut = functions.desiredOutputs(labels)
        AL = cache[f"A{numberOfLayers}"]

        dZL = AL - desiredOut
        grads[f"dW{numberOfLayers}"] = 1 / pixels.shape[1] * dZL.dot(cache[f"A{numberOfLayers - 1}"].T)
        grads[f"db{numberOfLayers}"] = 1 / pixels.shape[1] * np.sum(dZL, axis=1, keepdims=True)

        dA_prev = params[f"W{numberOfLayers}"].T.dot(dZL)

        for l in range(numberOfLayers - 1, 0, -1):
            dZ = dA_prev * functions.ReluDeriv(cache[f"Z{l}"])
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

class evaluate:

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
        cache = propagation.forwardProp(pixels, params, numberOfLayers)
        predictions = evaluate.getPredictions(cache[f"A{numberOfLayers}"])
        return predictions

    def testPrediction(trainLabels, trainPixels, index, params, layerSizes):
        '''
        gets prediction for chosen image
        also plots the image
        '''
        current_image = trainPixels[:, index, None]
        prediction = evaluate.makePredictions(trainPixels[:, index, None], params, layerSizes)
        label = trainLabels[index]
        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.title(f'Prediction : {prediction} , Label : {label}')
        plt.show()

class neuralNetwork:

    def gradientDescent(X,Y, alpha, iterations, layerSizes):
        '''
        function that trains the network
        also prints out and saves progress values as network is trained
        '''
        numberOfLayers = len(layerSizes) - 1
        params = initialise.initParams(layerSizes)

        iteration = []
        accuracy = []

        for i in range(iterations):
            cache = propagation.forwardProp(X, params, numberOfLayers)
            grads = propagation.backwardProp(params, cache,X, Y, numberOfLayers)
            params = propagation.updateParams(params, grads, alpha, numberOfLayers)

            if i % 10 == 0:
                AL = cache[f"A{numberOfLayers}"]
                predictions = evaluate.getPredictions(AL)
                print(f"Iteration: {i}, Accuracy: {evaluate.getAccuracy(predictions, Y)}")
                iteration.append(i)
                accuracy.append(evaluate.getAccuracy(predictions, Y))

        return params, iteration, accuracy