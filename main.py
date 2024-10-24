from neuralNetwork import neuralNetwork
from neuralNetwork import initialise
from neuralNetwork import evaluate
import matplotlib.pyplot as plt
import pickle
import numpy as np

def main():

    #initialse some values
    layerSizes = [784, 30, 30, 10, 10]
    trainLabels, trainPixels, testLabels, testPixels= initialise.getDataset("train.csv")

    #train network
    params, iteration, accuracy = neuralNetwork.gradientDescent(trainPixels,trainLabels, 0.20, 100, layerSizes)
    params['layerSizes'] = layerSizes

    #visualise training progress
    plt.plot(iteration, accuracy, marker='o', color='b')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Iterations')
    plt.show()

    #test network on chosen image
    evaluate.testPrediction(trainLabels, trainPixels, 100, params, params['layerSizes'])

    #evaluate model using test portion of data
    testResults = evaluate.makePredictions(testPixels, params, params['layerSizes'])
    print(f"model accuracy : {evaluate.getAccuracy(testResults, testLabels)}")

    #save model param to reuse the model without the need of training it again
    with open('params.pkl', 'wb') as file:
        pickle.dump(params, file)

    #load model
    with open('params.pkl', 'rb') as file:
        loadedParams = pickle.load(file)

    #use loaded model to create a submission for kaggle competition
    submissionPixels = initialise.getTestDataset("test.csv")
    testResults = evaluate.makePredictions(submissionPixels, loadedParams, loadedParams['layerSizes'])
    imageIds = np.linspace(1, submissionPixels.shape[1], submissionPixels.shape[1])
    combined_array = np.column_stack((imageIds, testResults))
    np.savetxt('submission.csv', combined_array, delimiter=',', header='ImageId,Label', comments='', fmt='%d')

if __name__ == "__main__":
    main()