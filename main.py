from neural import neuralNetwork
from neural import makePredictions
from neural import getAccuracy
import matplotlib.pyplot as plt
import pickle
import numpy as np

def main():

    #initialse some values
    layerSizes = [784, 10, 10]
    trainPath = "train.csv"
    testPath = "test.csv"

    network = neuralNetwork( trainPath, testPath, layerSizes )
    params, iteration, accuracy = network.gradientDescent( 0.1, 100)

    plt.plot(iteration, accuracy, marker='o', color='b')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Iterations')
    plt.show()

    network.testPrediction(100)
    results = makePredictions(network.testPixels, params, params['layerSizes'])
    print(f"model accuracy : {getAccuracy(results, network.testLabels)}")

    with open('params.pkl', 'wb') as file:
        pickle.dump(params, file)

    with open('params.pkl', 'rb') as file:
        loadedParams = pickle.load(file)

    testResults = makePredictions(network.testDataPixels, loadedParams, loadedParams['layerSizes'])
    imageIds = np.linspace(1, network.testDataPixels.shape[1], network.testDataPixels.shape[1])
    combined_array = np.column_stack((imageIds, testResults))
    np.savetxt('submission.csv', combined_array, delimiter=',', header='ImageId,Label', comments='', fmt='%d')

if __name__ == "__main__":
    main()