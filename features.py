# features.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import numpy as np
import util
import samples

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
closed_list = []

def basicFeatureExtractor(datum):
    """
    Returns a binarized and flattened version of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features indicating whether each pixel
            in the provided datum is white (0) or gray/black (1).
    """
    features = np.zeros_like(datum, dtype=int)
    features[datum > 0] = 1
    return features.flatten()

def enhancedFeatureExtractor(datum):
    """
    Returns a feature vector of the image datum.

    Args:
        datum: 2-dimensional numpy.array representing a single image.

    Returns:
        A 1-dimensional numpy.array of features designed by you. The features
            can have any length.

    ## DESCRIBE YOUR ENHANCED FEATURES HERE...

    ##
    """
    #features = basicFeatureExtractor(datum)
    features = np.zeros_like(datum, dtype=int)
    features[datum > 0] = 1
    #Calculating the number of rows and columns in input datum.
    number_rows, number_columns = datum.shape[0], datum.shape[1]
    #Initializing the number of white regions to zero to start with.
    count_white_regions = 0
    #Intialized the visited list or the closet list, with a matrix of size of features vector and initializing them to False
    closed_list = [[False for j in range(number_columns)] for i in range(number_rows)]
    # temp_features = features
    # if number_rows > 0 and number_columns > 0: return

    for i in range(number_rows):
        for j in range(number_columns):
            #Checking if the node was never visited and the value is 0.
            if closed_list[i][j] == False and features[i][j] == 0:
                #If yes, we perform DFS recursively on its VALID neighbors and we increment the number of connected components by 1.
                dfs(features, i, j, number_rows, number_columns, closed_list)
                count_white_regions += 1
    # Inferences from the number of white regions found.
    features_added = np.array([0, 0, 0])
    if count_white_regions == 1:
        features_added = np.array([1, 0, 0])
    elif count_white_regions == 2:
        features_added = np.array([0, 1, 0])
    elif count_white_regions > 2:
        features_added = np.array([0, 0, 1])
    return np.concatenate((features.flatten(), features_added), axis=0)


def dfs(features, i, j, k, l, closed_list):
    #Checking if the indices received are valid or not.
    if i < 0 or j < 0 or i >= k or j >= l or features[i][j] != 0 or closed_list[i][j] == True:
        return
    # temp_features[i][j] = 2
    #Marking the parameter is visited.
    closed_list[i][j] = True
    ##Recursive calls
    # dfs(features, i - 1, j - 1, k, l, closed_list)
    dfs(features, i - 1, j, k, l, closed_list)
    # dfs(features, i - 1, j + 1, k, l, closed_list)
    dfs(features, i + 1, j, k, l, closed_list)
    dfs(features, i, j + 1, k, l, closed_list)
    dfs(features, i, j - 1, k, l, closed_list)
    # dfs(features, i + 1, j + 1, k, l, closed_list)
    # dfs(features, i + 1, j - 1, k, l, closed_list)

    




def analysis(model, trainData, trainLabels, trainPredictions, valData, valLabels, validationPredictions):
    """
    This function is called after learning.
    Include any code that you want here to help you analyze your results.

    Use the print_digit(numpy array representing a training example) function
    to the digit

    An example of use has been given to you.

    - model is the trained model
    - trainData is a numpy array where each row is a training example
    - trainLabel is a list of training labels
    - trainPredictions is a list of training predictions
    - valData is a numpy array where each row is a validation example
    - valLabels is the list of validation labels
    - valPredictions is a list of validation predictions

    This code won't be evaluated. It is for your own optional use
    (and you can modify the signature if you want).
    """

    # Put any code here...
    # Example of use:
    # for i in range(len(trainPredictions)):
    #     prediction = trainPredictions[i]
    #     truth = trainLabels[i]
    #     if (prediction != truth):
    #         print "==================================="
    #         print "Mistake on example %d" % i
    #         print "Predicted %d; truth is %d" % (prediction, truth)
    #         print "Image: "
    #         print_digit(trainData[i,:])


## =====================
## You don't have to modify any code below.
## =====================

def print_features(features):
    str = ''
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    for i in range(width):
        for j in range(height):
            feature = i*height + j
            if feature in features:
                str += '#'
            else:
                str += ' '
        str += '\n'
    print(str)

def print_digit(pixels):
    width = DIGIT_DATUM_WIDTH
    height = DIGIT_DATUM_HEIGHT
    pixels = pixels[:width*height]
    image = pixels.reshape((width, height))
    datum = samples.Datum(samples.convertToTrinary(image),width,height)
    print(datum)

def _test():
    import datasets
    train_data = datasets.tinyMnistDataset()[0]
    for i, datum in enumerate(train_data):
        print_digit(datum)

if __name__ == "__main__":
    _test()
