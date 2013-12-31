from __future__ import division
import os.path
from math import log

from numpy.random import choice
from NaiveBayes import BayesianClassifier
from Classifier import Classifier
from Common import DataSet
import sys

class AdaBoostClassifier(Classifier):
    """ AdaBoostClassifer that uses BayesianClassifier
    as weak classifier
    Attributes-
    training_data - Input DataSe
    nclassifiers - Number of weak classifiers to generate
    """

    def __init__(self, training_data, nclassifiers=2):
        Classifier.__init__(self, training_data)
        self.nclassifiers = nclassifiers
        self.classifiers = []
        self.classifier_weight = []

    def train_model(self):
        ntuples = len(self.training_data)
        # An array to store weight for each tuple.
        # Initially all the weights are equal
        weight_per_tuple = [1 / ntuples] * ntuples
        for i in range(0, self.nclassifiers):
            classifier = None
            error = None
            # Keep on looping unless you find a classifier with error less than 50%
            # This is to tackle the case when a randomly selected sample performs
            # very bad. Do max 7 reattempts
            max_allowed_tries = 7
            tries = 0
            while True:
                # Use Numpy library to generate a Sample With Replacement with probability of each
                # tuple proportional to its weight
                sample = choice(self.training_data.tuples, ntuples, p=weight_per_tuple, replace=True).tolist()
                # Train a bayesian classifer for this sample
                classifier = BayesianClassifier(sample)
                classifier.train_model()
                error = 0.0
                # an array to store if the predicted values are correct
                is_prediction_correct = [True] * ntuples
                for i, tuple in enumerate(self.training_data):
                    predicted = classifier.classify_tuple(tuple)
                    actual = tuple.label
                    if predicted != actual:
                        error += weight_per_tuple[i]
                        is_prediction_correct[i] = False
                sum_of_old_weights = sum(weight_per_tuple)
                # Adjust the weights for all the tuples whose classification was correct
                for i, is_correct in enumerate(is_prediction_correct):
                    if is_correct:
                        weight_per_tuple[i] = (weight_per_tuple[i] * error) / (1 - error)
                sum_of_new_weights = sum(weight_per_tuple)
                # Normalize the weights so that sum is one
                weight_per_tuple = [(w * sum_of_old_weights) / sum_of_new_weights for w in weight_per_tuple]
                tries += 1
                if error < 0.5 or tries > max_allowed_tries:
                    break
            self.classifiers.append(classifier)
            self.classifier_weight.append(log((1 - error) / error, 2))

    def classify_tuple(self, tuple):
        """ Classify the given tuple as one of the two classes +1 or -1
        """
        weights = {"+1": 0, "-1": 0}
        #Get the vote of each classifier
        for i in range(0, self.nclassifiers):
            predicted_label = self.classifiers[i].classify_tuple(tuple)
            weights[predicted_label] += self.classifier_weight[i]
        # return the class label with maximum weight
        return max(weights.keys(), key=lambda x: weights[x])


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Command Format is ./NaiveBayes.py <<training_file>> <<test_file>>"
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    #Read the training data set into memory
    trainingDataSet = DataSet(os.path.abspath(train_file))
    trainingDataSet.readData()
    # Train the Bayesian Classifier
    classifier = AdaBoostClassifier(trainingDataSet, nclassifiers=2)
    classifier.train_model()
    # Classify the Training Data Set
    predictions = classifier.classify(trainingDataSet)
    metrics = classifier.calculate_metrics(trainingDataSet, predictions)
    #true positive in training, false negative in training, false positive in training and true negative in training
    print "%d %d %d %d" %(metrics["tp"], metrics["fn"], metrics["fp"], metrics["tn"])
    #Read the test data set into memory
    testDataSet = DataSet(os.path.abspath(test_file))
    testDataSet.readData()
    # Classify the Test Data Set
    predictions = classifier.classify(testDataSet)
    metrics = classifier.calculate_metrics(testDataSet, predictions)
    print "%d %d %d %d" %(metrics["tp"], metrics["fn"], metrics["fp"], metrics["tn"])
