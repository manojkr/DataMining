from __future__ import division
from collections import defaultdict
import json
import os.path
import sys

from Classifier import Classifier
from Common import DataSet


class BayesianClassifier(Classifier):
    """ Naive Bayes Classifier for classification

    Attributes -
    training_data - DataSet object corresponding to input data set
    """

    def __init__(self, training_data):
        Classifier.__init__(self, training_data)
        # dictionary to count the occurrences of attributes
        self.attr_counter = defaultdict(lambda: (defaultdict(lambda: defaultdict(int))))
        self.label_counter = defaultdict(int)
        self.max_feature_per_index = defaultdict(int)

    def train_model(self):
        """ Trains the BayesianClassifier model
        using given training DataSet
        """
        for tuple in self.training_data:
            label = tuple.label
            attrs = tuple.attrs
            # Increment the tuple count for this label
            self.label_counter[label] += 1
            for index, value in attrs.iteritems():
                self.attr_counter[label][index][value] += 1
                self.max_feature_per_index[index] = max(value, self.max_feature_per_index[index])

        #laplacian smoothing to account for unseen feature values
        #Basically we are assuming 1 tuple for each feature
        for c in ["+1", "-1"]:
            for (index, value) in self.max_feature_per_index.iteritems():
                for v in range(1, value + 1):
                    self.attr_counter[c][index][v] += 1
                    self.label_counter[c] += 1

    def calculate_prob(self, class_label, tuple):
        """ Calculate probability that a tuple belongs to a given class label
        """
        prob = 1.0
        # Number of tuples in training data set with the given label
        count_tuples_with_given_label = self.label_counter[class_label]
        # Prior Probability of a given label
        prob_label = count_tuples_with_given_label / len(self.training_data)
        # Intermediate Probability value
        prob = prob * prob_label
        # Calculate probability for each tuple attribute
        for index, value in tuple.attrs.iteritems():
            #This check is for detecting Unseen Features.
            #IF you see UNSEEN Features in Test Dataset that
            #you have not seen in training data set, ignore
            #such features
            if index in self.max_feature_per_index:
                prob_attr = self.attr_counter[class_label][index][value] / count_tuples_with_given_label
                if prob_attr == 0:
                    #In rare cases, some feature values would have escaped laplace smoothing
                    # because their value is above the maximum feature value encountered in test
                    # data. Ignore such featues
                    continue
                prob = prob * prob_attr
        return prob

    def classify_tuple(self, tuple):
        """ Classify the tuple as one of the classes - +1 or -1
        """
        #Probability that the label belongs to +1 class
        prob_plus_one = self.calculate_prob("+1", tuple)
        #Probability that the label belongs to -1 class
        prob_minus_one = self.calculate_prob("-1", tuple)
        label = "+1" if prob_plus_one > prob_minus_one else "-1"
        return label

    def __repr__(self):
        """ String representation of the inner state of Bayesian Classifier
        Useful for debugging purposes
        """
        return json.dumps(self.attr_counter, sort_keys=True, indent=4) + "\n" + json.dumps(self.label_counter,
                                                                                           sort_keys=True, indent=4)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Command Format is ./NaiveBayes.py <<training_file>> <<test_file>>"
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    #Read the training data set into memory
    trainingDataSet = DataSet(os.path.abspath(train_file))
    trainingDataSet.readData()
    # Train the Bayesian Classifier
    classifier = BayesianClassifier(trainingDataSet)
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