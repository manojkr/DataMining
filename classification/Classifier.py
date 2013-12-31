from __future__ import division
#!/usr/bin/env python

"""Classifier.py: Base Class for any Classifier

Classifer class provides common functionality that
any kind of classifier will need

"""

__author__ = "Manoj Kumar"


class Classifier:
    """ Classifer class provides common functionality that
    any kind of classifier will need.

    Attributes-
    training_data - Input DataSet represented as DataSet object
    """

    def __init__(self, training_data):
        self.training_data = training_data

    def classify(self, test_data):
        """ Classify all the tuples in test_data DataSet
        """
        predicted_values = []
        # Iterate over all the tuples in DataSet and invoke classify_tuple on each
        for tuple in test_data:
            #classify_tuple will be implemented by classes which inherit the Classifier
            label = self.classify_tuple(tuple)
            predicted_values.append(label)
        return predicted_values

    def calculate_metrics(self, test_data, predicted_labels):
        """ Given test_data containing actual class lebel and predicted_labels
        calculate different metrics e.g. Precision, Recall, Sensitivity, F-Measure
        """

        # Four variables for True Positive, True Negative, False Positive and False Negative
        tp, tn, fp, fn = 0, 0, 0, 0
        for i, tuple in enumerate(test_data):
            actual_label = tuple.label
            predicted_label = predicted_labels[i]
            if predicted_label == actual_label and predicted_label == "+1":
                tp += 1
            if predicted_label == actual_label and predicted_label == "-1":
                tn += 1
            if predicted_label != actual_label and predicted_label == "+1":
                fp += 1
            if predicted_label != actual_label and predicted_label == "-1":
                fn += 1

        #results dictionary to store all the metrics
        p = tp + fn
        n = tn + fp
        accuracy = (tp + tn ) / (p + n) if (p + n) > 0 else "NA"
        error_rate = (fp + fn) / (p + n) if (p + n) > 0 else "NA"
        recall = tp / p if p > 0 else "NA"
        precision = tp / (tp + fp) if (tp + fp) > 0 else "NA"
        specificity = tn / n if n > 0 else "NA"
        f1_score = (2 * precision * recall) / (precision + recall)
        f2_score = (5 * precision * recall) / ((4 * precision) + recall)
        f_point5_score = (1.25 * precision * recall) / ((.25 * precision) + recall)
        results = {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": accuracy, "error_rate": error_rate,
                   "recall": recall, "precision": precision, "specificity": specificity, "f1_score": f1_score,
                   "f2_score": f2_score, "f_point5_score": f_point5_score}
        return results
