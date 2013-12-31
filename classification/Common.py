#!/usr/bin/env python

"""Common.py: Module to represent an input dataset"""

__author__ = "Manoj Kumar"


class DataSet:
    """Object representation of a LIBSVM format input DataSet

    Attributes:
        datafile: Name of input data file
        tuples: A list of all the tuples present in LIBSVM format Input Data Set File
    """

    def __init__(self, datafile):
        """ Initializes datafile and tuples variables """
        self.datafile = datafile
        self.tuples = []

    def readData(self):
        """ Read dataset from Input File and store it in memory
        """
        f = open(self.datafile, "r")
        for next_line in f:
            #remove leading/training white space from the line
            next_line = next_line.strip()
            #Ignore empty lines in the DataSet
            if next_line != "":
                tuple = next_line.split()
                #The first field is the label field - +1 or -1
                label = tuple[0]
                attrs = {}
                #Rest of the line consists of attributes. Fetch them one by one
                for attr in tuple[1:]:
                    index_and_value = attr.split(":")
                    index = int(index_and_value[0])
                    value = int(index_and_value[1])
                    attrs[index] = value
                t = Tuple(label, attrs)
                self.tuples.append(t)

    def __repr__(self):
        """ String representation of all the tuples in DataSet
        """
        return "\n".join([str(t) for t in self])

    def __iter__(self):
        """ DataSet should as a list that can be iterated over its tuples
        """
        return iter(self.tuples)

    def __len__(self):
        """ Number of tuples in DataSet
        """
        return len(self.tuples)


class Tuple:
    """ Object repesentation of a row from LibSVMFormat Input Data Set

    Attributes
    label - Class label of this tuple
    attrs - Dictionary of (Index,Value) of different attributes
    """

    def __init__(self, label, attrs):
        self.label = label
        self.attrs = attrs

    def __repr__(self):
        """ String representation of a Tuple
        """
        return "%s %s" % (self.label, self.attrs)