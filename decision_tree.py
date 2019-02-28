# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import math


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE

    k = {val: (x == val).nonzero()[0] for val in np.unique(x)}

    return k

    raise Exception('Function not yet implemented!')


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    value,count = np.unique(y,return_counts = True)
    p = count.astype('float')/len(y)
    hy = 0.0
    for k in np.nditer(p,op_flags = ['readwrite']):
        hy+=(- k)*(math.log(k)/math.log(2))
        
    return hy
    # INSERT YOUR CODE HERE
    raise Exception('Function not yet implemented!')


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    valuex, countx = np.unique(x,return_counts = True)
    px = countx.astype('float')/len(x)
    hyx = 0.0
    for  pxval,xval in zip(px,valuex):
        hyx+=(pxval)*entropy(y[x==xval])
    hy  = entropy(y)
    ixy = hy -hyx
    return ixy
    # INSERT YOUR CODE HERE
    raise Exception('Function not yet implemented!')


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}

    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    
    root = {}

    # If split_criteria=None, then create a Cartesian product of splits
    if attribute_value_pairs is None:
        attribute_value_pairs = np.vstack([[(i, v) for v in np.unique(x[:, i])] for i in range(x.shape[1])])

    # Get the unique values of the labels and their counts
    yvalue, ycount = np.unique(y, return_counts=True)

    # If the entire set of labels y is pure, return that label
    if len(yvalue) == 1:
        return yvalue[0]

    # If there are no attributes left, or if we've reached the maximum depth, then return the most common value of
    # the current attribute
    if len(attribute_value_pairs) == 0 or depth == max_depth:
        return yvalue[np.argmax(ycount)]

    # Find the best attribute using mutual information as the criterion
    Listofmutualinformation = np.array([mutual_information(np.array(x[:, i] == v).astype(int), y)
                                 for (i, v) in attribute_value_pairs])
    (bestattr, bestvalue) = attribute_value_pairs[np.argmax(Listofmutualinformation)]

    # [print('{0}: {1}'.format(p, v)) for (p, v) in zip(split_criteria, information_gain)]

    # Binarize the selected column into 0 and 1 based on the split criterion, i.e., if attr = value and partition the
    # selected column into two sets
    partitions = partition(np.array(x[:, bestattr] == bestvalue).astype(int))

    # Remove the current attribute-value pair from the list of attributes
    dropindex = np.all(attribute_value_pairs == (bestattr, bestvalue), axis=1)
    attribute_value_pairs = np.delete(attribute_value_pairs, np.argwhere(dropindex), 0)

    for splitindex, index in partitions.items():
        xsubset = x.take(index, axis=0)
        ysubset = y.take(index, axis=0)
        decision = bool(splitindex)

        root[(bestattr, bestvalue, decision)] = id3(xsubset, ysubset, attribute_value_pairs=attribute_value_pairs,
                                            max_depth=max_depth, depth=depth + 1)

    return root

    raise Exception('Function not yet implemented!')


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    for splitval, sub_tree in tree.items():
        attr_index = splitval[0]
        attr_value = splitval[1]
        split_decision = splitval[2]

        if split_decision == (x[attr_index] == attr_value):
            if type(sub_tree) is dict:
                label = predict_example(x, sub_tree)
            else:
                label = sub_tree

            return label

    raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE

    n = len(y_true)
    err = [y_true[i] != y_pred[i] for i in range(n)]
    return sum(err) / n

    raise Exception('Function not yet implemented!')


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))

def confusion_matrix(y_true, y_pred):
    n=len(y_true)
    true_positive=0
    true_negative=0
    false_negative=0
    false_positive=0

    for i in range(n):
        if(y_true[i]==y_pred[i] and y_true[i]==1):
            true_positive=true_positive+1
        if(y_true[i]==y_pred[i] and y_true[i]==0):
            true_negative=true_negative+1
        if(y_true[i]!=y_pred[i]):
            if(y_true[i]==0):
                false_negative=false_negative+1
        if(y_true[i]!=y_pred[i]):
            if(y_true[i]==1):
                false_positive=false_positive+1
        #if(y_true[i]!=y_pred[i] and y_true[i]==0 and y_pred==1):
         #   false_positive=false_positive+1
    confusion=[true_positive,false_positive,true_negative,false_negative]

    return confusion

if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)
    visualize(decision_tree)

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    y_predi = [predict_example(x, decision_tree) for x in Xtrn]
    trn_err = compute_error(ytrn, y_predi)
    con=confusion_matrix(ytst,y_pred)
    truepos=con[0]
    falsepos=con[1]
    trueneg=con[2]
    falseneg=con[3]

    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
    print('Train Error = {0:4.2f}%.'.format(trn_err * 100))

    print('CONFUSION MATRIX')
    print('|-----------------------------------------|')
    print('|true_positive:',truepos,'  | false_positive',falsepos," |")
    print('|-----------------------------------------|')
    print('|true_negative:',trueneg,'  | false_negative',falseneg,'|')
    print('|-----------------------------------------|')

    