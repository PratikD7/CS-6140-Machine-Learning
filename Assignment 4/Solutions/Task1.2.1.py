import statistics
from random import seed, randrange
import math
import numpy as np
import numpy
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss


# =============================================================================#
# Load a CSV file
def load_csv(filename):
    data = pd.read_csv(filename)
    dataset = data.values
    return dataset


# =============================================================================#
# Z-normalize the dataset i.e value = (value - mean)/stdev
def z_normalize(ds, mean, std_dev):
    for row in ds:
        for i in range(len(row)):
            if row[i] != None:
                row[i] = ((row[i]) - mean[i]) / std_dev[i]


# =============================================================================#
# Calculate the mean and std dev of all the columns of a dataset
def mstd(ds):
    mean = []
    std_dev = []
    for i in range(len(ds[0])):
        col_values = [row[i] for row in ds]
        mean.append(statistics.mean(col_values))
        std_dev.append(statistics.stdev(col_values))
    return mean, std_dev


# =============================================================================#
# Split the dataset into training and cross validation set
def cv_split(ds, n_folds):
    data_set_split = []
    data = list(ds)
    fold_size = int(len(ds) / n_folds)
    for i in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            index = randrange(len(data))
            # print(index)
            fold.append(data.pop(index))
        data_set_split.append(fold)
    return data_set_split


# =============================================================================#
# Implementation of the gradient descent algorithm
def gradient_descent(train_data, Y_train, learning_rate, n_iterations, tolerance, flag):
    train_data = numpy.array(train_data)

    # Add 1 as a feature column
    constant_column_of_1 = np.ones((train_data.shape[0], 1))
    train_data = np.hstack((constant_column_of_1, train_data))

    # Initialize the coefficient vector
    coef = numpy.zeros(train_data.shape[1])

    # Algorithm runs for n_iterations
    for step in range(n_iterations):
        sum_error = 0
        index = 0
        for row in train_data:
            predictions = predict(row, coef)
            output_error = abs(Y_train[index] - predictions)
            # If there's a prediction error then adjust the weights accordingly
            if predictions * Y_train[index] <= 0:
                for i in range(len(coef)):
                    coef[i] += learning_rate * Y_train[index] * train_data[index][i]
            sum_error += output_error
            index += 1

        # Convergence criteria for the gradient descent algorithm
        print(sum_error)
        if sum_error == 0:
            break

    return coef, train_data


# =============================================================================#
# Make a prediction with weights
def predict(row, coef):
    activation = 0.0
    for i in range(len(row)):
        activation += coef[i] * row[i]
    return 1.0 if activation >= 0.0 else -1.0


# =============================================================================#
# Single Perceptron algorithm implementation
def single_perceptron(train_data, test_data, Y_train, Y_test, n_iterations, tolerance, learning_rate, flag):
    preds_train = []
    preds_test = []

    # Calculating the coefficients from the Gradient Descent algorithm
    coeff, train_data = gradient_descent(train_data, Y_train, learning_rate, n_iterations, tolerance,
                                         flag)
    print("-" * 50)
    for row in train_data:
        preds_train.append(predict(row, coeff))

    # Calculate the training accuracy, precision and recall
    accuracy_train = accuracy_score(Y_train, preds_train)
    precision_train = precision_score(Y_train, preds_train)
    recall_train = recall_score(Y_train, preds_train)

    # Creating Test data
    test_data = numpy.array(test_data)
    constant_column_of_1 = np.ones((test_data.shape[0], 1))
    test_data = np.hstack((constant_column_of_1, test_data))
    # final_scores = np.dot(test_data, coeff)
    for row in test_data:
        preds_test.append(predict(row, coeff))

    # Calculate the testing accuracy, precision and recall
    accuracy_test = accuracy_score(Y_test, preds_test)
    precision_test = precision_score(Y_test, preds_test)
    recall_test = recall_score(Y_test, preds_test)
    return accuracy_train, precision_train, recall_train, accuracy_test, precision_test, recall_test


# =============================================================================#
# Single Perceptron algorithm call
def perceptron_algorithm(ds, n_folds, n_iterations, tolerance, learning_rate, flag):
    # Split the dataset into training and cross validation set
    folds = cv_split(ds, n_folds)
    # Initialize the lists
    accuracy_list_train = []
    precision_list_train = []
    recall_list_train = []
    accuracy_list_test = []
    precision_list_test = []
    recall_list_test = []
    index = 0
    i = 0
    for fold in folds:
        # Create train data
        train_data = list(folds)
        train_data = numpy.array(train_data)
        train_data = numpy.delete(train_data, (index), axis=0)
        index += 1
        train_data = train_data.tolist()
        train_data = sum(train_data, [])

        # Create train labels
        Y_train = []
        for i in range(len(train_data)):
            Y_train.append(train_data[i][-1])
        for row in train_data:
            del row[-1]

        # Normalize the train data
        mean, stdev = mstd(train_data)
        z_normalize(train_data, mean, stdev)

        # Create test data
        test_data = []
        for row in fold:
            row_copy = list(row)
            test_data.append(row_copy)
            row_copy[-1] = None
        for row in test_data:
            del row[-1]

        # Normalize the test data
        z_normalize(test_data, mean, stdev)

        # Create target labels for test data
        Y_test = [row[-1] for row in fold]

        # Single Perceptron algorithm function call
        accuracy_train, precision_train, recall_train, accuracy_test, precision_test, recall_test = \
            single_perceptron(train_data, test_data, Y_train, Y_test, n_iterations, tolerance, learning_rate, flag)

        accuracy_list_train.append(accuracy_train)
        precision_list_train.append(precision_train)
        recall_list_train.append(recall_train)
        accuracy_list_test.append(accuracy_test)
        precision_list_test.append(precision_test)
        recall_list_test.append(recall_test)

    return accuracy_list_train, precision_list_train, recall_list_train \
        , accuracy_list_test, precision_list_test, recall_list_test


# =============================================================================#
# MAIN Function
def main():
    seed(1)

    # List of dataset files
    datasets_list = ['perceptronData.csv']
    flag = True

    for dataset in datasets_list:
        print("%s DATASET" % dataset)
        file = dataset

        # Load the csv file of the dataset
        ds = load_csv(file)

        # Parameters for the gradient descent
        n_folds = 10
        n_iterations = 50000  # SELECT THE OPTIMAL VALUES
        tolerance = 0.001  # SELECT THE OPTIMAL VALUES
        learning_rate = 1e-3

        # Main Single Perceptron algorithm
        accuracy_train, precision_train, recall_train, accuracy_test, precision_test, recall_test \
            = perceptron_algorithm(ds, n_folds, n_iterations, tolerance, learning_rate, flag)
        flag = False

        # Printing out the statistics of the model
        print("Training Accuracy : ", end='')
        print(accuracy_train)
        print("Mean Training Accuracy : ", end='')
        print(statistics.mean(accuracy_train))
        print("Testing Accuracy : ", end='')
        print(accuracy_test)
        print("Mean Testing Accuracy : ", end='')
        print(statistics.mean(accuracy_test))

        print("Training Precision : ", end='')
        print(precision_train)
        print("Mean Training Precision : ", end='')
        print(statistics.mean(precision_train))
        print("Testing Precision : ", end='')
        print(precision_test)
        print("Mean Testing Precision : ", end='')
        print(statistics.mean(precision_test))

        print("Training Recall : ", end='')
        print(recall_train)
        print("Mean Training Recall : ", end='')
        print(statistics.mean(recall_train))
        print("Testing Recall : ", end='')
        print(recall_test)
        print("Mean Testing Recall : ", end='')
        print(statistics.mean(recall_test))


# =============================================================================#
if __name__ == '__main__':
    main()
