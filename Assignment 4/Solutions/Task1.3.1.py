import statistics
from random import seed, randrange
import math
import numpy as np
import numpy
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score


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
def gradient_descent(train_data, Y_train, n_iterations):
    train_data = numpy.array(train_data)
    # Add 1 as a feature column
    constant_column_of_1 = np.ones((train_data.shape[0], 1))
    train_data = np.hstack((constant_column_of_1, train_data))
    # Initialize the coefficient vector
    coef = numpy.zeros(train_data.shape[1])
    alpha = numpy.zeros(train_data.shape[0])

    # Algorithm runs for n_iterations
    for step in range(n_iterations):
        sum_error = 0
        index = 0
        for row in train_data:
            yhat = predict(row, train_data, alpha, Y_train)
            sum_error += abs(yhat - Y_train[index])
            if yhat != Y_train[index]:
                alpha[index] += 1.0
            index += 1

        # Convergence criteria for the gradient descent algorithm
        if sum_error == 0:
            break

    return coef, train_data, alpha


# =============================================================================#
# Make a prediction with weights
def predict(row, train_data, alpha, Y_train):
    index = 0
    sum = 0.0
    for r in train_data:
        sum += alpha[index] * Y_train[index] * np.dot(train_data[index], row)
        index += 1

    if sum >= 0.0:
        return 1.0
    else:
        return -1.0


# =============================================================================#
# Dual Perceptron algorithm implementation
def dual_perceptron_with_linear_kernel(train_data, test_data, Y_train, Y_test, n_iterations, tolerance, learning_rate,
                                       flag):
    preds_train = []
    preds_test = []
    # Calculating the coefficients from the Gradient Descent algorithm
    coeff, train_data, alpha = gradient_descent(train_data, Y_train, n_iterations)
    print("-" * 50)
    # final_scores = np.dot(train_data, coeff)
    for row in train_data:
        preds_train.append(predict(row, train_data, alpha, Y_train))

    # Calculate the training accuracy, precision and recall
    accuracy_train = accuracy_score(Y_train, preds_train)
    precision_train = precision_score(Y_train, preds_train)
    recall_train = recall_score(Y_train, preds_train)

    # Test data
    test_data = numpy.array(test_data)
    constant_column_of_1 = np.ones((test_data.shape[0], 1))
    test_data = np.hstack((constant_column_of_1, test_data))
    # final_scores = np.dot(test_data, coeff)
    for row in test_data:
        preds_test.append(predict(row, train_data, alpha, Y_train))

    # Calculate the testing accuracy, precision and recall
    accuracy_test = accuracy_score(Y_test, preds_test)
    precision_test = precision_score(Y_test, preds_test)
    recall_test = recall_score(Y_test, preds_test)

    return accuracy_train, precision_train, recall_train, accuracy_test, precision_test, recall_test


# =============================================================================#
# Dual Perceptron algorithm call
def dual_perceptron_algorithm(ds, n_folds, n_iterations, tolerance, learning_rate, flag):
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

        # Dual Perceptron algorithm function call
        accuracy_train, precision_train, recall_train, accuracy_test, precision_test, recall_test = \
            dual_perceptron_with_linear_kernel(train_data, test_data, Y_train, Y_test, n_iterations, tolerance,
                                               learning_rate, flag)
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
    datasets_list = ['twoSpirals.csv']
    flag = True

    for dataset in datasets_list:
        print("%s DATASET" % dataset)
        file = dataset
        # Load the csv file of the dataset
        ds = load_csv(file)

        # Parameters for the  gradient descent
        n_folds = 10
        n_iterations = 10000
        tolerance = 0.001  # SELECT THE OPTIMAL VALUES
        learning_rate = 1e-3

        # Main Dual Perceptron function
        accuracy_train, precision_train, recall_train, accuracy_test, precision_test, recall_test \
            = dual_perceptron_algorithm(ds, n_folds, n_iterations, tolerance, learning_rate, flag)
        flag = False
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
