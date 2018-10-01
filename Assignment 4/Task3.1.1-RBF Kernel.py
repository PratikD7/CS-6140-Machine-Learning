import statistics
from numpy import unravel_index
from random import seed, randrange
import math
import numpy as np
import numpy
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
from scipy.spatial import distance
from sklearn import svm
import copy


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
def gradient_descent(train_data, Y_train, learning_rate, n_iterations, tolerance, flag, gamma):
    train_data = numpy.array(train_data)
    # Add 1 as a feature column
    constant_column_of_1 = np.ones((train_data.shape[0], 1))
    train_data = np.hstack((constant_column_of_1, train_data))
    # Initialize the coefficient vector
    coef = numpy.zeros(train_data.shape[1])
    alpha = numpy.zeros(train_data.shape[0])

    # Algorithm runs for n_iterations
    for step in range(n_iterations):
        # print(coef)
        sum_error = 0
        index = 0

        for row in train_data:
            yhat = predict(row, train_data, alpha, Y_train, gamma)
            sum_error += abs(yhat - Y_train[index])

            if yhat != Y_train[index]:
                alpha[index] += 1.0

            index += 1

        # if step % 100 == 0:
        #     print("Step number: ", step + 1)
        print("Error: ", sum_error)

        # Convergence criteria for the gradient descent algorithm
        if sum_error == 0:
            break

        # Calculating logistic loss
        '''
      for index in range(len(train_data)):
          wtxi = np.dot(train_data[index], coef)
          logistic_loss += np.log(1+np.exp(wtxi)) - Y_train[index]*wtxi
      ll.append(logistic_loss)
      '''

    # print(ll)
    return coef, train_data, alpha


# =============================================================================#
def rbf_kernel(gamma, x1, x2):
    dst = distance.euclidean(x1, x2)
    return numpy.exp(-gamma * (dst ** 2))


# =============================================================================#
# Make a prediction with weights
def predict(row, train_data, alpha, Y_train, gamma):
    index = 0
    sum = 0.0
    for r in train_data:
        sum += alpha[index] * Y_train[index] * rbf_kernel(gamma, train_data[index], row)
        index += 1

    if sum >= 0.0:
        return 1.0
    else:
        return -1.0


# =============================================================================#
# SVM algorithm implementation
def M_fold_CV(train_data, test_data, Y_train, Y_test, n_iterations, tolerance, learning_rate,
              flag, gamma, C, m_folds):
    test_accuracy_matrix = [[0.0 for i in range(len(C))] for j in range(len(gamma))]
    folds = cv_split(train_data, m_folds)

    index = 0
    for fold in folds:
        print(index)
        # Create train data
        inner_train_data = list(folds)
        inner_train_data = numpy.array(inner_train_data)
        inner_train_data = numpy.delete(inner_train_data, (index), axis=0)
        index += 1
        inner_train_data = inner_train_data.tolist()
        inner_train_data = sum(inner_train_data, [])

        # Create train labels
        inner_Y_train = []
        for i in range(len(inner_train_data)):
            inner_Y_train.append(inner_train_data[i][-1])
        for row in inner_train_data:
            del row[-1]

        inner_Y_train = [int(row) for row in inner_Y_train]

        # Normalize the train data
        mean, stdev = mstd(inner_train_data)
        z_normalize(inner_train_data, mean, stdev)

        # Create test data
        inner_test_data = []
        for row in fold:
            row_copy = list(row)
            inner_test_data.append(row_copy)
            row_copy[-1] = None
        for row in inner_test_data:
            del row[-1]

        # Normalize the test data
        z_normalize(inner_test_data, mean, stdev)

        # Create target labels for test data

        inner_Y_test = [row[-1] for row in fold]
        inner_Y_test = [int(row) for row in inner_Y_test]

        idx_i = 0
        # RBF Kernel SVM
        for g_value in gamma:
            print(index)
            idx_j = 0
            for c_value in C:
                model = svm.SVC(kernel='rbf', C=c_value, gamma=g_value)
                model.fit(inner_train_data, inner_Y_train)
                score = model.score(inner_test_data, inner_Y_test)
                test_accuracy_matrix[idx_i][idx_j] += score
                idx_j += 1
            idx_i += 1

    test_accuracy_matrix = numpy.array(test_accuracy_matrix)
    indices = unravel_index(test_accuracy_matrix.argmax(), test_accuracy_matrix.shape)
    return C[indices[1]], gamma[indices[0]]



# =============================================================================#
# SVM algorithm call
def SVM_algorithm(ds, k_folds, n_iterations, tolerance, learning_rate, flag, gamma, C, m_folds):
    # Split the dataset into training and cross validation set
    folds = cv_split(ds, k_folds)
    # Initialize the lists
    accuracy_list_train = []
    precision_list_train = []
    recall_list_train = []
    accuracy_list_test = []
    precision_list_test = []
    recall_list_test = []
    C_list = []
    G_list = []
    index = 0
    for fold in folds:
        # print(index)
        # Create train data
        train_data = list(folds)
        train_data = numpy.array(train_data)
        train_data = numpy.delete(train_data, (index), axis=0)
        index += 1
        train_data = train_data.tolist()
        train_data = sum(train_data, [])

        temp_train_data = copy.deepcopy(train_data)

        # Create train labels
        Y_train = []
        for i in range(len(train_data)):
            Y_train.append(train_data[i][-1])
        for row in train_data:
            del row[-1]

        Y_train = [int(row) for row in Y_train]

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
        Y_test = [int(row) for row in Y_test]

        # SVM algorithm function call
        c_value, g_value = M_fold_CV(temp_train_data, test_data, Y_train, Y_test, n_iterations, tolerance,
                                     learning_rate, flag, gamma, C, m_folds)
        print("-"*50)
        model = svm.SVC(kernel='rbf', C=c_value, gamma=g_value)
        model.fit(train_data, Y_train)
        preds_train = model.predict(train_data)
        preds_test = model.predict(test_data)
        accuracy_train = accuracy_score(Y_train, preds_train)
        accuracy_test = accuracy_score(Y_test, preds_test)
        precision_train = precision_score(Y_train, preds_train)
        precision_test = precision_score(Y_test, preds_test)
        recall_train = recall_score(Y_train, preds_train)
        recall_test = recall_score(Y_test, preds_test)

        accuracy_list_train.append(accuracy_train)
        precision_list_train.append(precision_train)
        recall_list_train.append(recall_train)
        accuracy_list_test.append(accuracy_test)
        precision_list_test.append(precision_test)
        recall_list_test.append(recall_test)

        C_list.append(c_value)
        G_list.append(g_value)

    return accuracy_list_train, precision_list_train, recall_list_train \
        , accuracy_list_test, precision_list_test, recall_list_test, C_list, G_list


# =============================================================================#
# MAIN Function
def main():
    seed(1)
    # List of dataset files
    datasets_list = ['spambase.csv']
    flag = True

    C = np.arange(-5, 11, 3)
    C = np.array([math.pow(2, x) for x in C])

    gamma = np.arange(-15, 6, 3)
    gamma = np.array([math.pow(2, x) for x in gamma])

    for dataset in datasets_list:
        print("%s DATASET" % dataset)
        file = dataset
        # Load the csv file of the dataset
        ds = load_csv(file)

        # Parameters for the SVM gradient descent algorithm
        k_folds = 10
        m_folds = 5
        # n_iterations = 1000  # SELECT THE OPTIMAL VALUES
        n_iterations = 100
        tolerance = 0.001  # SELECT THE OPTIMAL VALUES
        learning_rate = 1e-3

        # Main SVM function
        accuracy_train, precision_train, recall_train, accuracy_test, precision_test, recall_test, c_value, g_value \
            = SVM_algorithm(ds, k_folds, n_iterations, tolerance, learning_rate, flag, gamma, C, m_folds)

        print("Best C values: ", c_value)
        print("Best gamma values:", g_value)

        flag = False
        print("Training Accuracy : ", end='')
        print(accuracy_train)
        print("Mean Training Accuracy : ", end='')
        print(statistics.mean(accuracy_train))
        print("Std deviation: ", end='')
        print(statistics.stdev(accuracy_train))
        print("Testing Accuracy : ", end='')
        print(accuracy_test)
        print("Mean Testing Accuracy : ", end='')
        print(statistics.mean(accuracy_test))
        print("Std deviation: ", end='')
        print(statistics.stdev(accuracy_test))

        print("Training Precision : ", end='')
        print(precision_train)
        print("Mean Training Precision : ", end='')
        print(statistics.mean(precision_train))
        print("Std deviation: ", end='')
        print(statistics.stdev(precision_train))
        print("Testing Precision : ", end='')
        print(precision_test)
        print("Mean Testing Precision : ", end='')
        print(statistics.mean(precision_test))
        print("Std deviation: ", end='')
        print(statistics.stdev(precision_test))

        print("Training Recall : ", end='')
        print(recall_train)
        print("Mean Training Recall : ", end='')
        print(statistics.mean(recall_train))
        print("Std deviation: ", end='')
        print(statistics.stdev(recall_train))
        print("Testing Recall : ", end='')
        print(recall_test)
        print("Mean Testing Recall : ", end='')
        print(statistics.mean(recall_test))
        print("Std deviation: ", end='')
        print(statistics.stdev(recall_test))


# =============================================================================#
if __name__ == '__main__':
    main()
