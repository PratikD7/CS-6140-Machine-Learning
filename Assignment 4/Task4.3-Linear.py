import statistics
from random import seed, randrange
import math
import numpy as np
import numpy
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
from scipy.spatial import distance
from sklearn import svm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import copy


# =============================================================================#
# Load a CSV file
def load_csv(filename):
    data = pd.read_csv(filename, header=None)
    dataset = data.values
    return dataset


# =============================================================================#
# Z-normalize the dataset i.e value = (value - mean)/stdev
def z_normalize(ds, mean, std_dev):
    for row in ds:
        for i in range(len(row)):
            if row[i] != None:
                if std_dev[i] == 0:
                    row[i] = row[i]
                else:
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
              flag, C, m_folds):
    auc_score_matrix = [0.0 for i in range(len(C))]
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

        # Create test data
        inner_test_data = []
        for row in fold:
            row_copy = list(row)
            inner_test_data.append(row_copy)
            row_copy[-1] = None
        for row in inner_test_data:
            del row[-1]

        # Create target labels for test data

        inner_Y_test = [row[-1] for row in fold]
        inner_Y_test = [int(row) for row in inner_Y_test]

        idx = 0
        for c_value in C:
            print(index)
            model = svm.SVC(kernel='linear', C=c_value, probability=True)
            model.fit(inner_train_data, inner_Y_train)
            probability_matrix = model.predict_proba(inner_test_data)
            probability_predictions = [row[1] for row in probability_matrix]
            fpr, tpr, thresholds = roc_curve(inner_Y_test, probability_predictions, pos_label=1)
            roc_auc = auc(fpr, tpr)
            auc_score_matrix[idx] += roc_auc
            idx += 1
        print("-"*50)
    return C[auc_score_matrix.index(max(auc_score_matrix))]


# =============================================================================#
# SVM algorithm call
def SVM_algorithm(train_data, test_data, k_folds, n_iterations, tolerance, learning_rate, flag, C, m_folds):
    # Initialize the lists
    accuracy_list_train = []
    precision_list_train = []
    recall_list_train = []
    accuracy_list_test = []
    precision_list_test = []
    recall_list_test = []
    C_list = []

    temp_train_data = copy.deepcopy(train_data)

    train_data = train_data.tolist()
    # Create train labels
    Y_train = []
    for i in range(len(train_data)):
        Y_train.append(train_data[i][-1])
    for row in train_data:
        del row[-1]
    train_data = numpy.array(train_data)
    Y_train = [int(row) for row in Y_train]
    # Normalize the train data
    mean, stdev = mstd(train_data)
    z_normalize(train_data, mean, stdev)

    test_data = test_data.tolist()
    # Create test data
    Y_test = []
    for i in range(len(test_data)):
        Y_test.append(test_data[i][-1])
    for row in test_data:
        del row[-1]
    test_data = numpy.array(test_data)
    # Normalize the test data
    z_normalize(test_data, mean, stdev)
    Y_test = [int(row) for row in Y_test]

    # SVM function call
    c_value = M_fold_CV(temp_train_data, test_data, Y_train, Y_test, n_iterations, tolerance,
                                 learning_rate, flag, C, m_folds)

    # Train the model with train data and test it on test data
    model = svm.SVC(kernel='linear', C=c_value, probability=True)
    model.fit(train_data, Y_train)
    preds_train = model.predict(train_data)
    preds_test = model.predict(test_data)
    probability_matrix = model.predict_proba(test_data)
    probability_predictions = [row[1] for row in probability_matrix]
    fpr, tpr, thresholds = roc_curve(Y_test, probability_predictions, pos_label=1)
    plt.plot(fpr, tpr)

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

    # Plot the AUC-ROC curve
    str = ["Digits: ROC Curve: Model1: '0' Positive", "Digits: ROC Curve: Model1: '1' Positive",
           "Digits: ROC Curve: Model1: '2' Positive", "Digits: ROC Curve: Model1: '3' Positive",
           "Digits: ROC Curve: Model1: '4' Positive", "Digits: ROC Curve: Model1: '5' Positive",
           "Digits: ROC Curve: Model1: '6' Positive", "Digits: ROC Curve: Model1: '7' Positive",
           "Digits: ROC Curve: Model1: '8' Positive", "Digits: ROC Curve: Model1: '9' Positive"]
    plt.title(str[flag])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.show()
    return accuracy_list_train, precision_list_train, recall_list_train \
        , accuracy_list_test, precision_list_test, recall_list_test, C_list


# =============================================================================#
# MAIN Function
def main():
    seed(1)
    # List of dataset files
    train_data = load_csv('digits train.csv')
    test_data = load_csv('digits test.csv')

    flag = 0

    print("%DIGITS Dataset")

    temp_ds_train = copy.deepcopy(train_data)
    temp_ds_test = copy.deepcopy(test_data)

    # Create a different model for each of the target value
    for i in range(10):
        train_data = copy.deepcopy(temp_ds_train)
        for row in train_data:
            if row[-1] == i:
                row[-1] = 1
            else:
                row[-1] = 0
        test_data = copy.deepcopy(temp_ds_test)
        for row in test_data:
            if row[-1] == i:
                row[-1] = 1
            else:
                row[-1] = 0

        # Parameters for the logistic regression gradient descent algorithm
        k_folds = 10
        m_folds = 5
        # n_iterations = 1000  # SELECT THE OPTIMAL VALUES
        n_iterations = 100
        tolerance = 0.001  # SELECT THE OPTIMAL VALUES
        learning_rate = 1e-3

        # Main SVM function
        accuracy_train, precision_train, recall_train, accuracy_test, precision_test, recall_test, c_value \
            = SVM_algorithm(train_data, test_data, k_folds, n_iterations, tolerance, learning_rate, flag, C,
                            m_folds)
        flag += 1
        print("Best values of C: ", end='')
        print(c_value)
        print("Training Accuracy : ", end='')
        print(accuracy_train)
        print("Testing Accuracy : ", end='')
        print(accuracy_test)
        print("Training Precision : ", end='')
        print(precision_train)
        print("Testing Precision : ", end='')
        print(precision_test)
        print("Training Recall : ", end='')
        print(recall_train)
        print("Testing Recall : ", end='')
        print(recall_test)



# =============================================================================#
if __name__ == '__main__':
    main()
