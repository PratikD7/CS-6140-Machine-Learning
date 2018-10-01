import statistics
from random import seed, randrange
import math
import numpy as np
import numpy
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



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
# Calculate the sigmoid value of the data
def sigmoid(scores):
    return 1 / (1 + numpy.exp(-scores))


# =============================================================================#
# Implementation of the gradient descent algorithm
def gradient_descent(train_data, Y_train, learning_rate, n_iterations, tolerance):
    ll = []
    prev_sum_err = math.inf
    train_data = numpy.array(train_data)
    # Add 1 as a feature column
    constant_column_of_1 = np.ones((train_data.shape[0], 1))
    train_data = np.hstack((constant_column_of_1, train_data))
    # Initialize the coefficient vector
    coef = numpy.zeros(train_data.shape[1])

    # Algorithm runs for n_iterations
    for step in range(n_iterations):
        logistic_loss = 0

        scores = np.dot(train_data, coef)
        predictions = sigmoid(scores)

        # update the weights with gradient
        output_error = Y_train - predictions
        gradient = np.dot(train_data.T, output_error)
        coef += learning_rate * gradient
        output_error = [abs(n) for n in output_error]
        sum_error = sum(output_error)
        # print(output_error)
        # Convergence criteria for the gradient descent algorithm
        if abs(prev_sum_err - sum_error) <= tolerance:
            break
        else:
            prev_sum_err = sum_error

        # ll.append(sum_error)
        ll = sum_error

        # Calculating logistic loss
        '''
        for index in range(len(train_data)):
            wtxi = np.dot(train_data[index], coef)
            logistic_loss += np.log(1+np.exp(wtxi)) - Y_train[index]*wtxi
        ll.append(logistic_loss)
        '''

    # print(ll)
    return coef, train_data, ll


# =============================================================================#
# Predict Y values based on X values and coefficient
def predict(row, coeff, y):
    yhat = numpy.dot(coeff.T, row)
    # Bernoulli function for logistic regression
    yhat = math.pow(sigmoid(yhat), y) * math.pow(1 - sigmoid(yhat), 1 - y)
    return yhat


# =============================================================================#
# Logistic regression algorithm implementation
def logistic_regression(train_data, test_data, Y_train, Y_test, n_iterations, tolerance, learning_rate ):
    # Calculating the coefficients from the Gradient Descent algorithm
    coeff, train_data, training_loss = gradient_descent(train_data, Y_train, learning_rate, n_iterations, tolerance,
                                                        )
    final_scores = np.dot(train_data, coeff)
    preds_train = np.round(sigmoid(final_scores))

    # Calculate the training accuracy, precision and recall
    accuracy_train = accuracy_score(Y_train, preds_train)
    precision_train = precision_score(Y_train, preds_train)
    recall_train = recall_score(Y_train, preds_train)

    # Test data
    test_data = numpy.array(test_data)
    constant_column_of_1 = np.ones((test_data.shape[0], 1))
    test_data = np.hstack((constant_column_of_1, test_data))
    final_scores = np.dot(test_data, coeff)
    preds_test = np.round(sigmoid(final_scores))

    # Calculate the testing accuracy, precision and recall
    accuracy_test = accuracy_score(Y_test, preds_test)
    precision_test = precision_score(Y_test, preds_test)
    recall_test = recall_score(Y_test, preds_test)

    # logistic_loss = log_loss(Y_train, preds_train)
    # n = list(range(len(logistic_loss)))

    # Plotting the logistic loss vs number of iterations
    '''
    plt.scatter(n, logistic_loss)
    plt.title('Logistic Loss of Spambase against Number of Iterations')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Logistic Loss')
    plt.tight_layout()
    plt.show()
    '''

    return accuracy_train, precision_train, recall_train, accuracy_test, precision_test, recall_test, training_loss


# =============================================================================#
# Logistic Regression algorithm call
def lr_algorithm(ds, n_folds, n_iterations, tolerance, learning_rate ):
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

        # Sklearn's logistic regression algorithm to cross check the accuracy of our algorithm
        # clf = LogisticRegression(fit_intercept=True)
        # clf.fit(train_data, Y_train)
        # # print(clf.intercept_, clf.coef_)
        # print(clf.score(train_data, Y_train))

        # Logistic Regression algorithm function call
        accuracy_train, precision_train, recall_train, accuracy_test, precision_test, recall_test, training_loss = \
            logistic_regression(train_data, test_data, Y_train, Y_test, n_iterations, tolerance, learning_rate, )
        accuracy_list_train.append(accuracy_train)
        precision_list_train.append(precision_train)
        recall_list_train.append(recall_train)
        accuracy_list_test.append(accuracy_test)
        precision_list_test.append(precision_test)
        recall_list_test.append(recall_test)

    return accuracy_list_train, precision_list_train, recall_list_train \
        , accuracy_list_test, precision_list_test, recall_list_test, training_loss


# =============================================================================#
# MAIN Function
def main():
    seed(1)
    # List of dataset files
    dataset = ['spambase.csv']

    print("%s DATASET" % dataset[0])
    file = dataset[0]
    # Load the csv file of the dataset
    ds = load_csv(file)

    # Parameters for the logistic regression gradient descent algorithm
    n_folds = 10
    n_iterations = [10000,50000,100000]  # SELECT THE OPTIMAL VALUES
    tolerance = [0.1,0.01,0.001]  # SELECT THE OPTIMAL VALUES
    learning_rate = 1e-3
    sp = ['111',]

    for n in n_iterations:
        tl = []
        for t in tolerance:
            # Main Logistic Regression function
            accuracy_train, precision_train, recall_train, accuracy_test, precision_test, recall_test, training_loss \
                = lr_algorithm(ds, n_folds, n, t, learning_rate)
            tl.append(training_loss)

        plt.plot(tolerance, tl, marker='o')
        plt.title('Spambase: Tolerance vs Training Loss for maximum iterations: {0}'.format(n))
        plt.xlabel("Tolerance")
        plt.ylabel("Training Loss")
        plt.tight_layout()
        plt.grid('on')
        plt.show()

# =============================================================================#
if __name__ == '__main__':
    main()
