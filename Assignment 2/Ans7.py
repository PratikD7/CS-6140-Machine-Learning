import statistics
from random import randrange, seed

import numpy
import pandas as pd
import math
import matplotlib.pyplot as plt


# =============================================================================#
# Load the dataset
def load_csv(filename):
    data = pd.read_csv(filename)
    dataset = data.values
    return dataset


# =============================================================================#
# Center the dataset around mean
def subrtact_mean(ds, mean, std_dev):
    for row in ds:
        for i in range(len(row)):
            if row[i] != None:
                row[i] = ((row[i]) - mean[i])


# =============================================================================#
#Calculate mean and stddev
def mstd(ds):
    mean = []
    std_dev = []
    for i in range(len(ds[0])):
        col_values = [row[i] for row in ds]
        mean.append(statistics.mean(col_values))
        std_dev.append(statistics.stdev(col_values))
    return mean, std_dev


# =============================================================================#
#Add polynomial features to the dataset
def new_dataset(ds, power):
    temp_ds = ds[:, :-1]
    dataset = ds

    for col in range(len(ds[0]) - 1):
        temp_ds_col = temp_ds[:, col]

        if power == 1:
            pass
        else:
            for i in range(1, power):
                exp_columns = [[j ** (i + 1)] for j in temp_ds_col]
                dataset = numpy.append(exp_columns, dataset, axis=1)
                dataset = list(dataset)
    return dataset


# =============================================================================#
# Make predictions with coefficients
def predict(row, coef, w0):
    # yhat = coef[0]
    # yhat = 0
    yhat = w0
    for i in range(len(row)):
        yhat += coef[i] * row[i]
    return yhat


# =============================================================================#
# Calculate root mean squared error
def calculate_rmse(actual, predicted):
    sum_sq_error = 0.0

    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_sq_error += (prediction_error ** 2)
    mean_error = sum_sq_error / float(len(actual))
    return math.sqrt(mean_error), sum_sq_error


# =============================================================================#
#Calculate the coefficients of the model
def linear_regression(train_data, test_data, Y_train, w0, lambd):
    predictions_test = list()
    predictions_train = list()

    # From the normal equation
    X = numpy.array(train_data)
    X_transpose = X.T
    Y = Y_train
    lambda_I = numpy.eye(len(train_data[0])) * lambd
    coeff = ((numpy.linalg.inv((X_transpose.dot(X)) + lambda_I)).dot(X_transpose)).dot(Y)

    for row in test_data:
        yhat = predict(row, coeff, w0)
        predictions_test.append(yhat)

    for row in train_data:
        yhat = predict(row, coeff, w0)
        predictions_train.append(yhat)

    return (predictions_test, predictions_train)


# =============================================================================#
def cv_split(ds, n_folds):
    data_set_split = []
    data = list(ds)
    fold_size = int(len(ds) / n_folds)
    for i in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            index = randrange(len(data))
            fold.append(data.pop(index))
        data_set_split.append(fold)
    return data_set_split


# =============================================================================#
#Linear Regression Algorithm
def lrAlgorithm(ds, n_folds, w0, lambd):
    folds = cv_split(ds, n_folds)
    rmse_test = list()
    rmse_train = list()
    sse_train = list()
    sse_test = list()
    index = 0

    for fold in folds:
        train_data = list(folds)
        train_data = numpy.array(train_data)
        train_data = numpy.delete(train_data, (index), axis=0)
        index += 1
        train_data = train_data.tolist()
        train_data = sum(train_data, [])
        mean, stdev = mstd(train_data)
        subrtact_mean(train_data, mean, stdev)

        test_data = []
        for row in fold:
            row_copy = list(row)
            test_data.append(row_copy)
            row_copy[-1] = None
        # mean, stdev = mstd(train_data)
        subrtact_mean(test_data, mean, stdev)

        Y_train = []
        for i in range(len(train_data)):
            Y_train.append(train_data[i][-1])

        train_data = numpy.array(train_data)
        train_data = train_data[:, :-1]
        train_data = list(train_data)
        X_train = train_data

        test_data = numpy.array(test_data)
        test_data = test_data[:, :-1]
        test_data - list(test_data)
        X_test = test_data

        predicted_test, predicted_train, = linear_regression(X_train, X_test, Y_train, w0, lambd)
        actual = [row[-1] for row in fold]
        for i in range(len(actual)):
            actual[i] = (actual[i] - mean[-1]) / stdev[-1]

        rmse, sse = calculate_rmse(actual, predicted_test)
        rmse_test.append(rmse)
        sse_test.append(sse)

        rmse, sse = calculate_rmse(Y_train, predicted_train)
        rmse_train.append(rmse)
        sse_train.append(sse)

    return rmse_test, rmse_train, sse_test, sse_train


# =============================================================================#
def main():
    seed(1)
    n_folds = 10
    dataset_file = 'sinData_Train.csv'
    print("%s DATASET" % dataset_file)
    file = dataset_file
    ds = load_csv(file)
    m, s = mstd(ds)
    subrtact_mean(ds, m, s)
    w0 = m[-1]
    lambd = [x / 5.0 for x in range(51)]
    poly = [5, 9]
    for power in poly:
        t1 = []
        t2 = []
        for lamb in lambd:
            ds = load_csv(file)
            ds = new_dataset(ds, power)

            rmse_test, rmse_train, sse_test, sse_train = lrAlgorithm(ds, n_folds, w0, lamb)

            print('Polynomial =:%d,' % power)
            print('Lambda =:%.1f,' % lamb)

            print('RMSE_test: %s' % rmse_test)
            test_mean_rmse = sum(rmse_test) / float(len(rmse_test))
            print('Mean test RMSE: %f' % test_mean_rmse)
            print('RMSE_train: %s' % rmse_train)
            train_mean_rmse = sum(rmse_train) / float(len(rmse_train))
            print('Mean test RMSE: %f' % train_mean_rmse)
            print('SSE_test: %s' % sse_test)
            print('Mean SSE_test: %f' % (sum(sse_test) / float(len(sse_test))))
            print('SSE_train: %s' % sse_train)
            print('Mean SSE_train: %f' % (sum(sse_train) / float(len(sse_train))))

            print("")

            t1.append(test_mean_rmse)
            t2.append(train_mean_rmse)

        plt.subplot(111)
        plt.plot(lambd, t1, marker='o')
        plt.plot(lambd, t2, marker='o')
        plt.title("Mean RMSE vs Lambda")
        plt.xlabel("Lambda")
        plt.ylabel("RMSE")
        plt.legend(["Test Mean RMSE", "Train Mean RMSE"])
        plt.tight_layout()
        plt.show()


# =============================================================================#
if __name__ == "__main__":
    main()
    # =============================================================================#
