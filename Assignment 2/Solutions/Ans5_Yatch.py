import statistics
from random import randrange, seed

import numpy
import pandas as pd
import math
import matplotlib.pyplot as plt


# =============================================================================#
def load_csv(filename):
    data = pd.read_csv(filename)
    dataset = data.values
    return dataset


# =============================================================================#
def z_normalize(ds, mean, std_dev):
    for row in ds:
        for i in range(len(row)):
            if row[i] != None:
                row[i] = ((row[i]) - mean[i]) / std_dev[i]


# =============================================================================#
def mstd(ds):
    mean = []
    std_dev = []
    for i in range(len(ds[0])):
        col_values = [row[i] for row in ds]
        mean.append(statistics.mean(col_values))
        std_dev.append(statistics.stdev(col_values))
    return mean, std_dev


# =============================================================================#
def add_constant_feature(train_data):
    z = [[1]] * len(train_data)
    train_data = numpy.append(z, train_data, axis=1)
    train_data = list(train_data)
    return train_data


# =============================================================================#
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
def predict(row, coef):
    # yhat = coef[0]
    yhat = 0
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
def linear_regression(train_data, test_data, Y_train):
    predictions_test = list()
    predictions_train = list()

    # From the normal equation
    X = numpy.array(train_data)
    X_transpose = X.T
    Y = Y_train
    # print(X_transpose)
    coeff = ((numpy.linalg.inv(X_transpose.dot(X))).dot(X_transpose)).dot(Y)
    # print(coeff)
    # print(coeff)

    for row in test_data:
        yhat = predict(row, coeff)
        predictions_test.append(yhat)

    for row in train_data:
        yhat = predict(row, coeff)
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

def lrAlgorithm(ds, power, n_folds):
    folds = cv_split(ds, n_folds)
    rmse_test = list()
    rmse_train = list()
    sse_train = list()
    sse_test = list()
    index = 0

    for fold in folds:
        # print(fold)
        train_data = list(folds)
        train_data = numpy.array(train_data)
        train_data = numpy.delete(train_data, (index), axis=0)
        index += 1
        train_data = train_data.tolist()
        train_data = sum(train_data, [])
        mean, stdev = mstd(train_data)
        z_normalize(train_data, mean, stdev)

        test_data = []
        for row in fold:
            row_copy = list(row)
            test_data.append(row_copy)
            row_copy[-1] = None
        # mean, stdev = mstd(train_data)
        z_normalize(test_data, mean, stdev)



        Y_train = []
        for i in range(len(train_data)):
            Y_train.append(train_data[i][-1])

        train_data = numpy.array(train_data)
        train_data = train_data[:, :-1]
        train_data = list(train_data)
        X_train = add_constant_feature(train_data)

        test_data = numpy.array(test_data)
        test_data = test_data[:, :-1]
        test_data - list(test_data)
        X_test = add_constant_feature(test_data)

        # print(Y_train[:5]) #######################################

        predicted_test, predicted_train, = linear_regression(X_train, X_test, Y_train)
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
def plot_rmse_values(t1, t2, poly):
    # Plotting RMSE values
    plt.subplot(211)
    plt.plot(t1, poly, marker='o')
    plt.plot(t2, poly, marker='o')
    plt.title("Mean RMSE vs Max(P) of YATCH")
    plt.xlabel("MeanRMSE")
    plt.ylabel("P")
    # plt.xlim(-10,100)
    plt.legend(["Test Mean RMSE", "Train Mean RMSE"])
    plt.tight_layout()
    # plt.show()

    plt.subplot(212)
    plt.plot(t1, poly, marker='o')
    plt.plot(t2, poly, marker='o')
    plt.title("Enlarged figure of the above")
    plt.xlabel("MeanRMSE")
    plt.ylabel("P")
    plt.xlim(-1, 4)
    plt.legend(["Test Mean RMSE", "Train Mean RMSE"])
    plt.tight_layout()

    plt.show()

# =============================================================================#
def main():
    seed(1)
    t1 = []
    t2 = []
    n_folds = 10
    dataset_file = 'yachtData.csv'
    print("%s DATASET" % dataset_file)
    file = dataset_file

    poly = [1, 2, 3, 4, 5, 6, 7]
    # poly = [5]
    for power in poly:

        ds = load_csv(file)
        ds = new_dataset(ds, power)

        rmse_test, rmse_train, sse_test, sse_train = lrAlgorithm(ds, power, n_folds)

        print('Polynomial =:%d' % power)

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

    #Plot RMSE values
    plot_rmse_values(t1, t2, poly)


# =============================================================================#
if __name__ == "__main__":
    main()
    # =============================================================================#
