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
def add_constant_feature(train_data):
    z = [[1]] * len(train_data)
    train_data = numpy.append(z, train_data, axis=1)
    train_data = list(train_data)
    return train_data


# =============================================================================#
def new_dataset(ds, power):
    dataset = []
    temp_ds = ds[:, :-1]
    dataset = ds
    if power == 1:
        return ds
    else:
        for i in range(1, power):
            exp_columns = [j ** (i + 1) for j in temp_ds]
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
    coeff = ((numpy.linalg.inv(X_transpose.dot(X))).dot(X_transpose)).dot(Y)

    for row in test_data:
        yhat = predict(row, coeff)
        predictions_test.append(yhat)

    for row in train_data:
        yhat = predict(row, coeff)
        predictions_train.append(yhat)

    return (predictions_test, predictions_train)


# =============================================================================#
def lrAlgorithm(ds, power):
    train_data = ds
    train_data = add_constant_feature(train_data)
    test_data = load_csv('sinData_Validation.csv')
    test_data = new_dataset(test_data, power)

    Y_train = []
    for i in range(len(train_data)):
        Y_train.append(train_data[i][-1])

    train_data = numpy.array(train_data)
    X_train = train_data[:, :-1]
    X_train = list(X_train)

    Y_test = []
    for i in range(len(test_data)):
        Y_test.append(test_data[i][-1])

    test_data = add_constant_feature(test_data)
    test_data = numpy.array(test_data)
    X_test = test_data[:, :-1]
    X_test = list(X_test)

    predicted_test, predicted_train, = linear_regression(X_train, X_test, Y_train)

    rmse, sse_test = calculate_rmse(Y_test, predicted_test)
    rmse_test = (rmse)

    rmse, sse_train = calculate_rmse(Y_train, predicted_train)
    rmse_train = (rmse)

    return rmse_test, rmse_train, sse_test, sse_train


# =============================================================================#
# Plot SSE values : train and test
def plot_sse_values(mean_SumSqError_test, mean_SumSqError_train, poly):
    plt.subplot(211)
    plt.plot(mean_SumSqError_test, poly, marker='o')
    plt.plot(mean_SumSqError_train, poly, marker='o')
    plt.title("Mean SSE vs Max(P) of SINUSOID")
    plt.xlabel("MeanSSE")
    plt.ylabel("P")
    # plt.xlim(-10,100)
    plt.legend(["Test Mean SSE", "Train Mean SSE"])
    plt.tight_layout()
    # plt.show()

    plt.subplot(212)
    plt.plot(mean_SumSqError_test, poly, marker='o')
    plt.plot(mean_SumSqError_train, poly, marker='o')
    plt.title("Enlarged figure of the above")
    plt.xlabel("MeanSSE")
    plt.ylabel("P")
    plt.xlim(-1, 6)
    plt.legend(["Test Mean SSE", "Train Mean SSE"])
    plt.tight_layout()

    plt.show()
# =============================================================================#

def main():
    mean_SumSqError_train = []
    mean_SumSqError_test = []
    dataset_file = 'sinData_Train.csv'
    print("%s DATASET" % dataset_file)

    poly = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for power in poly:
        file = dataset_file
        ds = load_csv(file)

        # Add new polynomial features into the dataset
        ds = new_dataset(ds, power)

        rmse_test, rmse_train, sse_test, sse_train = lrAlgorithm(ds, power)

        print('Polynomial =:%d' % power)
        print('RMSE_test: %s' % rmse_test)
        print('RMSE_train: %s' % rmse_train)
        print('SSE_test: %s' % sse_test)
        print('SSE_train: %s' % sse_train)

        print("")

        mean_SumSqError_test.append(rmse_test ** 2)
        mean_SumSqError_train.append(rmse_train ** 2)

    #Plotting SSE
    plot_sse_values(mean_SumSqError_test, mean_SumSqError_train, poly, )


# =============================================================================#
if __name__ == "__main__":
    main()
    # =============================================================================#
