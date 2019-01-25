from random import randrange, seed
import math
import numpy
import pandas as pd
import statistics


# =============================================================================#
# Load a CSV file
def load_csv(filename):
    data = pd.read_csv(filename)
    dataset = data.values
    return dataset


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
# Calculate root mean squared error
def calculate_rmse(actual, predicted):
    sum_sq_error = 0.0

    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_sq_error += (prediction_error ** 2)
    mean_error = sum_sq_error / float(len(actual))
    return math.sqrt(mean_error)


# =============================================================================#

# Make predictions with coefficients
def predict(row, coef):
    # yhat = coef[0]
    yhat = 0
    for i in range(len(row)):
        yhat += coef[i] * row[i]
    return yhat


# =============================================================================#
def linear_regression(train_data, test_data, Y_train):
    predictions_test = list()
    predictions_train = list()

    # From the normal equation formula
    X = numpy.array(train_data)
    X_transpose = X.T
    Y = Y_train
    coeff = ((numpy.linalg.inv(X_transpose.dot(X))).dot(X_transpose)).dot(Y)
    # print(coeff)

    # Predict test data
    for row in test_data:
        yhat = predict(row, coeff)
        predictions_test.append(yhat)

    # Predict train data
    for row in train_data:
        yhat = predict(row, coeff)
        predictions_train.append(yhat)

    return (predictions_test, predictions_train)


# =============================================================================#
# Add column of 1's at the beginning of the dataset
def add_constant_feature(train_data):
    z = [[1]] * len(train_data)
    train_data = numpy.append(z, train_data, axis=1)
    train_data = list(train_data)
    return train_data


# =============================================================================#
# Make training and testing data and apply main linear regression algorithm
def lrAlgorithm(dataset, ds, n_folds):
    folds = cv_split(ds, n_folds)
    rmse_test = list()
    rmse_train = list()
    sse = list()
    index = 0

    for fold in folds:
        # Create train data
        train_data = list(folds)
        train_data = numpy.array(train_data)
        train_data = numpy.delete(train_data, (index), axis=0)
        index += 1
        train_data = train_data.tolist()
        train_data = sum(train_data, [])
        mean, stdev = mstd(train_data)
        z_normalize(train_data, mean, stdev)

        # Create test data
        test_data = []
        for row in fold:
            row_copy = list(row)
            test_data.append(row_copy)
            row_copy[-1] = None
        z_normalize(test_data, mean, stdev)

        Y_train = []
        for i in range(len(train_data)):
            Y_train.append(train_data[i][-1])

        train_data = numpy.array(train_data)
        train_data = train_data[:, :-1]
        train_data = list(train_data)
        train_data = add_constant_feature(train_data)

        test_data = numpy.array(test_data)
        test_data = test_data[:, :-1]
        test_data - list(test_data)
        test_data = add_constant_feature(test_data)

        predicted_test, predicted_train, = linear_regression(train_data, test_data, Y_train)
        actual = [row[-1] for row in fold]
        for i in range(len(actual)):
            actual[i] = (actual[i] - mean[-1]) / stdev[-1]
        rmse = calculate_rmse(actual, predicted_test)
        rmse_test.append(rmse)

        rmse = calculate_rmse(Y_train, predicted_train)
        rmse_train.append(rmse)

    return rmse_test, rmse_train, sse


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
# Main Function
def main():
    seed(1)
    datasets_list = ['housing.csv', 'yachtData.csv']
    learning_rates = [0.0004, 0.001, 0.0007]
    tolerance = [0.005, 0.001, 0.0001]
    index = 0

    for dataset in datasets_list:
        print("%s DATASET" % dataset)
        file = dataset
        ds = load_csv(file)

        n_folds = 10
        l_rate = learning_rates[index]
        n_epoch = 1000

        score_test, score_train, SSE = lrAlgorithm(dataset, ds, n_folds, l_rate, n_epoch, tolerance[index])
        index += 1

        print('RMSE_test: %s' % score_test)
        print('Mean RMSE_test: %.3f' % (sum(score_test) / float(len(score_test))))

        print('RMSE_train: %s' % score_train)
        print('Mean RMSE_train: %.3f' % (sum(score_train) / float(len(score_train))))

        print("")


# =============================================================================#
if __name__ == '__main__':
    main()
