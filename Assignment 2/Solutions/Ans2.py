from random import randrange, seed
import math
import numpy
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import random


# =============================================================================#
# Load a CSV file
def load_csv(filename):
    data = pd.read_csv(filename)
    dataset = data.values
    return dataset


# =============================================================================#
# Divide the dataset into n folds for training and cross validation
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
    yhat = coef[0]
    for i in range(len(row) - 1):
        yhat += coef[i + 1] * row[i]
    return yhat


# =============================================================================#
# Gradient Descent algorithm to find the values of co-efficients
def gradient_descent(train_data, l_rate, n_epoch, tolerance):
    temp_sum_error = math.inf
    rmse_list = []
    coef = [0.0 for i in range(len(train_data[0]))]

    # No of iterations
    for e in range(n_epoch):
        sum_sq_error = 0
        # Calculate coefficients
        for row in train_data:
            yhat = predict(row, coef)
            error = yhat - row[-1]
            sum_sq_error += error ** 2
            coef[0] = coef[0] - l_rate * error * 1  # row[i] = 1 Presumably
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
        # Calculate root mean square values
        rmse = math.sqrt(sum_sq_error / len(train_data))
        rmse_list.append(rmse)
        # print('>epoch=%d, lrate=%.4f, error=%.3f, rmse=%.5f' % (e, l_rate, sum_sq_error, rmse))
        prev_sum_error = temp_sum_error
        temp_sum_error = rmse
        # Tolerance values stops the execution if it is below a certain value
        if abs(prev_sum_error - rmse) < tolerance:
            break
    return coef, sum_sq_error, rmse_list


# =============================================================================#
# Linear regression algorithm to find predicted values, Root mean square error and sum of squared errors
def linear_regression(train_data, test_data, l_rate, n_epoch, tolerance):
    predictions_test = list()
    predictions_train = list()
    coeff, sum_sq_error, rmse_list = gradient_descent(train_data, l_rate, n_epoch, tolerance)

    # Predict for test data
    for row in test_data:
        yhat = predict(row, coeff)
        predictions_test.append(yhat)

    # Predict for train data
    for row in train_data:
        yhat = predict(row, coeff)
        predictions_train.append(yhat)

    return (predictions_test, predictions_train, sum_sq_error, rmse_list)


# =============================================================================#
def plot_rmse_values(index, random_fold, rmse_train, dataset):
    # Plot rmse values for any random fold
    if index == random_fold:
        plt.subplot(111)
        x1 = rmse_train
        y1 = list(range(1, len(rmse_train) + 1))
        plt.plot(y1, x1, marker='o')
        # plt.scatter(x2, y2)
        plt.title('Training RMSE for %s' % dataset[:-4])
        plt.xlabel('# of iterations for convergence')
        plt.ylabel('RMSE')
        plt.tight_layout()


# =============================================================================#
# Make training and testing data and apply main linear regression algorithm
def lrAlgorithm(dataset, ds, n_folds, l_rate, n_epoch, tolerance):
    folds = cv_split(ds, n_folds)
    rmse_test = list()
    rmse_train = list()
    sse = list()
    index = 0
    random_fold = random.randint(1, n_folds)

    for fold in folds:
        # Create train data
        train_data = list(folds)
        train_data = numpy.array(train_data)
        train_data = numpy.delete(train_data, (index), axis=0)
        index += 1
        train_data = train_data.tolist()
        train_data = sum(train_data, [])
        mean, stdev = mstd(train_data)
        # Z-normalize the training data
        z_normalize(train_data, mean, stdev)

        # Create test data
        test_data = []
        for row in fold:
            row_copy = list(row)
            test_data.append(row_copy)
            row_copy[-1] = None
        # mean, stdev = mstd(train_data)
        # Z-normalize the training data
        z_normalize(test_data, mean, stdev)

        Y_train = []
        for i in range(len(train_data)):
            Y_train.append(train_data[i][-1])

        # Apply linear regression algorithm
        predicted_test, predicted_train, sum_sq_error, rmse_list = linear_regression(train_data, test_data, l_rate,
                                                                                     n_epoch, tolerance)
        actual = [row[-1] for row in fold]
        for i in range(len(actual)):
            actual[i] = (actual[i] - mean[-1]) / stdev[-1]
        rmse1 = calculate_rmse(actual, predicted_test)
        rmse_test.append(rmse1)

        rmse2 = calculate_rmse(Y_train, predicted_train)
        rmse_train.append(rmse2)
        sse.append(sum_sq_error)

        plot_rmse_values(index, random_fold, rmse_list, dataset)

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
#Main function
def main():
    seed(1)
    datasets_list = ['housing.csv', 'yachtData.csv', 'concreteData.csv']
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

        print('Sum of Squared Error: %s' % SSE)
        print('Mean SSE: %.3f' % (sum(SSE) / float(len(SSE))))
        print("Standard deviation of SSE across the folds: %f" % statistics.stdev(SSE))
        plt.show()
        print("")


# =============================================================================#
if __name__ == '__main__':
    main()
