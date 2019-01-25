from operator import itemgetter
from random import randrange
import random
from math import log, ceil
import numpy
import pandas as pd
import statistics
import  matplotlib.pyplot as plt
import math
random.seed(1)


# ------------------------------------------------------------------------------------------------------------
class decisionNode():
    # Constructor for the decisionNode class
    def __init__(self, col=-1, value=None, results=None, leftTree=None, rightTree=None):
        self.col = col  # column index of criteria being tested
        self.value = value  # value of that particular test case
        self.results = results  # dict of results for a branch, None for everything except endpoints
        self.leftTree = leftTree  # true decision nodes
        self.rightTree = rightTree  # false decision nodes

    # ------------------------------------------------------------------------------------------------------------
    # Load a CSV file
    def load_csv(self, filename):
        data = pd.read_csv(filename)
        dataset = data.values
        # print("Dataframe's size: ",dataset.shape)
        return dataset

    # ------------------------------------------------------------------------------------------------------------
    # Number of rows of data
    def data_instances(self, data):
        return (len(data) + 1)

    # ------------------------------------------------------------------------------------------------------------
    # Traverse the whole tree and then predict the values of the appropriate leaf node
    def predict(self, test, tree):
        if (tree.col == -1):
            return ((tree.results))
        else:
            if (test[(tree.col)] >= (tree.value)):
                if isinstance(tree.leftTree, decisionNode):
                    return self.predict(test, tree.leftTree)
                else:
                    return tree.leftTree
            else:
                if isinstance(tree.rightTree, decisionNode):
                    return self.predict(test, tree.rightTree)
                else:
                    return tree.rightTree

    # ------------------------------------------------------------------------------------------------------------
    # Calculate the predicted values for the test set
    def predictions_for_DT(self, tree, test_data):
        predictions = list()
        for row in test_data:
            predicted_value = self.predict(row, tree)
            predictions.append(predicted_value)
        return (predictions)

    # ------------------------------------------------------------------------------------------------------------
    # Calculate the number of counts of the target column
    def uniqueCounts(self, train_data):
        results = {}
        if len(train_data) == 0:
            results = {}
        else:
            for row in train_data:
                r = row[len(row) - 1]
                if r not in results:
                    results[r] = 0
                results[r] += 1
        return results

    # ------------------------------------------------------------------------------------------------------------
    def sq_error_value(self, train_data, mean_value):
        squared_error = 0.0
        for row in range(len(train_data)):
            squared_error = squared_error + (mean_value - train_data[row][-1]) ** 2
        return squared_error

    # ------------------------------------------------------------------------------------------------------------
    # Divide the set into two branches accoding to the split value
    def divideSet(self, sorted_data, feature, value):
        # Handling nominal values
        split_function = lambda row: row[feature] >= value
        set1 = [row for row in sorted_data if numpy.all(split_function(row))]  # True values
        set2 = [row for row in sorted_data if numpy.all(not split_function(row))]  # False values
        return set1, set2

    # ------------------------------------------------------------------------------------------------------------
    # Build the decision tree
    def buildTree(self, sorted_data, min_size):

        sum = 0.0
        for row in range(len(sorted_data)):
            sum = sum + sorted_data[row][-1]
        mean_value = sum/len(sorted_data)

        if (len(sorted_data) == 0):
            return decisionNode()
        current_error = self.sq_error_value(sorted_data, mean_value)

        best_error = math.inf
        best_sets = None
        count_of_features = len(sorted_data[0]) - 1
        index = 0

        # For each feature calculate the split values and choose the best
        for feature in range(0, count_of_features):
            y = sorted_data[0][count_of_features]

            for i in range(len(sorted_data)):
                set1, set2 = self.divideSet(sorted_data, feature, sorted_data[i][feature])

                # Sum of squared error
                if (len(set1) == 0 or len(set2) == 0):
                    p = 0
                else:
                    p1 = float(len(set1)) / len(sorted_data)
                    p2 = float(len(set2)) / len(sorted_data)
                    sum1=0.0
                    for row in range(len(set1)):
                        sum1 = sum1 + set1[row][-1]
                    sum2=0.0
                    for row in range(len(set2)):
                        sum2 = sum2 + set2[row][-1]
                    mean_value1 = sum1 / len(set1)
                    mean_value2 = sum2 / len(set2)
                    error = p1 * self.sq_error_value(set1, mean_value1) +\
                            p2 * self.sq_error_value(set2, mean_value2)

                    if (error < best_error and len(set1) >= min_size and len(set2) >= min_size):
                        best_error = error
                        best_criteria = (feature, sorted_data[i][feature])
                        best_sets = (set1, set2)

        # Choosing the branches of the decision tree
        if best_error < current_error:
            trueBranch = self.buildTree(best_sets[0], min_size)
            falseBranch = self.buildTree(best_sets[1], min_size)
            return decisionNode(col=best_criteria[0], value=best_criteria[1],
                                leftTree=trueBranch, rightTree=falseBranch)
        else:
            return decisionNode(col=-1, results=mean_value )  # Leaf

            # ------------------------------------------------------------------------------------------------------------

    # Print the decision tree
    def printTree(self, tree, indent=''):
        if tree.results != None:
            # Print the leaf
            print(str(tree.results))
        else:
            # Print the node
            print('Column' + str(tree.col) + " : " + str(tree.value) + '? ')
            # Print all the branches
            print(
                indent + 'True->', end="")
            self.printTree(tree.leftTree, indent + '  ')
            print(
                indent + 'False->', end="")
            self.printTree(tree.rightTree, indent + '  ')

    # --------------------------------------------------------------------------#

    # Construct the decision tree and return the predicted values
    def decisionTree(self, sorted_data, test_data, X_train, min_size):
        tree = self.buildTree(sorted_data, min_size)
        predictions = self.predictions_for_DT(tree, test_data)
        predictions_train = self.predictions_for_DT(tree, X_train)
        return predictions, predictions_train

    # --------------------------------------------------------------------------#
    # Calculate the accuracy from actual and predicted values
    def accuracy_calculation(self, actual, predicted):
        score = 0.0
        for i in range(len(actual)):
            score = score + (actual[i] - predicted[i]) ** 2
        # accuracy = score / float(len(actual)) * 100.0
        return score

    # --------------------------------------------------------------------------#
    # Create folds of the dataset according to the n_folds value
    def split(self, dataset, n_folds):
        data_set_split = []
        data = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for i in range(n_folds):
            fold = []
            while len(fold) < fold_size:
                index = randrange(len(data))
                fold.append(data.pop(index))
            data_set_split.append(fold)
        return data_set_split

    # --------------------------------------------------------------------------#
    # Calculate the cross validation splits of the dataset for training and testing the data
    def cv_split(self, dataset, n_folds, min_size):
        # Generate folds according to the value of n_folds
        folds = self.split(dataset, n_folds)
        scores = list()
        scores_train = list()
        index = 0
        X = []
        Y = []
        for fold in folds:
            # Create train data
            train_data = list(folds)
            train_data = numpy.array(train_data)
            train_data = numpy.delete(train_data, (index), axis=0)
            index += 1
            train_data = train_data.tolist()
            train_data = sum(train_data, [])

            # Create test data
            test_data = []
            for row in fold:
                row_copy = list(row)
                test_data.append(row_copy)
                row_copy[-1] = None

            # Sort the training data according to its target column
            sorted_data = sorted(train_data, key=itemgetter(len(train_data[0]) - 1))

            # Calculate accuracy score
            X_train = train_data
            Y_train = []

            for i in range(len(train_data)):
                Y_train.append(train_data[i][-1])

            #Predictions
            actual = [row[-1] for row in fold]
            X.append(actual)
            predicted, predicted_train = self.decisionTree(sorted_data, test_data, X_train, min_size)
            Y.append(predicted)
            accuracy = self.accuracy_calculation(actual, predicted)
            scores.append(accuracy)
            accuracy = self.accuracy_calculation(Y_train, predicted_train)
            scores_train.append(accuracy)
        return X, Y, scores, scores_train


# --------------------------------------------------------------------------#
# MAIN FUNCTION
# HOUSING DATASET

def main():

    # Create an object of decisionNode class to access its various functions
    print("HOUSING")
    instance1 = decisionNode()

    # Loading the csv file of iris dataset
    housing_file = 'housing.csv'
    dataset_iris = instance1.load_csv(housing_file)
    size_of_data = instance1.data_instances(dataset_iris)

    N = [0.05, 0.10, 0.15, 0.20]

    avg_score = []
    avg_score_train = []
    Act = []
    Pre = []

    # Calculating the accuracy scores for various n values
    for i in range(len(N)):
        n_folds = 10
        min_size = ceil(size_of_data * N[i])
        eta = N[i]
        X, Y, score, score_train = instance1.cv_split(dataset_iris, n_folds, min_size)
        print("\nTesting Scores across all the folds")
        print(score)
        print("Average Score for min_size ", end="")
        print(min_size, end="")
        print(" and eta ", end="")
        print(eta)
        avg_score.append(sum(score) / float(len(score)))
        print(avg_score[i])
        print("Standard deviation ", end="")
        print(statistics.stdev(score))

        print("Training  Scores across all the folds")
        print(score_train)
        avg_score_train.append(sum(score_train) / float(len(score_train)))
        print("Average score")
        print(avg_score_train[i])
        print("Standard deviation ", end="")
        print(statistics.stdev(score_train))
        print("")

    #------------------------------------------------------------------

    plt.subplot(211)
    x1 = avg_score
    x2 = avg_score_train
    y = N
    plt.scatter(x1,y)
    plt.scatter(x2,y)
    plt.title("SSE comparison for HOUSING-binary")
    plt.xlabel("SSE")
    plt.ylabel("eta_min")
    plt.legend(["Test acc", "Train acc"])
    plt.tight_layout()

    plt.subplot(212)
    x1 = avg_score
    x2 = avg_score_train
    x2 = [x / 9 for x in x2]
    y = N
    plt.scatter(x1,y)
    plt.scatter(x2,y)
    plt.title("SSE comparison for HOUSING-binary normalized form")
    plt.xlabel("SSE")
    plt.ylabel("eta_min")
    plt.legend(["Test acc", "Train acc"])
    plt.tight_layout()

    plt.show()

#------------------------------------------------------------------

if __name__ == "__main__":
    main()



