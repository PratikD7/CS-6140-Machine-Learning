from operator import itemgetter
from random import randrange
import random
from math import log, ceil
import operator
import numpy
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
random.seed(1)


# ------------------------------------------------------------------------------------------------------------
class decisionNode():
    # Constructor for the decisionNode class
    def __init__(self, col=-1, value=None, results=None, leftTree=None, rightTree=None):
        self.col = col  # column index of criteria being tested
        self.value = value  # value of that particular test case (split threshold)
        self.results = results  # dict of results for a branch, None for everything except endpoints
        self.leftTree = leftTree  # Left branch
        self.rightTree = rightTree  # Right branch

    # ------------------------------------------------------------------------------------------------------------
    # Load a CSV file
    def load_csv(self, filename):
        data = pd.read_csv(filename)
        dataset = data.values
        return dataset

    # ------------------------------------------------------------------------------------------------------------
    # Number of rows of data
    def data_instances(self, data):
        return (len(data) + 1)

    # ------------------------------------------------------------------------------------------------------------
    # Traverse the whole tree and then predict the values of the appropriate leaf node
    def predict(self, test, tree):
        #If there'a a leaf node
        if (tree.col == -1):
            return (max(tree.results.items(), key=operator.itemgetter(1))[0])
        else:
            #Check for the threshold value and choose the branch accordingly
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
    # Calculate the number of unique counts of the target column of the data
    def uniqueCounts(self, train_data):
        results = {}
        if train_data is None:
            results = {}
        else:
            for row in train_data:
                r = row[len(row) - 1]
                if r not in results:
                    results[r] = 0
                results[r] += 1
        return results

    # ------------------------------------------------------------------------------------------------------------
    # Calculate the entropy of the training data
    def entropy(self, train_data):
        log2 = lambda x: log(x) / log(2)
        results = self.uniqueCounts(train_data)
        # Calculating the entropy for only threshold values
        entropy_value = 0.0
        for r in results.keys():
            prob = float(results[r]) / len(train_data)
            entropy_value = entropy_value - prob * log2(prob)
        return entropy_value

    # ------------------------------------------------------------------------------------------------------------
    # Divide the set into two branches accoding to the split value
    def divideSet(self, sorted_data, feature, split_values, min_size):
        best_gain = 0.0
        current_score = self.entropy(sorted_data)
        best_value = None
        best_set1 = None
        best_set2 = None

        for value in split_values:
            split_function = None
            # Handling numeric values
            if (isinstance(value, int) or isinstance(value, float)):
                split_function = lambda row: row[feature] >= value

            set1 = [row for row in sorted_data if split_function(row)]  # True values
            set2 = [row for row in sorted_data if not split_function(row)]  # False values

            # Information Gain for selecting the best threshold value
            p = float(len(set1)) / len(sorted_data)
            gain = current_score - p * self.entropy(set1) - (1 - p) * self.entropy(set2)
            if (gain > best_gain and len(set1) >= min_size and len(set2) >= min_size):
                best_gain = gain
                best_set1 = set1
                best_set2 = set2
                best_value = value
        return best_set1, best_set2, best_value

    # ------------------------------------------------------------------------------------------------------------
    # Build the decision tree
    def buildTree(self, sorted_data, min_size):

        if (len(sorted_data) == 0):
            return decisionNode()
        current_score = self.entropy(sorted_data)
        best_gain = 0.0
        best_sets = None
        split_values = []
        count_of_features = len(sorted_data[0]) - 1
        index = 0

        # For each feature calculate the split values and choose the best
        for feature in range(0, count_of_features):
            y = sorted_data[0][count_of_features]

            #Choosing only those threshold values where value of target changes
            for row in range(len(sorted_data)):
                if (y == sorted_data[row][len(sorted_data[0]) - 1]):
                    continue
                else:
                    split_values.append(statistics.mean([sorted_data[row - 1][index], sorted_data[row][index]]))
                    y = sorted_data[row][len(sorted_data[0]) - 1]
            index += 1
            set1, set2, best_value = self.divideSet(sorted_data, feature, split_values, min_size)
            split_values.clear()

            # Information Gain
            if (set1 == None or set2 == None):
                p = 0
            else:
                p = float(len(set1)) / len(sorted_data)
                gain = current_score - p * self.entropy(set1) - (1 - p) * self.entropy(set2)
                if (gain > best_gain and len(set1) >= min_size and len(set2) >= min_size):
                    best_gain = gain
                    best_criteria = (feature, best_value)
                    best_sets = (set1, set2)

        # Choosing the branches of the decision tree
        if best_gain > 0:
            trueBranch = self.buildTree(best_sets[0], min_size)
            falseBranch = self.buildTree(best_sets[1], min_size)
            return decisionNode(col=best_criteria[0], value=best_criteria[1], results=self.uniqueCounts(sorted_data),
                                leftTree=trueBranch, rightTree=falseBranch)
        else:
            return decisionNode(col=-1, results=self.uniqueCounts(sorted_data), )  # Leaf

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
        score = 0
        for i in range(len(actual)):
            if (actual[i] == predicted[i]):
                score += 1
        accuracy = score / float(len(actual)) * 100.0
        return accuracy

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
    def dt_algorithm(self, dataset, n_folds, min_size):
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

            #Create the decision tree and predict the value
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

def main():
    # Create an object of decisionNode class to access its various functions
    print("IRIS")
    instance1 = decisionNode()

    # Loading the csv file of iris dataset
    iris_file = 'iris.csv'
    dataset_iris = instance1.load_csv(iris_file)
    size_of_data = instance1.data_instances(dataset_iris)

    N = [0.05, 0.10, 0.15, 0.20]

    avg_score = []
    avg_score_train = []

    # Calculating the accuracy scores for various eta values
    for i in range(len(N)):
        n_folds = 10
        min_size = ceil(size_of_data * N[i])
        eta = N[i]
        X, Y, score, score_train = instance1.dt_algorithm(dataset_iris, n_folds, min_size)
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

    #Choosing the best eta value according to the scores
    best = []
    maxi = max(avg_score)
    j = 0
    for i in avg_score:
        if i == maxi:
            best.append(j)
        j += 1
    # print(best)
    print("Best values of eta min")
    for i in range(len(best)):
        eta = N[best[i]]
        print(eta, end=" ")
    print("\n")

    #Plot the training and testing accuracy against eta_min
    plt.subplot(211)
    x1 = avg_score
    x2 = avg_score_train
    y = N
    plt.scatter(x1, y)
    plt.scatter(x2, y)
    plt.title("Accuracy comparison for IRIS")
    plt.xlabel("Accuracy")
    plt.ylabel("eta_min")
    plt.legend(["Test acc", "Train acc"])
    plt.tight_layout()

    avg_score = []
    avg_score_train = []

    # --------------------------------------------------------------------------#
    # SPAMBASE DATASET
    print("SPAMBASE")
    instance2 = decisionNode()

    # Loading the csv file of spambase dataset
    spambase_file = 'spambase.csv'
    dataset_spam = instance2.load_csv(spambase_file)
    size_of_data = instance2.data_instances(dataset_spam)

    N = [0.05, 0.10, 0.15, 0.20, 0.25]

    # start = time.clock()
    # Calculating the accuracy scores for various n values
    for i in range(len(N)):
        n_folds = 10
        min_size = ceil(size_of_data * N[i])
        eta = N[i]
        X, Y, score, score_train = instance2.dt_algorithm(dataset_spam, n_folds, min_size)
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
        # print("")

        print("Training  Scores across all the folds")
        print(score_train)
        avg_score_train.append(sum(score_train) / float(len(score_train)))
        print("Average score")
        print(avg_score_train[i])
        print("Standard deviation ", end="")
        print(statistics.stdev(score_train))
        print("")

    #Choosing the best value of eta
    best = []
    maxi = max(avg_score)
    j = 0
    for i in avg_score:
        if i == maxi:
            best.append(j)
        j += 1
    # print(best)
    print("Best values of eta min")
    for i in range(len(best)):
        eta = N[best[i]]
        print(eta, end=" ")
    print("\n")

    # ------------------------------------------------------------------

    # Drawing confusion matrix for iris for eta = 0.05(best value)
    # eta = 0.05
    n_folds = 10
    size_of_data = instance1.data_instances(dataset_iris)
    min_size = ceil(size_of_data * eta)
    X, Y, score, score_train = instance1.dt_algorithm(dataset_iris, n_folds, min_size)
    X = sum(X, [])
    Y = sum(Y, [])
    results = confusion_matrix(Y, X, labels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    print("Confusion matrix for iris")
    y_actu = pd.Series(X, name='Actual')
    y_pred = pd.Series(Y, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print(df_confusion)

    # ------------------------------------------------------------------

    # Drawing confusion matrix for spambase for eta = 0.05(best value)
    # eta = 0.05
    n_folds = 10
    size_of_data = instance2.data_instances(dataset_spam)
    min_size = ceil(size_of_data * eta)
    X, Y, score, score_train = instance2.dt_algorithm(dataset_spam, n_folds, min_size)
    X = sum(X, [])
    Y = sum(Y, [])
    results = confusion_matrix(Y, X, labels=[1, 0])
    print("\nConfusion matrix for spambase")
    y_actu = pd.Series(X, name='Actual')
    y_pred = pd.Series(Y, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print(df_confusion)

    # ------------------------------------------------------------------

    #Plotting training and testing accuracy vs eta
    plt.subplot(212)
    x1 = avg_score
    x2 = avg_score_train
    y = N
    plt.scatter(x1, y)
    plt.scatter(x2, y)
    plt.title("Accuracy comparison for SPAMBASE")
    plt.xlabel("Accuracy")
    plt.ylabel("eta_min")
    plt.legend(["Test acc", "Train acc"])
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------
#Script starts here
if __name__ == "__main__":
    main()
