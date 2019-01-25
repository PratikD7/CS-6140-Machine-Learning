import numpy
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

# ================================================================================#
# Load a CSV file
def load_csv(filename):
    data = pd.read_csv(filename, header=None, delimiter=' ')
    dataset = data.values
    return dataset

#========================================================================#
def main():
    dir_path1 = 'C:/Users/prati/PycharmProjects/MlPractice/Assignment 3/20NG_Data/'
    dir_path2 = 'C:/Users/prati/PycharmProjects/MlPractice/Assignment 3/Y Predicted values Smoothing/'
    dir_path3 = 'C:/Users/prati/PycharmProjects/MlPractice/Assignment 3/Y Predicted values - SmoothingME/'
    data = load_csv(dir_path1 + 'test_label.csv')
    data = numpy.array(data).tolist()
    Y_actual = []
    for d in data:
        Y_actual.append(d)
    Y_actual = numpy.array(Y_actual).tolist()
    Y_actual = sum(Y_actual, [])

    vocab = [1]
    accuracies1 = []
    accuracies2 = []

    classes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    classes2 = [i + 0.35 for i in classes]

    Y_pred_files_1 = ['Y_predicted_values_smoothingTopAll.csv',]

    Y_pred_files_2 = ['Y_predicted_values_TopAll_MEvent_Smoothing.csv',]

    for v in range(len(vocab)):
        Y_pred = []
        f = open(dir_path2 + Y_pred_files_1[v],'r')
        for row in f:
            Y_pred.append(int(row[:-1]))
        p1, r1, f, s = precision_recall_fscore_support(Y_actual, Y_pred)
        p1 = p1[:20]
        r1 = r1[:20]

        Y_pred = []
        f = open(dir_path3 + Y_pred_files_2[v], 'r')
        for row in f:
            Y_pred.append(int(row[:-1]))
        p2, r2, f, s = precision_recall_fscore_support(Y_actual, Y_pred)
        p2 = p2[:20]
        r2 = r2[:20]

    plt.bar(classes, p1, width=0.35, label='Multivariate Bernoulli')
    plt.bar(classes2, p2, width=0.35, label='Multinomial Event')
    plt.xlabel('Classes')
    plt.title('Precision values of both models with Smoothing against the classes')
    plt.ylabel('Testing Accuracy')
    plt.xticks(classes+[1], (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20))
    plt.grid('on')
    plt.legend(['Multivariate Bernoulli model', 'Multinomial Event model'])
    plt.show()
    plt.bar(classes, r1, width=0.35, label='Multivariate Bernoulli')
    plt.bar(classes2, r2, width=0.35, label='Multinomial Event')
    plt.xlabel('Classes')
    plt.title('Recall values of both models with smoothing against the classes')
    plt.ylabel('Testing Accuracy')
    plt.xticks(classes + [1], (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20))
    plt.grid('on')
    plt.legend(['Multivariate Bernoulli model', 'Multinomial Event model'])
    plt.show()

# ========================================================================#
if __name__ == '__main__':
    main()