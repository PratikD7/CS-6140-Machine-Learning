import numpy
import pandas as pd
from sklearn.metrics import accuracy_score
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

    vocab = [100,500,1000,2500,5000,7500,10000,12500,25000,50000,53958]
    accuracies1 = []
    accuracies2 = []

    Y_pred_files_1 = ['Y_predicted_values_smoothingTop100.csv',
                    'Y_predicted_values_smoothingTop500.csv',
                    'Y_predicted_values_smoothingTop1000.csv',
                    'Y_predicted_values_smoothingTop2500.csv',
                    'Y_predicted_values_smoothingTop5000.csv',
                    'Y_predicted_values_smoothingTop7500.csv',
                    'Y_predicted_values_smoothingTop10000.csv',
                    'Y_predicted_values_smoothingTop12500.csv',
                    'Y_predicted_values_smoothingTop25000.csv',
                    'Y_predicted_values_smoothingTop50000.csv',
                    'Y_predicted_values_smoothingTopAll.csv',]

    Y_pred_files_2 = ['Y_predicted_values_Top100_MEvent_Smoothing.csv',
                      'Y_predicted_values_Top500_MEvent_Smoothing.csv',
                      'Y_predicted_values_Top1000_MEvent_Smoothing.csv',
                      'Y_predicted_values_Top2500_MEvent_Smoothing.csv',
                      'Y_predicted_values_Top5000_MEvent_Smoothing.csv',
                      'Y_predicted_values_Top7500_MEvent_Smoothing.csv',
                      'Y_predicted_values_Top10000_MEvent_Smoothing.csv',
                      'Y_predicted_values_Top12500_MEvent_Smoothing.csv',
                      'Y_predicted_values_Top25000_MEvent_Smoothing.csv',
                      'Y_predicted_values_Top50000_MEvent_Smoothing.csv',
                      'Y_predicted_values_TopAll_MEvent_Smoothing.csv', ]

    for v in range(len(vocab)):
        Y_pred = []
        f = open(dir_path2 + Y_pred_files_1[v],'r')
        for row in f:
            Y_pred.append(int(row[:-1]))
        temp = accuracy_score(Y_actual, Y_pred)
        accuracies1.append(temp)

        Y_pred = []
        f = open(dir_path3 + Y_pred_files_2[v], 'r')
        for row in f:
            Y_pred.append(int(row[:-1]))
        temp = accuracy_score(Y_actual, Y_pred)
        accuracies2.append(temp)
        # print(temp)

    accuracies1 = [i*100 for i in accuracies1]
    accuracies2 = [i * 100 for i in accuracies2]

    plt.plot(vocab, accuracies1, marker='o')
    plt.plot(vocab, accuracies2, marker='o')
    plt.xlabel('Vocabulary size')
    plt.title('Smoothing - Accuracy of multivariate Bernoulli model against max vocabulary size ')
    plt.ylabel('Testing Accuracy')
    plt.grid('on')
    plt.legend(['Multivariate Bernoulli','Multinomial Event'])
    plt.show()

# ========================================================================#
if __name__ == '__main__':
    main()