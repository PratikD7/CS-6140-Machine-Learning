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
    print("Simple event model")
    dir_path1 = 'C:/Users/prati/PycharmProjects/MlPractice/Assignment 3/20NG_Data/'
    dir_path2 = 'C:/Users/prati/PycharmProjects/MlPractice/Assignment 3/Y Predicted values - SmoothingME/'
    data = load_csv(dir_path1 + 'test_label.csv')
    data = numpy.array(data).tolist()
    Y_actual = []
    for d in data:
        Y_actual.append(d)
    Y_actual = numpy.array(Y_actual).tolist()
    Y_actual = sum(Y_actual, [])

    vocab = [1,2,3,4,5,6,7,8,9,10,11]
    accuracies = []

    Y_pred_files = ['Y_predicted_values_Top100_MEvent_Smoothing.csv',
                    'Y_predicted_values_Top500_MEvent_Smoothing.csv',
                    'Y_predicted_values_Top1000_MEvent_Smoothing.csv',
                    'Y_predicted_values_Top2500_MEvent_Smoothing.csv',
                    'Y_predicted_values_Top5000_MEvent_Smoothing.csv',
                    'Y_predicted_values_Top7500_MEvent_Smoothing.csv',
                    'Y_predicted_values_Top10000_MEvent_Smoothing.csv',
                    'Y_predicted_values_Top12500_MEvent_Smoothing.csv',
                    'Y_predicted_values_Top25000_MEvent_Smoothing.csv',
                    'Y_predicted_values_Top50000_MEvent_Smoothing.csv',
                    'Y_predicted_values_TopAll_MEvent_Smoothing.csv',]

    for v in range(len(vocab)):
        Y_pred = []
        f = open(dir_path2 + Y_pred_files[v],'r')
        for row in f:
            Y_pred.append(int(row[:-1]))
        temp = accuracy_score(Y_actual, Y_pred)
        accuracies.append(temp)
        print(temp)

    plt.plot(vocab, accuracies, marker='o')
    # plt.show()

# ========================================================================#
if __name__ == '__main__':
    main()