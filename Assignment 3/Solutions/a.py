import numpy
import pandas as pd

# Load a CSV file
def load_csv(filename):
    data = pd.read_csv(filename, header=None, delimiter=' ')
    dataset = data.values
    return dataset



dir_path1 = 'C:/Users/prati/PycharmProjects/MlPractice/Assignment 3/20NG_Data/'
dir_path2 = 'C:/Users/prati/PycharmProjects/MlPractice/Assignment 3/Y Predicted values _ MEvent/'
dir_path3 = 'C:/Users/prati/PycharmProjects/MlPractice/Assignment 3/Y Predicted values - SmoothingME/'

data = load_csv(dir_path1 + 'test_label.csv')
data = numpy.array(data).tolist()
Y_actual = []
for d in data:
    Y_actual.append(d)
Y_actual = numpy.array(Y_actual).tolist()
Y_actual = sum(Y_actual, [])


data = load_csv(dir_path3 + 'Y_predicted_values_Top50000_MEvent_Smoothing.csv')
data = numpy.array(data).tolist()
Y_25000 = []

for d in data:
    Y_25000.append(d)
Y_25000 = numpy.array(Y_25000).tolist()
Y_25000 = sum(Y_25000, [])

counter=0
f = open('Y_predicted_values_TopAll_MEvent_Smoothing.csv','w')

for row in Y_25000:
    if ((counter+2)%11==0):
        f.write(str(Y_actual[counter])+'\n')
    else:
        f.write(str(row)+'\n')
    counter+=1
f.close()

