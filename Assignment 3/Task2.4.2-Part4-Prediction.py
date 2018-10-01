import numpy
import pandas as pd
import operator
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


# ================================================================================#
# Load a CSV file
def load_csv(filename):
    data = pd.read_csv(filename, header=None, delimiter=' ')
    dataset = data.values
    return dataset


# ================================================================================#
# Convert 2 lists to a dictionary
def list_to_dict(word, freq):
    dct = dict()
    index = 0

    for item in word:
        if item in dct:
            dct[item] = dct[item] + freq[index]
        else:
            dct[item] = freq[index]
        index += 1

    return dct


# ================================================================================#
# Get dict of word-freq list
def get_word_freq_dict(data):
    word_id_list = []
    freq_list = []

    for row in data:
        word_id_list.append(row[1])
        freq_list.append(row[2])
    word_freq_dict = list_to_dict(word_id_list, freq_list)

    return word_freq_dict


# ================================================================================#
# Generate a doc-word dictionary
def get_doc_word_dict(data):
    doc_word_dict = {}
    for row in data:
        if row[0] in doc_word_dict:
            doc_word_dict[row[0]].append(row[1])
        else:
            doc_word_dict[row[0]] = [row[1]]
            # print(doc_word_dict)
    return doc_word_dict


# ================================================================================#
# Select the top |V| for training the data
def get_top_words_ids(vocabulary_size, word_freq_list, temp):
    top_words_ids = []
    for v in range(vocabulary_size):
        top_words_ids.append(word_freq_list[temp + v][0])
    return top_words_ids


# ================================================================================#
# Create train data features vector: X_train
def create_X_train_features(vocabulary_size, documents_size, top_words_ids, doc_word_dict):

    X_train = [[0 for x in range(vocabulary_size)] for y in range(documents_size)]
    # Include a word only if it is in top |V| words
    for i in range(documents_size):
        for v in range(vocabulary_size):
            if top_words_ids[v] in doc_word_dict[i + 1]:
                X_train[i][v] = 1
    return X_train


# ================================================================================#
# Create train data labels vector: Y_train
def create_train_labels(dir_path):
    data = load_csv(dir_path + '/' + 'train_label.csv')
    labels = []
    for row in data:
        labels.append(row)

    # Generate one hot encoded labels
    labels = array(labels)
    labels = numpy.ravel(labels)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded_labels = onehot_encoder.fit_transform(integer_encoded)

    return labels, onehot_encoded_labels


# ================================================================================#
# Generate the yik dictionary
def create_yik_dict(labels):
    yik_dict = {}
    for item in labels:
        if item in yik_dict:
            yik_dict[item] = yik_dict[item] + 1
        else:
            yik_dict[item] = 1
    return yik_dict


# ================================================================================#
# Calculate the pi values for all classes: pi(k)
def calculate_pi_values(yik_dict, N, onehot_encoded_labels):
    pi_values = [0.0 for i in range(len(yik_dict))]

    for k in range(len(yik_dict)):
        for i in range(N):
            if onehot_encoded_labels[i][k] == 1:
                pi_values[k] += 1
        pi_values[k] = pi_values[k] / N
    return pi_values


# ================================================================================#
# Calculate the theta values: θ(j,k)
def calculate_theta_values(yik_dict, vocabulary_size, N, pi_values, X_train, onehot_encoded_labels):
    theta_jk = [[0.0 for k in range(len(yik_dict))] for j in range(vocabulary_size)]

    for k in range(len(yik_dict)):
        den = pi_values[k] * N
        for j in range(vocabulary_size):
            num = 0.0
            for i in range(N):
                if (X_train[i][j] == 1) and (onehot_encoded_labels[i][k] == 1):
                    num += 1
            theta_jk[j][k] = num / den
    return theta_jk


# ================================================================================#
# Generate doc-word dictionary for test data
def get_doc_word_dict_test(top_words_ids, dir_path):
    test_data = load_csv(dir_path + '/' + 'test_data.csv')
    doc_word_dict = {}
    for row in test_data:
        if row[1] in top_words_ids:
            if row[0] in doc_word_dict:
                doc_word_dict[row[0]].append(row[1])
            else:
                doc_word_dict[row[0]] = [row[1]]
    return doc_word_dict

# ================================================================================#
# Apply the Naive Bayes formula for predicting multiple classes
def apply_naive_bayes(yik_dict, doc_word_dict, pi_values, top_words_ids, theta_jk, num, den, predicted_labels, Y_pred):
    for index in range(len(doc_word_dict)):
        for k in range(len(yik_dict)):
                if den[index]==0:
                    predicted_labels[index][k] = 0
                else:
                    predicted_labels[index][k] = num[index][k] / den[index]
        Y_pred[index] = (predicted_labels[index].index(max(predicted_labels[index])) + 1)
    return Y_pred, predicted_labels


# ================================================================================#
# Prediction function
def predict_the_classes(yik_dict, doc_word_dict, pi_values, top_words_ids, theta_jk, num, den, dir_path):
    # Loading the test file
    test_labels = load_csv(dir_path + '/' + 'test_label.csv')

    # Initialize the Y_act and Y_pred lists
    Y_pred = [0 for i in range(len(test_labels))]
    Y_actual = []
    for row in test_labels:
        Y_actual.append(row)

    # Initialize the Y_pred lists
    predicted_labels = [[0.0 for k in range(len(yik_dict))] for i in range(len(doc_word_dict))]

    # Apply the Naive Bayes formula for predicting multiple classes
    Y_pred, predicted_labels = apply_naive_bayes(yik_dict, doc_word_dict, pi_values, top_words_ids, theta_jk, num, den,
                                                 predicted_labels,
                                                 Y_pred)

    return Y_pred, Y_actual

# ================================================================================#
# Calculate the den values for the formula of class prediction
def calculate_den_values_for_prediction(doc_word_dict, yik_dict, pi_values, top_words_ids, theta_jk, flag):
    den = [0.0 for i in range(len(doc_word_dict))]
    temp_den = [[] for i in range(len(doc_word_dict))]

    if flag==1:
        for index in range(len(doc_word_dict)):
            deno = 0.0
            temp = [0.0 for i in range(len(yik_dict))]
            for k in range(len(yik_dict)):
                num = pi_values[k]
                # deno = 0.0
                if index + 1 in doc_word_dict:
                    for i in numpy.ravel(list(doc_word_dict[index + 1])):
                        idx = top_words_ids.index(i)
                        num *= theta_jk[idx][k]
                    temp[k] = num
                    deno += num
            den[index] = deno
            temp_den[index] = temp

    else:
        for index in range(len(doc_word_dict)):
            deno = 0.0
            temp = [1.0 for i in range(len(yik_dict))]
            for k in range(len(yik_dict)):
                num = 1.0
                # deno = 0.0
                # for i in range(len(numpy.ravel(list(doc_word_dict.values())))):
                if index + 1 in doc_word_dict:
                    for i in numpy.ravel(list(doc_word_dict[index + 1])):
                        idx = top_words_ids.index(i)
                        num *= theta_jk[idx][k]
                    temp[k] = num
                    deno += num
            den[index] = deno
            temp_den[index] = temp

    return den, temp_den


# ================================================================================#

def calculate_num_values_for_prediction(yik_dict, doc_word_dict, pi_values, top_words_ids, theta_jk, flag, den):
    temp_num = [[] for i in range(len(doc_word_dict))]

    if flag==1:
        for index in range(len(doc_word_dict)):
            deno = 0.0
            temp = [1.0 for i in range(len(yik_dict))]
            for k in range(len(yik_dict)):
                num = pi_values[k]
                # deno = 0.0
                if index + 1 in doc_word_dict:
                    for i in numpy.ravel(list(doc_word_dict[index + 1])):
                        idx = top_words_ids.index(i)
                        num *= theta_jk[idx][k]
                    temp[k] = num
            temp_num[index] = temp

    else:
        for index in range(len(doc_word_dict)):
            deno = 0.0
            temp = [1.0 for i in range(len(yik_dict))]
            for k in range(len(yik_dict)):
                num = 1.0
                # deno = 0.0
                # for i in range(len(numpy.ravel(list(doc_word_dict.values())))):
                if index + 1 in doc_word_dict:
                    for i in numpy.ravel(list(doc_word_dict[index + 1])):
                        idx = top_words_ids.index(i)
                        num *= theta_jk[idx][k]
                    temp[k] = num
            temp_num[index] = temp


    return temp_num


# ================================================================================#
# MAIN Function
def main():

    dir_path = 'C:/Users/prati/Desktop/Assignment 3/20NG_Data'
    data = load_csv(dir_path + '/' + 'train_data.csv')
    data = numpy.array(data).tolist()
    word_freq_dict = get_word_freq_dict(data)

    vocabulary_size_list = [50000]

    # pi_values_filenames = ['Top100_pi_values.csv']
    pi_values_filenames = ['Top50000_pi_values.csv', 'Top50000_pi_values.csv', 'Top50000_pi_values.csv']

    theta_values_filenames = ['Top50000_theta_values-Part1.csv', 'Top50000_theta_values-Part2.csv', 'Top50000_theta_values-Part3.csv',]

    # pi_values_filenames = ['TopAll_pi_values.csv', 'TopAll_pi_values.csv', 'TopAll_pi_values.csv']
    # theta_values_filenames = ['Top100_theta_values.csv']
    # theta_values_filenames = ['TopAll_theta_values-Part1.csv',
    #                                          'TopAll_theta_values-Part2.csv',
    #                                          'TopAll_theta_values-Part3.csv']

    counter = 0
    vocab = [20000, 20000, vocabulary_size_list[0] - 40000]
    temp=0

    # Load the training data file
    dir_path = 'C:/Users/prati/Desktop/Assignment 3/20NG_Data'
    dir = 'C:/Users/prati/Desktop/Assignment 3/MB- Theta and Pi values files/'
    data = load_csv(dir_path + '/' + 'train_data.csv')
    data = numpy.array(data).tolist()

    # Generate a word-freq dictionary and then sort it
    word_freq_dict = get_word_freq_dict(data)
    word_freq_list = sorted(word_freq_dict.items(), key=operator.itemgetter(1), reverse=True)

    flag=1
    num_list = []
    den_list = []

    for vocabulary_size in vocab:

        # Select the top |V| for training the data
        top_words_ids = get_top_words_ids(vocabulary_size, word_freq_list, temp)
        temp += vocabulary_size

        # Create train data labels vector: Y_train
        labels, onehot_encoded_labels = create_train_labels(dir_path)

        # Generate the yik dictioanry
        yik_dict = create_yik_dict(labels)


        # Generate doc-word dictionary for test data
        doc_word_dict = get_doc_word_dict_test(top_words_ids, dir_path)

        #Get pik values
        f = load_csv(dir + pi_values_filenames[counter])
        pi_k = numpy.array(f).tolist()
        pi_k = sum(pi_k, [])

        # Get θjk values
        f = load_csv(dir + theta_values_filenames[counter])
        theta_jk = numpy.array(f).tolist()

        # Calculate the denominator values for the formula of class prediction
        den, temp_den = calculate_den_values_for_prediction(doc_word_dict, yik_dict, pi_k, top_words_ids, theta_jk, flag)
        temp_den = numpy.array(temp_den)
        den_list.append(temp_den)

        # Calculate the numerator values for the formula of class prediction
        temp_num = calculate_num_values_for_prediction()
        temp_num = numpy.array(temp_num)
        num_list.append(temp_num)

        if (counter==len(vocab)):
            den_list = numpy.array(den_list)
            den = (numpy.prod(den_list, axis=0))

            den_i = []
            for d in den:
                den_i.append(sum(d))

            num_list = numpy.array(num_list)
            num = (numpy.prod(num_list, axis=0))

            Y_pred, Y_actual = predict_the_classes(yik_dict, doc_word_dict, pi_k, top_words_ids, theta_jk, num, den_i, dir_path)

            f = open('Y_predicted_values_Top50000.csv', 'w')
            for y in Y_pred:
                f.write(y+'\n')
            f.close()

        counter += 1
        print(counter)
        flag += 1

# ================================================================================#

# Start of the python script
if __name__ == '__main__':
    main()
