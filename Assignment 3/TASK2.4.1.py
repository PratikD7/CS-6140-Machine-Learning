import numpy
import pandas as pd
import operator


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
# Write the obtained word-freq pair into the files
def write_wf_into_files(word_freq_dict):
    file_list = ['Top100_wordfreq.txt',
                 'Top500_wordfreq.txt',
                 'Top1000_wordfreq.txt',
                 'Top2500_wordfreq.txt',
                 'Top5000_wordfreq.txt',
                 'Top7500_wordfreq.txt',
                 'Top10000_wordfreq.txt',
                 'Top12500_wordfreq.txt',
                 'Top25000_wordfreq.txt',
                 'Top50000_wordfreq.txt',
                 'TopAll_wordfreq.txt', ]

    vocabulary_size_list = [100, 500, 1000, 2500, 5000, 7500, 10000, 12500, 25000, 50000, len(word_freq_dict)]

    index = 0
    for v in vocabulary_size_list:
        f = open(file_list[index], 'w')
        f.write('WordID Frequency' + '\n')
        for i in range(v):
            f.write(str(word_freq_dict[i]) + '\n')
        f.close()
        index += 1


# ================================================================================#
# MAIN Function
def main():
    dir = 'C:/Users/prati/PycharmProjects/MlPractice/Assignment 3/20NG_Data'
    data = load_csv(dir + '/' + 'train_data.csv')
    data = numpy.array(data).tolist()

    word_freq_dict = get_word_freq_dict(data)
    word_freq_dict = sorted(word_freq_dict.items(), key=operator.itemgetter(1), reverse=True)

    # Write word-freq list into files
    write_wf_into_files(word_freq_dict)


# ================================================================================#
if __name__ == '__main__':
    main()
