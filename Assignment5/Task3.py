import numpy
import pandas as pd
import math
import copy
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score, precision_score, recall_score

# =============================================================================#
# Load a CSV file
def load_csv(filename):
    data = pd.read_csv(filename, header=None)
    dataset = data.values
    return dataset


# =============================================================================#
def initialize_centroids(k, dataset):
    centroids = []
    for i in range(k):
        centroids.append(dataset[i])
    return centroids


# =============================================================================#
def calculate_znk(data, centroids, k, n):
    znk = numpy.array([[0.0 for i in range(k)] for j in range(n)])
    idx = 0
    for row in data:
        list_of_distances = []
        for cen in centroids:
            dist = numpy.linalg.norm(row - cen)
            list_of_distances.append(dist)
        index = list_of_distances.index(min(list_of_distances))
        znk[idx][index] = 1.0
        idx += 1
    return znk


# =============================================================================#
def calculate_centroids(k, znk, data, cen):
    centroids = []
    SSE = []
    outer_idx = 0
    for i in range(k):
        temp = []
        idx = 0
        sum_of_squared_errors = 0.0
        for row in znk:
            if row[i] == 1.0:
                temp.append(data[idx])
                sum_of_squared_errors += abs(numpy.linalg.norm(cen[outer_idx] - data[idx])) ** 2
            idx += 1
        temp = numpy.array(temp)
        centroids.append(temp.mean(axis=0))
        SSE.append(sum_of_squared_errors)
        outer_idx += 1
    return centroids, SSE


# =============================================================================#
def k_means_algorithm(dataset, k, max_iterations):
    centroids = initialize_centroids(k, dataset)
    n = len(dataset)
    data = []
    cen = []

    for row in dataset:
        data.append(row[:-1])
    data = numpy.array(data)
    for row in centroids:
        cen.append(row[:-1])
    cen = numpy.array(cen)
    centroids = cen

    prev_SSE = math.inf

    for i in range(max_iterations):
        prev_cen = centroids
        znk = calculate_znk(data, centroids, k, n)
        centroids, SSE = calculate_centroids(k, znk, data, centroids)
        total_SSE = sum(SSE)

        if prev_SSE - total_SSE <= 0.01:
            return prev_SSE, znk, centroids

        prev_SSE = total_SSE


# =============================================================================#
# def calculate_NMI():


# =============================================================================#
# MAIN FUNCTION
def main():
    # dataset_list = ['dermatologyData.csv', 'ecoliData.csv', 'glassData.csv', 'soybeanData.csv', 'vowelsData.csv', 'yeastData.csv']
    dataset_list = ['yeastData.csv']
    true_labels = []
    nmi_score = []
    pred_labels = []
    for ds in dataset_list:
        print(ds[:-4])
        # Careful, the dataset also consists of labels
        dataset = load_csv(ds)
        max_iterations = 100
        sum_sq_error = []

        copy_of_dataset = copy.deepcopy(dataset)
        copy_of_dataset = numpy.array(copy_of_dataset)

        # K = numpy.arange(2, 21, 1)
        K = numpy.array([3])
        for k in K:
            SSE, znk, centroids = k_means_algorithm(dataset, k, max_iterations)
            sum_sq_error.append(SSE)
            print("SSE for k= ", end='')
            print(k, end='')
            print(' is ', end='')
            print(SSE)

            for row in copy_of_dataset:
                true_labels.append(row[-1])

            znk = znk.tolist()
            for row in znk:
                pred_labels.append(row.index(max(row)) + 1)

            nmi_score.append(normalized_mutual_info_score(true_labels, pred_labels))
            print("K=", end='')
            print(k, end='')
            print(" NMI score = ", normalized_mutual_info_score(true_labels, pred_labels))
            print(accuracy_score(true_labels,pred_labels))

            # calculate_NMI()
        print("-" * 50)
        # plt.plot(K, sum_sq_error)
        # plt.title("SSE against K: Yeast Dataset")
        # plt.xlabel("K")
        # plt.ylabel("SSE")
        # plt.show()


# =============================================================================#

if __name__ == "__main__":
    main()
