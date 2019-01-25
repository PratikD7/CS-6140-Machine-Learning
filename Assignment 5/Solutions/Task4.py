import numpy
import copy
import math
from numpy import linalg
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
import Task3 as kmeans_model
from math import log


# =============================================================================#

def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = linalg.det(sigma)
        if det == 0:
            return 0.0
            raise NameError("The covariance matrix can't be singular")
        try:
            norm_const = 1.0 / (math.pow((2 * math.pi), float(size) / 2) * math.pow(det, 1.0 / 2))
        except:
            return 0.0 # Correction
        try:
            x_mu = numpy.matrix(x - mu)
            inv = numpy.linalg.inv(sigma)
            result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        except:
            return 0.0 # Correction
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")


# =============================================================================#
# Initialize the parameters theta
def initialize_the_parameters(centroids, znk, mu_k, pi_k, copy_of_dataset, sigma_k):
    sigk = []

    for row in centroids:
        mu_k.append(row)

    for i in range(len(znk[0])):
        pi_k.append(sum(row[i] for row in znk))

    for col in range(len(znk[0])):
        temp = []
        for row in range(len(znk)):
            if znk[row][col] == 1.0:
                temp.append(copy_of_dataset[row])
        temp = numpy.array(temp)
        sigma_k.append(numpy.cov(temp.T))

    return pi_k, sigma_k, mu_k


# =============================================================================#
def e_step(col, row, pi_k, sigma_k, X, mu_k):
    znk = [[0.0 for j in range(col)] for i in range(row)]
    den = 0.0

    # Loop only for n not for k (Check the parameter dimensions)
    for n in range(row):
        # Calculate den
        for j in range(col):
            den += pi_k[j] * norm_pdf_multivariate(X[n], mu_k[j], sigma_k[j])
        for k in range(col):
            # Calculate num
            num = pi_k[k] * norm_pdf_multivariate(X[n], mu_k[k], sigma_k[k])
            if num==0.0:
                znk[n][k] = math.pow(10.0, -4)  # Correction
            if den == 0.0:
                znk[n][k] = math.pow(10.0, -4)  # Correction

    return znk


# =============================================================================#
def calculate_mu(znk, X, prev_mu_k):
    mu_k = []
    for k in range(len(znk[0])):
        den = 0.0
        num = 0.0
        for n in range(len(znk)):
            den += znk[n][k]
        for n in range(len(znk)):
            num += znk[n][k] * X[n]
        if den == 0.0:
            mu_k.append(prev_mu_k[k])
        else:
            mu_k.append(num / den)
    return numpy.array(mu_k)


# =============================================================================#
def calculate_sigma(znk, X, mu_k, prev_sigma_k):
    sigma_k = []
    for k in range(len(znk[0])):
        den = 0.0
        num = 0.0
        for n in range(len(znk)):
            den += znk[n][k]
        for n in range(len(znk)):
            first_term = (X[n] - mu_k[k])
            first_term = numpy.array(first_term)
            second_term = numpy.array([first_term])
            second_term = second_term.transpose()
            dp = first_term * second_term
            num += znk[n][k] * dp

        if den == 0.0:
            sigma_k.append(prev_sigma_k[k])
        else:
            sigma_k.append(num / den)
    return sigma_k


# =============================================================================#
def calculate_pi(znk, X):
    pi_k = []
    for k in range(len(znk[0])):
        num = 0.0
        for n in range(len(znk)):
            num += znk[n][k]
        pi_k.append(num / len(znk))
    return pi_k


# =============================================================================#
def m_step(znk, X, prev_sigma_k, prev_mu_k):
    mu_k = calculate_mu(znk, X, prev_mu_k)
    sigma_k = calculate_sigma(znk, X, mu_k, prev_sigma_k)
    pi_k = calculate_pi(znk, X)

    return mu_k, sigma_k, pi_k


# =============================================================================#
def calculate_convergence_value(mu_k, prev_mu_k):
    prev_mu_k = numpy.array(prev_mu_k)
    value = 0.0
    for i in range(len(mu_k)):
        value += abs(numpy.linalg.norm(mu_k[i] - prev_mu_k[i]))
    return value


# =============================================================================#
def GMM_algoithm(max_iterations, copy_of_dataset, flag, tolerance, znk, pi_k, sigma_k, mu_k, prev_mu_k, prev_sigma_k):
    for i in range(max_iterations):
        # E-STEP
        znk = e_step(len(znk[0]), len(znk), pi_k, sigma_k, copy_of_dataset, mu_k)

        # M-STEP
        mu_k, sigma_k, pi_k = m_step(znk, copy_of_dataset, prev_sigma_k, prev_mu_k)

        # CONVERGENCE CRITERIA
        value = calculate_convergence_value(mu_k, prev_mu_k)
        # print(value)
        if flag == 1:
            flag += 1
            pass
        elif (value - prev_value) > tolerance:
            return mu_k, sigma_k, pi_k

        prev_value = value


# =============================================================================#
def calculate_SSE(znk, mu_k, X):
    SSE = []
    outer_idx = 0
    znk = znk.tolist()

    for i in range(len(znk[0])):
        sum_squared_error = 0.0
        idx = 0
        temp = []
        for row in znk:
            if row[i] == row.index(max(row)):
                temp.append(X[idx])
                sum_squared_error += abs(numpy.linalg.norm(mu_k[outer_idx] - X[idx])) ** 2

            idx += 1
        SSE.append(sum_squared_error)
        outer_idx += 1
    return sum(SSE)


# =============================================================================#
def main():
    # dataset_list = ['dermatologyData.csv', 'ecoliData.csv', 'glassData.csv', 'soybeanData.csv', 'vowelsData.csv', 'yeastData.csv']
    dataset_list = ['vowelsData.csv']
    max_iterations = 100
    K = numpy.arange(3, 17, 1)
    # K = numpy.array([3])

    tolerance = 0.05

    for ds in dataset_list:
        sum_sq_err = []
        print(ds[:-8])
        copy_of_dataset = []
        dataset = kmeans_model.load_csv(ds)

        # Make a copy of the original dataset
        cp = copy.deepcopy(dataset)
        cp = numpy.array(cp)
        for row in cp:
            copy_of_dataset.append(row[:-1])

        for k in K:
            print(k)
            flag = 1
            mu_k = []
            pi_k = []
            sigma_k = []
            # Initialize the parameters theta
            SSE, znk, centroids = kmeans_model.k_means_algorithm(dataset, k, max_iterations)
            pi_k, sigma_k, mu_k = initialize_the_parameters(centroids, znk, mu_k, pi_k, copy_of_dataset, sigma_k)
            prev_mu_k = mu_k
            prev_sigma_k = sigma_k
            try:
                mu_k, sigma_k, pi_k = GMM_algoithm(max_iterations, copy_of_dataset, flag, tolerance, znk, pi_k, sigma_k, mu_k, prev_mu_k, prev_sigma_k)
                SSE = calculate_SSE(znk, mu_k, copy_of_dataset)
                sum_sq_err.append(SSE)
            #     print("SSE for k= ", end='')
            #     print(k, end='')
            #     print(' is ', end='')
            #     print(sum(SSE))

                true_labels = []
                pred_labels = []
                nmi_score = []
                for row in cp:
                    true_labels.append(row[-1])
                znk = znk.tolist()
                for row in znk:
                    pred_labels.append(row.index(max(row)) + 1)
                nmi_score.append(normalized_mutual_info_score(true_labels, pred_labels))
                # print("K= ", end='')
                # print(k, end='')
                # print(" NMI score = ", normalized_mutual_info_score(true_labels, pred_labels))
            except TypeError:
                SSE = calculate_SSE(znk, mu_k, copy_of_dataset)
                sum_sq_err.append(SSE)
                # print("SSE for k= ", end='')
                # print(k, end='')
                # print(' is ', end='')
                # print(sum(SSE))

                true_labels = []
                pred_labels = []
                nmi_score = []
                for row in cp:
                    true_labels.append(row[-1])
                znk = znk.tolist()
                for row in znk:
                    pred_labels.append(row.index(max(row)) + 1)
                nmi_score.append(normalized_mutual_info_score(true_labels, pred_labels))
                # print("K= ", end='')
                # print(k, end='')
                # print(" NMI score = ", normalized_mutual_info_score(true_labels, pred_labels))
        plt.plot(K, sum_sq_err)
        plt.title("GMM-SSE against K: Vowels Dataset")
        plt.xlabel("K")
        plt.ylabel("SSE")
        plt.show()


# =============================================================================#
if __name__ == '__main__':
    main()
