import random
import numpy as np

def regression():
    # TODO:: regression
    regression_data = data_reader("data/regression.txt")
    normalized_data = normalize_data(regression_data)
    training_data, test_data = randomly_sampled_data(normalized_data)
    print("Training data size: " + str(training_data.shape) + " Testing data size: " + str(test_data.shape))
    K(training_data, 1,1,1)



def classification(data):
    #TODO:: Classification
    pass

def data_reader(file_path):
    return np.loadtxt(file_path, skiprows=1)

def randomly_sampled_data(data, percentage = 0.7):
    how_many_data_points_to_sample = int(np.around(data.shape[0] * percentage))
    idxs = np.random.choice(data.shape[0], size=how_many_data_points_to_sample, replace= False)
    mask = np.zeros(data.shape[0], dtype=bool)
    mask[idxs] = True
    training_data = data[mask]
    test_data = data[~mask]
    return training_data, test_data

def normalize_data(data):
    #vector normalization
    normalized_data = data / np.linalg.norm(data, axis = 1).reshape(-1,1)
    return normalized_data

def K(data, teta1, teta2, teta3):
    K = np.zeros((data.shape[0], data.shape[0]))
    for i in np.arange(data.shape[0]):
        for j in np.arange(data.shape[0]):
            distance = np.linalg.norm(data[i,:] - data[j,:])
            if (i == j):
                K[i,j] = teta1 * np.exp(- distance / (2* np.square(teta2)))
            else:
                K[i,j] = teta1 * np.exp(- distance / (2 * np.square(teta2))) + teta3
    print("K calculated, shape " + str(K.shape))
    return K

def decompose_K(K):
    return np.linalg.cholesky(K)



regression()











# import matplotlib.pyplot as plt
#
# def kernel(a, b):
#     sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
#     return np.exp(-0.5 * sqdist)
#
# n = 50
# Xtest = np.linspace(-5, 5, n).reshape(-1, 1)
# K_ = kernel(Xtest, Xtest)
#
# L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
# f_prior=np.dot(L, np.random.normal(size=(n, 10)))
#
# plt.plot(Xtest, f_prior)
# plt.show()