import random
import numpy as np

def regression(data):
    #TODO:: regression
    print(data.shape)



def classification(data):
    #TODO:: Classification
    pass

def data_reader(file_path):
    return np.loadtxt(file_path, skiprows=1)

def randomly_sampled_data(data, percentage = 0.7):
    how_many_data_points_to_sample = int(np.around(data.shape[0] * percentage))
    idxs = np.random.choice(data.shape[0], size=how_many_data_points_to_sample, replace= False)
    sampled_data = data[idxs]
    return sampled_data

def normalize_data(data):
    #vector normalization
    normalized_data = data / np.linalg.norm(data, axis = 1).reshape(-1,1)
    return normalized_data

regression_data = data_reader("data/regression.txt")
normalized_data = normalize_data(regression_data)
randomly_sampled_data(normalized_data)
#regression(regression_data)







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