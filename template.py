import random
import numpy as np

def data_reader(file_path):
    return np.loadtxt(file_path, skiprows=1)

############################################################################################
#  Regression - Cholesky decomposition and data normalization
############################################################################################
'''def normalize_data(data):
    #vector normalization
    normalized_data = data / np.linalg.norm(data, axis = 1).reshape(-1,1)
    return normalized_data'''
###Could be mean/std norm also
def normalize_data(data: np.ndarray) -> np.ndarray:
    #vector normalization
    return ((data-np.mean(data))/np.std(data))

def K(data1, data2, teta1, teta2, teta3):
    a: float = 0.5
    K = np.zeros((data1.shape[0], data2.shape[0]))
    for i in np.arange(data1.shape[0]):
        for j in np.arange(data2.shape[0]):
            distance = np.linalg.norm(data1[i,:] - data2[j,:])
            if (i == j):
                K[i,j] = teta1 * np.exp(- distance / (2* np.square(teta2))) + teta3*1
            else:
                K[i,j] = teta1 * np.exp(- distance / (2 * np.square(teta2)))
    #print("K calculated, shape " + str(K.shape))
    return K

def inverse_K(K):
    L = np.linalg.cholesky(K)
    Kinv = np.linalg.inv(L.T) @ np.linalg.inv(L)
    return Kinv

############################################################################################
# Marginal log likelihood - randomly sampled data 70% of the data for training and eq (4) evaluation
############################################################################################

def randomly_sampled_data(data, y, percentage = 0.7):
    how_many_data_points_to_sample = int(np.around(data.shape[0] * percentage))
    idxs = np.random.choice(data.shape[0], size=how_many_data_points_to_sample, replace= False)
    mask = np.zeros(data.shape[0], dtype=bool)
    mask[idxs] = True
    x_train, y_train = data[mask], y[mask]
    x_test, y_test = data[~mask], y[~mask]
    return x_train, y_train, x_test, y_test

def calculate_marginal_log_likelihood(K,y):
    inv_Ka = inverse_K(K)
    logp = -0.5 * y.T @ inv_Ka @ y - 0.5*np.log(np.linalg.det(K)) - (K.shape[0]/2.0)*np.log(2.0*np.pi)
    return logp

############################################################################################
# Gradient Computation
############################################################################################

def Ka_gradient_teta(X, y, Ka, teta1, teta2, teta3):
    inv_Ka = inverse_K(Ka)
    alpha = inv_Ka @ y
    d_k_theta1 = np.zeros((X.shape[0], X.shape[0]))
    d_k_theta2 = np.zeros((X.shape[0], X.shape[0]))
    d_k_theta3 = np.zeros((X.shape[0], X.shape[0]))
    for i in np.arange(X.shape[0]):
        for j in np.arange(X.shape[0]):
            distance= np.linalg.norm(X[i, :] - X[j, :])
            if (i == j):
                d_k_theta1[i,j] = np.exp(- distance / (2 * teta2 ** 2))
                d_k_theta2[i,j] = (np.exp(- distance / (2 * teta2 ** 2)) * teta1 * distance) / teta2 ** 3
                d_k_theta3[i,j] = 1
            else:
                d_k_theta1[i,j] = np.exp(- distance / (2 * teta2 ** 2))
                d_k_theta2[i,j] = (np.exp(- distance / (2 * teta2 ** 2)) * teta1 * distance) / teta2 ** 3
                d_k_theta3[i,j] = 0



    ### trace = sum of diagonal should give the same results?
    #minimize_t1 = 0.5 * np.sum(np.diag((alpha @ alpha.T - inv_Ka) * d_k_theta1))
    #minimize_t2 = 0.5 * np.sum(np.diag((alpha @ alpha.T - inv_Ka) * d_k_theta2))
    #minimize_t3 = 0.5 * np.sum(np.diag((alpha @ alpha.T - inv_Ka) * d_k_theta3))

    minimize_t1 = 0.5 * np.trace((alpha @ alpha.T - inv_Ka) @ d_k_theta1)
    minimize_t2 = 0.5 * np.trace((alpha @ alpha.T - inv_Ka) @ d_k_theta2)
    minimize_t3 = 0.5 * np.trace((alpha @ alpha.T - inv_Ka) @ d_k_theta3)

    return np.exp(minimize_t1), np.exp(minimize_t2), np.exp(minimize_t3)


def regression():
    # TODO:: regression
    regression_data = data_reader("data/regression.txt")
    
    regression_data = normalize_data(regression_data)
    X, y = regression_data[:,:7], regression_data[:,7]
    x_train, y_train, x_test, y_test = randomly_sampled_data(X, y)
    print("Training data size: " + str(x_train.shape) + " Testing data size: " + str(x_test.shape))
    k_train = K(x_train, x_train, 1,1,1)
    margin_llgh = calculate_marginal_log_likelihood(k_train,y_train)
    t1,t2,t3 = Ka_gradient_teta(x_train,y_train,k_train,1,1,1)
    mue, cov = predict(x_train,y_train,x_test,y_test,t1,t2,t3,inverse_K(k_train))
    ## Todo: MSE ???


def classification(data):
    #TODO:: Classification
    pass









def predict(x_train,y_train,x_test,y_test,t1,t2,t3,inv_Ka):
    k_n = np.empty((x_train.shape[0],x_test.shape[0]))
    mue = np.empty((x_test.shape[0]))
    sig = np.empty((x_test.shape[0]))
    for j,_ in enumerate(x_test):
        for i,sample in enumerate(x_train):
            k_n[i,j] = K(np.expand_dims(x_train[i,:],0),np.expand_dims(x_test[j],0),t1,t2,t3)
            
        mue[j] = (np.expand_dims(k_n[:,j],0) @ inv_Ka) @ y_train
        sig[j] = K(np.expand_dims(x_test[j],0),np.expand_dims(x_test[j],0),t1,t2,t3)- (k_n[:,j] @ inv_Ka @ k_n[:,j].T)
    return mue,sig


regression()