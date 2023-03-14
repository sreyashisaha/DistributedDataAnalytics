import warnings

from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")
from itertools import chain
import pandas as pd
import numpy as np
from numpy import array_split, random
from mpi4py import MPI
import math
from math import sqrt

def train_test_split(data):
    np.random.seed(0)
    split_ratio = np.random.rand(len(data))<0.7
    train_data = data[split_ratio]
    test_data = data[~split_ratio] 
    return train_data, test_data


# calculate column means
def column_means(dataset):
	means = [0 for i in range(len(dataset[0]))]
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		means[i] = sum(col_values) / float(len(dataset))
	return means
 
# calculate column standard deviations
def column_stdevs(dataset, means):
	stdevs = [0 for i in range(len(dataset[0]))]
	for i in range(len(dataset[0])):
		variance = [pow(row[i]-means[i], 2) for row in dataset]
		stdevs[i] = sum(variance)
	stdevs = [sqrt(x/(float(len(dataset)-1))) for x in stdevs]
	return stdevs
 
# standardize dataset
def standardize_dataset(dataset, means, stdevs):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - means[i]) / stdevs[i]
 
# def getdata(data):
#     col_means = column_means(data)
#     # print("the column means",col_means)
#     col_std = column_stdevs(data,col_means)
#     # print("the column stdevs",col_std)
#     standardized_data = standardize_dataset(data,col_means,col_std)
#     # print("Standardized Data is", standardized_data)
#     return standardized_data


def MyCustomSGD(train_data,test_data,y_test,learning_rate,n_iter,k,divideby):
    
    # Initially we will keep our Weights and Intercept as 0 as per the Training Data
    w=np.zeros(shape=(1,train_data.shape[1]-1))
    b=0
    train_data = pd.DataFrame(train_data)
    cur_iter=1
    local_rmse = []
    timeList = []
    while(cur_iter<=n_iter): 
        # We will create a small training data set of size K  
        temp=train_data.sample(k)
        # We create our X and Y from the above temp dataset
        y=np.array(temp.iloc[:,-1])
        x=np.array(temp.iloc[:,0:265])
        # print("temp is",temp)
        # We keep our initial gradients as 0
        w_gradient=np.zeros(shape=(1,train_data.shape[1]-1))
        b_gradient=0
        # temp.to_numpy()

        # print(x)      
        for i in range(k): # Calculating gradients for point in our K sized dataset
            prediction=np.dot(w,x[i])+b
            w_gradient=w_gradient+(-2)*x[i]*(y[i]-(prediction))
            b_gradient=b_gradient+(-2)*(y[i]-(prediction))      
        #Updating the weights(W) and Bias(b) with the above calculated Gradients
        w=w-learning_rate*(w_gradient/k)
        
        b=b-learning_rate*(b_gradient/k)
        # print("Intercept is",b)
        
        # Incrementing the iteration value
        cur_iter=cur_iter+1
        
        #Dividing the learning rate by the specified value
        learning_rate=learning_rate/divideby
        y_pred_epoch = predict(test_data, w, b)
        rmse_ranks = RootMeanSquare_Result(y_pred_epoch,y_test)
        local_rmse.append(rmse_ranks)
        # print(local_mse)
    # print("Coefficients",w)    
    return w,b, local_rmse #Returning the weights and Intercept

def predict(data,w,b):
    y_pred=[]
    # print("New Shape",x_test.shape)
    for i in range(len(data)):
        y=np.asscalar(np.dot(w,data[i])+b)
        y_pred.append(y)
    return np.array(y_pred)

def RootMeanSquare_Result(y_test_actual,y_test_pred):
    MSE = np.square(np.subtract(y_test_pred,y_test_actual)).mean() 
    # print("MSE is",MSE)
    RMSE = math.sqrt(MSE)
    # print("Root Mean Square Error:\n")
    # print(RMSE)
    return RMSE

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
root = 0
start_time = MPI.Wtime()
array_split=[]
if rank == root:
    # Loading Data from CSV File
    file_data = pd.read_csv('virus.csv')
    X = file_data.iloc[:, 0:265].to_numpy()
    Y = file_data.iloc[:, -1].to_numpy()
    # Split Data into train and test
    x_train, x_test = train_test_split(X)
    y_train, y_test = train_test_split(Y)

    print("X Shape: ", X.shape)
    print("Y Shape: ", Y.shape)
    print("X_train Shape: ", x_train.shape)
    print("Y_train Shape: ", y_train.shape)
    # Normalizing Data
    norm_matrix_traindata = np.linalg.norm(x_train)
    normalised_array_train = x_train / norm_matrix_traindata

    norm_matrix_testdata = np.linalg.norm(x_test)
    normalised_array_test = x_test / norm_matrix_testdata

    # Adding the target Column in the training data
    train_data = pd.DataFrame(normalised_array_train)
    train_data['target'] = y_train
    x_test = np.array(normalised_array_test)
    y_test = np.array(y_test)
    # split array in the number of workers opened
    array_split = np.array_split(train_data, size)
else:
    x_test = None
    y_test = None
    split_array_train_data = None
# Scattering chunks of data to all processes
train_data = comm.scatter(array_split, root=root)
print(f'rank:{rank} got this size of training Data:{len(train_data)}')
x_test = comm.bcast(x_test, root=root)
y_test = comm.bcast(y_test, root=root)
# Call the SGD function to get the weights from all the ranks 

w, b, RMSE_gathered= MyCustomSGD(train_data, x_test, y_test,learning_rate=0.01, n_iter=200,k=50,divideby=1)
rmse_each_epoch_Gather = comm.gather(RMSE_gathered, root=root)
# print('msegather:', rmse_each_epoch_Gather,'\n')
w_reduced = comm.reduce(w, root=root)
b_reduced = comm.reduce(b, root=root)
# print('w_reduced:', w_reduced,'\n')
# print('b_reduced:', b_reduced,'\n')
end_time = MPI.Wtime()
Net_time = end_time - start_time
all_times = comm.gather(Net_time, root=0)

if rank == 0:
    times = np.vstack(all_times)
    # print('times in Array:', times)
    time_sum = np.sum(times)
    print('Total Time for processes is Net_time=%.3f' % time_sum)
    # Find the global weights and intercepts
    average_coeffs = w_reduced / size
    average_intercept = b_reduced / size
    # print(f'rank:{rank} weights are:{average_coeffs}\n')
    # print(f'rank:{rank} coefficient is:{average_intercept}\n')
    y_pred = predict(x_test, average_coeffs, average_intercept)
    print('Root Mean Squared Error :', RootMeanSquare_Result(y_test, y_pred))

    for i, row in enumerate(rmse_each_epoch_Gather):
        plt.plot(range(len(row)), row, label='rank ' + str(i))
    plt.xlabel('Number of Iterations')
    plt.ylabel('RMSE')
    plt.title('RMSE Vs Number of Iterations')
    plt.grid(ls='--')
    plt.legend()
    plt.show()





                                                                                    