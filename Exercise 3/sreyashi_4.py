## Sreyashi Saha
from cProfile import label
from pickle import TRUE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math 
from math import sqrt,floor
import time
import random
np.random.seed(0)

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0
k = 3 # No of clusters


# Random Method Of Centroid Initialisation
def centroid_initialisation1(data, number_of_clusters):
    # Randomly choosing 3 rows from the dataset as our centroids
    first_centroids = np.random.randint(data.shape[0], size=number_of_clusters)
    centroid_list = [] # Initialise an empty list to store the 3 random centroids
    # print('centroid rows are ', first_centroids)
    for i in first_centroids:
        centroid_list.append(np.asanyarray(data[i], dtype=float))
    return centroid_list


# define a function to find the average of the Total Sum found from Naive Sharding method
def average(sums, split_size):
    value = sums / split_size
    return value


# Naive Sharding Method Of Centroid Initialization
def centroid_initialization(ds,k_clusters):
    n = np.shape(ds)[1]
    # print(n) # print the number of attributes in the dataset
    m = np.shape(ds)[0]
    # print(m) # print the number of rows in the dataset
    centroids = np.mat(np.zeros((k_clusters,n))) 
     # Sum all elements of each row, add as col to original dataset, sort
    sum_column = np.mat(np.sum(ds, axis=1)) # Calculate the sum of each attributes and storing in a 1-d array 
    Transposed_sum_column=sum_column.T
    ds = np.append(sum_column.T, ds, axis=1) # storing the sums in a separate column 
    # print(ds)
    ds.sort(axis=0) # sorting the summed up values in ascending order
    # Step value for dataset sharding
    step = floor(m/k)
    # print(step)
    vectorize_func = np.vectorize(average)
    # Vectorize mean ufunc for numpy array
    # finding the average of some of the data points
    for i in range(k):
        if i == k-1:
            centroids[i:] = vectorize_func(np.sum(ds[i*step:,1:], axis=0), step)
        else:
            centroids[i:] = vectorize_func(np.sum(ds[i*step:(i+1)*step,1:], axis=0), step)
    return centroids #returning initially the 3 centroids since we have considered k(=3) clusters


# Calculating the Euclidean Distance
def distance(row_data,centroid):
    euclidian_distance = np.linalg.norm(row_data-centroid) #finding out the euclidean distance between the data ppoints and the centroids
    return euclidian_distance


# Find the minimum distance obtained by calculating Euclidean distance for each data point w.r.t. the 3 centroids
# Return the index of the minimum distance found 
def minimum_index(distance_array):
    # minimum_distance = np.min(distance_array)
    #finding the minimum distance for each instance that is calculated by computing the euclidean distance with respect to all the centroids
    Minimum_position = np.argmin(distance_array)# stores the position which has the minimum distance 
    return Minimum_position+1 # to get which cluster that minimum distance belongs to


# returning a list which will give me the rows obtaianed for each of the clusters
def filtered_clusters(dictA, clusters):
    resultant = []
    for i in range(clusters):
        resultant_list = [[keys for keys, values in dictA.items() if values == i + 1]]
        resultant.append(resultant_list)
    return resultant

def assigning_clusters(data_chunks,centroids):
    array_distances = np.array([])
    eu_dist = []
    my_dict={}
    for i,row in enumerate(data_chunks):
        dist=[]
        for j in range(len(centroids)):
            # appending the euclidean distance obtained by subracting each centroids from each of the columns of each rows in the dataset
            dist.append(distance(row,centroids[j]))
            # print("The distances obtained after computing euclidean dist",d)
            eu_dist.append(dist) #appending 3 sets of distances obtained wrt 3 centroids to a list
            # Finding out the minimum distance among 3 distances obtained for 1 row
            minimum_position = minimum_index(dist)
            my_dict[i] = minimum_position # labeling the ith row according to the clusters 
        array_distances = np.append(array_distances,eu_dist)
    # cluster_list = filtering_cluster(my_dict,len(centroids))
    return my_dict


# Finding sum according to the clusters
# for eg if there are 20 rows in cluster 1 then returning the sum of all those rows  
def sum_rows(dataset, centroids):
    labelled_dictionary= assigning_clusters(dataset, centroids)
    # print(labelled_dictionary)
    resultant_list = filtered_clusters(labelled_dictionary, len(centroids))
    # print(resultant_list)
    sum_to_update_centroids = update_centroids(resultant_list, dataset)
    # print(sum_to_update_centroids)
    return sum_to_update_centroids


# Update centroids in every iteration untilmstopping condition is met
def update_centroids(list_keys, dataset):
    updated_center = []
    for i, row in enumerate(list_keys):
        for grp in row:
            # print('Cluster No: i:', grp)
            # convert array into a df to make extraction of rows easier
            convert = pd.DataFrame(dataset)
            # print('df:', converted_df )
            df = convert.loc[convert.index[grp]]
            count = len(df)
            # print('df:', df)
            # Sum of all rows attribute wise
            sum = df.sum(axis=0)
            sum = sum.to_numpy()#convert back to numpy array
        # print(f'Sum of columns for cluster[{i}] is: {sum} \n')
        updated_center.append([count, sum])
    # print("Updated centroid:", updated_center)
    return updated_center

iter=0
start = MPI.Wtime()
Initial_centroids=[]
if rank == root:
    data = pd.read_csv('Absenteeism_at_work.csv')
    # data_new = data.iloc[0:12,1:3]
    # k=3
    dataset= data.to_numpy()  
    split_array = np.array_split(dataset,size)
    Initial_centroids = centroid_initialization(dataset,k)
    print("My initial centroids are",Initial_centroids,'\n')
    # step = floor(np.shape(dataset)[0]/k)
else:
    split_array = None
    Initial_centroids=None
    k=None
    # step=None

# Scatter the chunks of data obtained after splitting into each of the processor
scatter = comm.scatter(split_array, root=0)
# print('rank {} and scatter data is {}'.format(rank, scatter))
# print(f'Sacttered Datasets for rank: {rank} of shape: {scatter.shape}')

# Broadcast the initial centroids to all the processors
centroids_bcast = comm.bcast(Initial_centroids, root=0)
# print('rank {} and broadcast data is {}'.format(rank, centroids_bcast))
# print(f'Broadcasted centroids for rank: {rank} of shape: {centroids_bcast.shape}')

#Assigning a flag variable
flag=True
previous_global_centroid = None
while flag:
    
    previous_global_centroid = centroids_bcast
    # dictionary_cluster_assignment = assigning_clusters(scatter, centroids_bcast)
    # print('Each Row Assignment', dictionary_cluster_assignment, '\n')
    sum_array = sum_rows(scatter, centroids_bcast)
    print('Sum_array are  ', (sum_array), '\n')
    
    
    local_sums = []
    count = []
    for i in range(len(sum_array)):
        # print(len(sum_array))
        count.append(sum_array[i][0])
        local_sums.append(sum_array[i][1])
  
    print('counts are', count)
    print('localsums are', local_sums)

    count = np.array(count)
    local_sums = np.array(local_sums)#convert into array for simpler execution

    # summing up the counts obtained from each process for all clusters
    total_counts = comm.reduce(count, root=root)
    # print('The total number counts are ', total_counts,'\n')
    
    # summing up the sums obtained from each process for all clusters 
    final_sum = comm.reduce(local_sums, root = root)
    # print('Final sum', final_sum,'\n')
  

    global_average = []
    if total_counts is not None:
        if final_sum is not None:
            for i in range(len(total_counts)):
                global_average.append(final_sum[i]/total_counts[i])
        print('global avg is', global_average,'\n')

    #  Redistribute the new centroids to each cluster until it converges
    centroids_bcast = comm.bcast(global_average, root=root)
    # cent = np.array([centroids_bcast])
    # print(cent.shape)
    iter+=1
    if np.array_equal(previous_global_centroid,centroids_bcast):
        print('The global centroid is same....Exit the loop')
        flag = False
    else:
        continue
end = MPI.Wtime()
time_taken = end-start
times = comm.gather(time_taken,root=0)
seq_time = 5.365523
new_array =[]
if rank==0:
    time = np.vstack(times)
    total_time = np.sum(time)
    print("total time for processes is ", total_time)
    print("Global Centroids",centroids_bcast,"and no of iterations",iter)
    array_time_p = np.array(time)
    print(array_time_p.T)
    list_p = (array_time_p.T).tolist()
    print(list_p)

    newList = []
    for x in array_time_p:
        newList.append(float(seq_time/x))
    print(newList)

    p = np.arange(size)
    plt.plot(p, newList)
    plt.xlabel("No Of Processors")
    plt.ylabel("Speedup")
    plt.title("Speed Up Graph for 3 clusters")
    plt.show()



