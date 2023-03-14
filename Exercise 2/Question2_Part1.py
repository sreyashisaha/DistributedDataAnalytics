
import math
from typing import Counter
from mpi4py import MPI
import numpy as np
import cv2
from itertools import groupby
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import collections
import itertools


# Save image in the set directory
# Read RGB image
# img = cv2.imread('pic1.jpeg')
# Output img with window name as 'image'
# Output Images
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(img)


# # tuple to select colors of each channel line
# colors = ("red", "green", "blue")
# channel_ids = (0, 1, 2)

# # create the histogram plot, with three lines, one for
# # each color
# plt.figure()
# plt.xlim([0, 256])
# for channel_id, c in zip(channel_ids, colors):
#     histogram, bin_edges = np.histogram(img[:, channel_id], bins=256, range=(0, 255))
#     plt.plot(bin_edges[0:-1], histogram, color=c)

# plt.title("Color Histogram")
# plt.xlabel("Color value")
# plt.ylabel("Pixel count")

# plt.show()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0
split_array=np.array([])
start=MPI.Wtime()

if rank == 0:
    img = cv2.imread('images.jpeg', cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    split_array = np.array_split(img, size)
else:
    split_array = None

# frequency=[]
#scatter the splitted array elements the elements in the order of process rank
# sent_data = comm.scatter(split_array, root=0)

sent_data = comm.scatter(split_array,root=0)
# print("Process {} of {} has result after scatter {}".format(rank,size,sent_data))
# sent_data = [l.tolist() for l in sent_data]
# # print(sent_array)

# elements = [item for sublist in sent_data for item in sublist]
elements=[]
for list in sent_data:
    for element in list:
        elements.append(element)
# print(elements)
# print(type(elements))
    
freq = {} 
for item in elements: 
    if (item in freq): 
        freq[item] += 1
    else: 
        freq[item] = 1

# freq = Counter(elements)
# print(freq)

# hist, bin_edges = np.histogram(sent_data,freq)
# plt(hist)

# # Gathers the elements from all cores
data = comm.gather(freq, root=0)
end = MPI.Wtime()

# print(data)
# # data2 = comm.gather(bin_edges, root=0)
# # print(data2)
# # result = comm.gather(sent_data, root=0)

# # # only the conductor node has all of the small lists
# # if rank == 0:
# #     print("Process {} of {} has result after gather {}".format(rank, size, result))

import collections
import functools
import operator


if rank == 0:
    # sum the values with same keys
    res = dict(functools.reduce(operator.add,map(collections.Counter, data)))

    # print("New dict : ", res)
    plt.bar(res.keys(), res.values())
    plt.xlabel("Pixels")
    plt.ylabel("Frequency")
    plt.title("Histogram of a Grayscale Image")
    plt.show()


total_time = end - start
if rank==0:
    print(f" Rank: {rank} Time taken to calculate the histogram: {total_time}")




