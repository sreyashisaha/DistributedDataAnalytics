
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
import functools
import operator
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

rgb = cv2.imread('image1.jpg',cv2.IMREAD_COLOR)
# print(rgb.shape)

blue_image = rgb.copy() # Make a copy
blue_image[:,:,1] = 0
blue_image[:,:,2] = 0
blue_image_array = np.array(blue_image[1])
# print(green_image_array)

green_image = rgb.copy() # Make a copy
green_image[:,:,0] = 0
green_image[:,:,2] = 0
green_image_array = np.array(green_image[2])
# print(green_image_array)

red_image = rgb.copy() # Make a copy
red_image[:,:,0] = 0
red_image[:,:,1] = 0
red_image_array = np.array(red_image[3])
# print(red_image_array)

start=MPI.Wtime()
if rank == 0:
    red_image_split = np.array_split(red_image_array,size)
    blue_image_split = np.array_split(blue_image_array,size)
    green_image_split  = np.array_split(green_image_array,size)
else:
    red_image_split=None
    blue_image_split=None
    green_image_split=None

# frequency=[]
#scatter the splitted array elements the elements in the order of process rank

sent_data_b = comm.scatter(blue_image_split,root=0)
sent_data_g = comm.scatter(green_image_split,root=0)
sent_data_r = comm.scatter(red_image_split,root=0)

# print("Process {} of {} has result after scatter {}".format(rank,size,sent_data_b))
# print("Process {} of {} has result after scatter {}".format(rank,size,sent_data_g))
# print("Process {} of {} has result after scatter {}".format(rank,size,sent_data_r))

elements_b=[]
for list in sent_data_b:
    for element in list:
        elements_b.append(element)
# print(elements_b)

elements_g=[]
for list in sent_data_g:
    for element in list:
        elements_g.append(element)

elements_r=[]
for list in sent_data_r:
    for element in list:
        elements_r.append(element)
# print(elements)

freq_b = {}
for item in elements_b: 
    if (item in freq_b): 
        freq_b[item] += 1
    else: 
        freq_b[item] = 1
# print(freq_b)

freq_g = {}
for item in elements_g: 
    if (item in freq_g): 
        freq_g[item] += 1
    else: 
        freq_g[item] = 1
# print(freq_g)

freq_r = {}
for item in elements_r: 
    if (item in freq_r): 
        freq_r[item] += 1
    else: 
        freq_r[item] = 1
# print(freq_r)

# # # Gathers the elements from all cores
data_b = comm.gather(freq_b, root=0)
data_r = comm.gather(freq_r, root=0)
data_g = comm.gather(freq_g, root=0)


if rank == 0:
    # sum the values with same keys
    res_b = dict(functools.reduce(operator.add,map(collections.Counter, data_b)))
    res_g = dict(functools.reduce(operator.add,map(collections.Counter, data_g)))
    res_r = dict(functools.reduce(operator.add,map(collections.Counter, data_r)))

    # print("New dict : ", res)

    plt.bar(res_b.keys(), res_b.values())
    plt.bar(res_g.keys(), res_g.values())
    plt.bar(res_r.keys(), res_r.values())
    plt.xlabel("Pixels")
    plt.ylabel("Frequency")
    plt.title("Histogram of a RGB Image")
    plt.show()

# # end = MPI.Wtime()
# # total_time = end - start
# # print(f" Rank: {rank} Time taken to calculate the histogram: {total_time}")






