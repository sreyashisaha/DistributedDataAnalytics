#!/usr/bin/env python

import sys

rating_dict = {}
for line in sys.stdin:
    # print(line)
    movie, ratings = line.split('\t')
    # print(movie,ratings)
    ratings = ratings.strip()
    if movie in rating_dict:
        rating_dict[movie].append(int(float(ratings)))
    else:
        rating_dict[movie]=[]
        rating_dict[movie].append(int(float(ratings)))

for key,value in rating_dict.items():
    rating_dict[key]=sum(value)/len(value)

sorted_values = sorted(rating_dict.items(), key=lambda x: x[1], reverse=True)[:5]

print("Movie Id",'\t\t\t\t',"Avg Ratings",'\n')
for key in sorted_values:
    print(*key, sep='\t\t')
    print('\n')
#     # print(ratings)
#     # if current_movie is not None:
#     #     if current_movie == movie:
#     #         ratings_list.append(int(float(ratings)))
#     #         # print(ratings_list)
#     #     else:
#     #         # print(ratings_list)
#     #         # if ratings_list:
#     #         print(current_movie, (sum(ratings_list)/len(ratings_list)))
#     #     current_movie=movie
#     #     ratings_list = []
#     # else:
#     #     current_movie = movie
#     #     ratings_list.append(int(float(ratings)))
#     if movie in rating_dict:
#         rating_dict[movie].append(int(float(ratings)))
#         # if current_movie == movie:
#         #     ratings_list.append(int(float(ratings)))
#     else:
#             # if ratings_list:
#                 # movie_list.append(current_movie)
#                 # average.append((sum(ratings_list) / len(ratings_list)))
#         rating_dict[movie]=[]
#         rating_dict[movie].append(int(float(ratings)))
#         # current_movie = movie
#         # ratings_list = []
#     # else:
#     #     current_movie = movie
#     #     ratings_list.append(int(float(ratings)))

# for key,value in rating_dict.items():
#     rating_dict[key]=sum(value)/len(value)
# # print(rating_dict)
# # rating_dict[current_movie] = (sum(ratings_list) / len(ratings_list))

# print(sorted(rating_dict.items(), key=lambda x: x[1], reverse=True)[:5])


