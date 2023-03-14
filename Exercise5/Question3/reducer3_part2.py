#!/usr/bin/env python
import sys

user_rating_dict={}
# user_ratings_list = []
# current_user = None
user_list = []
average = []


for line in sys.stdin:
    userid, ratings = line.split('\t')
    ratings = ratings.strip()
    if userid in user_rating_dict:
        user_rating_dict[userid].append(int(float(ratings)))
    else:
        user_rating_dict[userid]=[]
        user_rating_dict[userid].append(int(float(ratings)))
# print(user_rating_dict)

for user in user_rating_dict:
    length_userid = len(user_rating_dict[user])
    if length_userid>40:
        user_list.append(user)
        avg = sum(user_rating_dict[user])/len(user_rating_dict[user])
        average.append(avg)

sort_function = zip(user_list, average)
sorted_values = sorted(sort_function, key=lambda val: val[1], reverse=False)[:1]

print("User Id",'\t\t',"Avg Ratings",'\n')
for key in sorted_values:
    print(*key, sep='\t\t')
    print('\n')
