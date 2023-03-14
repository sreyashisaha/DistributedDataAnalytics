#!/usr/bin/env python
import sys

genre_rating_dict={}
genre_average_rating_dict={}
# user_ratings_list = []
# current_user = None
genre_list = []
average = []


for line in sys.stdin:
    genre, ratings = line.split('\t')
    ratings = ratings.strip()
    # print(genre, ratings)
    genre_parts = genre.split('|')
    for part in genre_parts:
        genre = part
        if genre in genre_rating_dict:
            genre_rating_dict[genre].append(int(float(ratings)))
        else:
            genre_rating_dict[genre]=[]
            genre_rating_dict[genre].append(int(float(ratings)))

for genre_type in genre_rating_dict:
    genre_list.append(genre_type)
    avg = sum(genre_rating_dict[genre_type])/len(genre_rating_dict[genre_type])
    average.append(avg)

sort_function = zip(genre_list, average)
sorted_values = sorted(sort_function, key=lambda val: val[1], reverse=True)[:5]
print("Genre",'\t\t',"Avg Ratings",'\n')
for key in sorted_values:
    print(*key, sep='\t\t')
    print('\n')