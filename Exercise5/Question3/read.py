import pandas as pd
# pandas.set_option('display.max_rows', None)
# df = pandas.read_csv('data1.csv')
# # print(df)
# df = df.dropna()
# # print(df.isna().sum)
# df.to_csv('data.csv',index=False)

movies_columns = ["movieId", "title", "genre"]
ratings_columns = ["userId", "movieId", "rating", "timestamp"]
df_movies = pd.read_csv('movies.dat', sep='::', names=movies_columns)
df_ratings = pd.read_csv('ratings.dat', sep='::', names=ratings_columns)
# pd.set_option('display.max_columns', None)
#
df_merged = df_ratings.merge(df_movies, on="movieId", how='left').drop(['timestamp', 'movieId'], axis=1)
df_merged.sort_values('userId')
df_merged.to_csv('data1.csv', index=False)