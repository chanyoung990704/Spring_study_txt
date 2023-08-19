# 영화 추천 시스템
# 1. Demographic Filtering (인구통계학적 필터링)
# 2. Content Based Filtering (컨텐츠 기반 필터링)
# 3. Collaborative Filtering (협업 필터링)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)


# 1. Demographic Filtering
csv1 = pd.read_csv("tmdb_5000_credits.csv")
csv2 = pd.read_csv("tmdb_5000_movies.csv")

csv1.columns = ['id', 'title', 'cast', 'crew']

csv2 = csv2.merge(csv1[['id', 'cast', 'crew']], on='id')

C = csv2['vote_average'].mean()
print(C)

m = csv2['vote_count'].quantile(0.9)
print(m)

q_movies = csv2.copy().loc[csv2['vote_count'] >= m]
print(q_movies.shape)

print(q_movies['vote_count'].sort_values())

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

q_movies = q_movies.sort_values('score', ascending=False)

print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10))

pop= csv2.sort_values('popularity', ascending=False)
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(10),pop['popularity'].head(10), align='center',
         color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")
plt.show()