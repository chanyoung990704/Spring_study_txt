# 영화 추천 시스템
# 1. Demographic Filtering (인구통계학적 필터링)
# 2. Content Based Filtering (컨텐츠 기반 필터링)
# 3. Collaborative Filtering (협업 필터링)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)


def get_recommendation(title, cosine_sim):
    idx = indices[title]
    # 코사인 유사도 매트릭스에서 idx에 해당하는 데이터를 (idx, 유사도) 형태로 얻기
    similarity_idx_list = list(enumerate(cosine_sim[idx]))
    # 코사인 유사도 내림차순 정렬
    similarity_idx_list = sorted(similarity_idx_list, key=lambda x: x[1], reverse=True)
    # 자기 자신을 제외한 10개의 추천 영화 슬라이싱
    similarity_idx_list = similarity_idx_list[1: 11]
    movie_indices = [i[0] for i in similarity_idx_list]
    # 인덱스 정보를 통해 제목 추출
    return csv2['title'].iloc[movie_indices].values


# 감독 정보 추출
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# 처음 3개의 데이터 중에서 name에 해당하는 value만 추출
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(' ', '')) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(' ', ''))
        else:
            return ''


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


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

pop = csv2.sort_values('popularity', ascending=False)
plt.figure(figsize=(12, 4))

plt.barh(pop['title'].head(10), pop['popularity'].head(10), align='center',
         color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")
plt.show()

# 2. Content Based Filtering (컨텐츠 기반 필터링)
# 줄거리 기반 필터링
print(csv2['overview'].head(5))

# Bag Of Words - BOW
# => 단어들이 각각 몇 번씩 나왔는가

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
print(csv2['overview'].isnull().values.any())  # null 값이 있는지 확인

csv2['overview'] = csv2['overview'].fillna('')  # null 값을 공백으로 채움
tfidf_matrix = tfidf_vectorizer.fit_transform(csv2['overview'])  # tfidf방식 이용해 텍스트 유사도 측정
print(tfidf_matrix.shape)

# 코사인 유사도
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)  # 값이 1이여야 유사함.
print(cosine_sim)

indices = pd.Series(csv2.index, index=csv2['title']).drop_duplicates()  # 1차원 배열생성
print(indices)

# 영화의 제목을 입력받으면 코사인 유사도를 통해 가장 유사도가 높은 상위 10개 영화 목록 반환
recommendation_movie = get_recommendation('Avatar', cosine_sim)
print(recommendation_movie)

# 다양한 요소 기반 추천 (장르, 감독, 키워드)
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    csv2[feature] = csv2[feature].apply(literal_eval)

csv2['director'] = csv2['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    csv2[feature] = csv2[feature].apply(get_list)

features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    csv2[feature] = csv2[feature].apply(clean_data)

csv2['soup'] = csv2.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(csv2['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

csv2 = csv2.reset_index()
indices = pd.Series(csv2.index, index=csv2['title'])

print(get_recommendation('The Dark Knight Rises', cosine_sim2))

movies = csv2[['id', 'title']].copy()
pickle.dump(movies, open('movies.pickle', 'wb'))
pickle.dump(cosine_sim2, open('cosine_sim.pickle', 'wb'))
