import pickle
import streamlit as st
from tmdbv3api import Movie, TMDb

movie = Movie()
tm_db = TMDb()
tm_db.api_key = '14a77c40ff3440343792dec21646d0de'
tm_db.language = 'ko-KR'

movies = pickle.load(open('./project/movies.pickle', 'rb'))
cosine_sim = pickle.load(open('./project/cosine_sim.pickle', 'rb'))

st.set_page_config(layout='wide')
st.header('MovieRecommend')

movie_list = movies['title'].values
title = st.selectbox('Choose a movie you like', movie_list)


def get_recommendations(title):
    # 영화 제목을 통해 전체 데이터 기준 영화 idx 값 얻기
    idx = movies[movies['title'] == title].index[0]
    # 코사인 유사도 매트릭스에서 idx에 해당하는 데이터를 (idx, 유사도) 형태로 변환
    sim_scores = list(enumerate(cosine_sim[idx]))
    # 코사인 유사도 내림차순 정렬
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # 코사인 유사도 슬라이싱 ( 데이터 10개 추출 )
    sim_scores = sim_scores[1:11]
    # 추천 영화 목록 10개의 idx 정보 추출
    movie_indices = [i[0] for i in sim_scores]
    # idx 정보를 통해 영화 제목 추출
    images = []
    titles = []
    for i in movie_indices:
        id = movies['id'].iloc[i]
        details = movie.details(id)
        image_path = details['poster_path']
        if image_path:
            image_path = 'https://image.tmdb.org/t/p/w500' + image_path
        else:
            image_path = 'no_image.jpg'

        images.append(image_path)
        titles.append(details['title'])

    return images, titles


if st.button('Recommend'):
    with st.spinner('Loading...'):
        imgs, titles = get_recommendations(title)
        idx = 0
        for i in range(0, 2):
            cols = st.columns(5)
            for col in cols:
                col.image(imgs[idx])
                col.write(titles[idx])
                idx += 1
