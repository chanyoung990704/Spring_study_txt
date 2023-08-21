# 영화 추천 시스템 - 협업 시스템
import surprise
import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

ratings = pd.read_csv('ml-latest-small/ratings.csv')
min = ratings['rating'].min()
max = ratings['rating'].max()

reader = Reader(rating_scale=(min, max))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader=reader)

svd = SVD(random_state=0)
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# 교차 검증(K-Fold 교차검증)
# 100개 데이터
# cv:5 => 100개 데이터를 5개로 분할
# A:1 ~ 20, B:21 ~ 40, C:41 ~ 60, D:61 ~ 80, E: 81 ~ 100
# ABCD(train set) E (test set)
# ABCE(train set) D (test set)
# ABDE(train set) C (test set)
# ACDE(train set) B (test set)
# ...

trainset = data.build_full_trainset()
svd.fit(trainset)

ratings_user_1 = ratings[ratings['userId'] == 1]
# userId가 1번인 사람이 movieId가 302인 영화에 대해서 모델 예측 평가 점수
predict = svd.predict(1, 302)
# userId가 1번인 사람이 movieId가 316인 영화에 대해서 실제 평가 점수가 3점일 때 모델 예측 평가 점수
test_predict = svd.predict(1, 316, 3)
