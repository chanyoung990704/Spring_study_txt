# 다항 회귀
# 공부 시간에 따른 시험 점수 ( 우등생 )
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


read_csv = pd.read_csv("PolynomialRegressionData.csv")
x = read_csv.iloc[:, :-1].values
y = read_csv.iloc[:, -1].values

# 3-1 단순 선형 회귀 (Simple Linear Regression)

regression = LinearRegression()
regression.fit(x, y)

# 데이터 시각화
plt.scatter(x, y, color='blue') # 산점도
plt.plot(x, regression.predict(x), color = 'green')
plt.title('Score by hours (genius)')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

regression_score = regression.score(x, y)
print(regression_score)

# 3-2 다항 회귀 (Polynomial Regression)
poly_reg = PolynomialFeatures(degree=3) # 2차 방정식
x_poly = poly_reg.fit_transform(x)
print(x_poly[: 5]) # [x] -> [x^0, x^1, x^2] 데이터 변형
print(poly_reg.get_feature_names_out())

linear_regression = LinearRegression()
linear_regression.fit(x_poly, y)  # 변환된 x와 y를 갖고 모델 생성

# 데이터 시각화 ( 변환된 x와 y )
plt.scatter(x, y, color='blue')
plt.plot(x, linear_regression.predict(poly_reg.fit_transform(x)), color='green'  )
plt.title('Score by hours (genius)')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

x_range = np.arange(min(x), max(x), 0.1)
print(x_range)

x_range = x_range.reshape(-1, 1)  # row 개수는 자동 계산, column 개수는 1개

print(x_range)

# 데이터 시각화 ( 변환된 x와 y )
plt.scatter(x, y, color='red')
plt.plot(x_range, linear_regression.predict(poly_reg.fit_transform(x_range)), color='green'  )
plt.title('Score by hours (genius)')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

# 공부 시간에 따른 시험 성적 예측
predict = regression.predict([[2]]) # 2시간을 공부했을 때 선형 회귀 모델
print(predict)

regression_predict = linear_regression.predict(poly_reg.fit_transform([[2]]))
print(regression_predict)

linear_regression_score = linear_regression.score(x_poly, y)
print(linear_regression_score)
