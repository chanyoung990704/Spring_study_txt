# One-Hot-Encoding
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import  mean_squared_error
from sklearn.metrics import r2_score

csv = pd.read_csv("./MultipleLinearRegressionData.csv")
x = csv.iloc[:, : -1]
y = csv.iloc[:, -1]
print(x)
print("-------")
print(y)

# drop : 다중 공선성 해결 remainder : 나머지는 그냥 통과
transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [2])], remainder='passthrough')

x = transformer.fit_transform(x)

# 1 0 : Home, 0 1 : Library, 0 0 : cafe
print(x)


# 데이터 셋 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 학습
regression = LinearRegression()
regression.fit(x_train, y_train)

# 예측 값과 실제 값 비교
y_pred = regression.predict(x_test)
print("y_pred  : ", y_pred)
print("y_test  :", y_test)

# 0 0으로 설정된 cafe의 기울기는 0이 된다.
# 처음 Home 두번째 Library
print("기울기", regression.coef_)
print("y절편", regression.intercept_)

# 모델평가
score = regression.score(x_train, y_train)
print("train 점수 ", score)


# 다양한 평가 지표 ( 회귀 모델 )
# MAE, MSE, RMSE, R^2

mae_err = mean_absolute_error(y_test, y_pred) # 실제 값과 예측 값의 MAE
print("mae : ", mae_err)
mse_err = mean_squared_error(y_test, y_pred)
print("mse : ", mse_err)
rmse_err = mean_squared_error(y_test, y_pred, squared=False)
print("rmse : ", rmse_err)
r2_err = r2_score(y_test, y_pred)
print("r2 : ", r2_err)