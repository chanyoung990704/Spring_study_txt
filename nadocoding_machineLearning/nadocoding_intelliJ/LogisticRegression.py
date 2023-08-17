# 공부 시간에 따른 자격증 시험 합격 가능성
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

read_csv = pd.read_csv("LogisticRegressionData.csv")

x = read_csv.iloc[:, : -1].values
y = read_csv.iloc[:, -1].values

# 데이터 분리

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=0)

# 학습 (로지스틱 회귀 모델)

classifier = LogisticRegression()
classifier.fit(x_train, y_train)

# 6시간 공부했을 때 합격 / 불합격

predict = classifier.predict([[6]]) # 0 : 불합격, 1: 합격
print(predict)
print("6시간 공부시 불합격 / 합격 확률 : ", classifier.predict_proba([[6]]))

predict = classifier.predict([[4]])
print(predict)
print("4시간 공부시 불합격 / 합격 확률 : ", classifier.predict_proba([[4]]))

# 분류 결과 예측 ( 테스트 셋 )
y_pred = classifier.predict(x_test)
print("예측 값 : ", y_pred, " 실제 값 : ", y_test)

print(classifier.score(x_test, y_test))

# 데이터 시각화 (훈련 셋)
x_range = np.arange(min(x), max(x), 0.1) # x의 최소값에서 최대값까지를 0.1 단위로 잘라서 데이터 생성
print(x_range)

p = 1 / (1 + np.exp(-(classifier.coef_ * x_range + classifier.intercept_))) # y = mx + b
print(p)

p = p.reshape(-1) # 1차원 배열 형태로 변환
print(p)

plt.scatter(x_train, y_train, color = 'pink')
plt.plot(x_range, p, color = 'green')
plt.plot(x_range, np.full(len(x_range), 0.5), color = 'red') # x_range 개수만큼 0.5로 가득찬 배열 만들기
plt.title('Probability by hours')
plt.xlabel('hours')
plt.ylabel('P')
plt.show()


# 데이터 시각화 (테스트 셋)
plt.scatter(x_test, y_test, color = 'pink')
plt.plot(x_range, p, color = 'green')
plt.plot(x_range, np.full(len(x_range), 0.5), color = 'red') # x_range 개수만큼 0.5로 가득찬 배열 만들기
plt.title('Probability by hours  (test) ')
plt.xlabel('hours')
plt.ylabel('P')
plt.show()

predict_proba = classifier.predict_proba([[4.5]])
print(predict_proba)

# 혼돈 행렬(Confusion Matrix)

# 불합격 (예측)      합격 (예측)
# 불합격 (실제)      불합격(실제)

# 불합격 (예측)      합격 (예측)
# 합격 (실제)        합격 (실제)
# => True Negative / False Positive / False Negative / True Positive
matrix = confusion_matrix(y_test, y_pred)
print(matrix)

