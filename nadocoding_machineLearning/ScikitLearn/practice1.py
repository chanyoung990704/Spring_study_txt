# Linear Regression
## 공부 시간에 따른 시간 점수

import matplotlib.pyplot as plt
import pandas as pd


dataSet = pd.read_csv(r'C:\Users\ChanYoungPark\Desktop\study_txt\nadocoding_machineLearning\ScikitLearn\LinearRegressionData.csv')
dataSet.head()

# 독립 변수 설정

X = dataSet.iloc[:, : -1].values # row, colunm //  : => 전부 다 ,  : -1 => 마지막 직전까지

# 종속 변수 설정
Y = dataSet.iloc[:, -1].values


from sklearn.linear_model import LinearRegression
reg = LinearRegression() # 선형 모델 객체 생성
reg.fit(X, Y) # 선형 모델 학습 (모델 생성)// fit => 학습 // 독립변수, 종속변수

predict_val = reg.predict(X) # 모델의 학습 결과를 출력 // X에 대한 출력 값
print(predict_val)

# 산점도
plt.scatter(X, Y, color = 'blue')
#선 그래프
plt.plot(X, predict_val, color = 'green')
# 타이틀
plt.title('Score by hours')
# X축 이름
plt.xlabel('hours')
# Y축 이름
plt.ylabel('score')
# 그래프 출력
plt.show()


###
print('9시간 공부 예상 점수는 ? ', reg.predict([    [9]      ])) # 입력했을 때 변수 자료형에 맞게 변수를 설정해야 


# 기울기 출력
print(reg.coef_)

# y 절편
print(reg.intercept_)

###### => y = mx + b     => y = ( reg.coef_ )x + (reg.intercept_)



