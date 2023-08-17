# Linear Regression
## 공부 시간에 따른 시간 점수
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import MatplotlibDeprecationWarning
from sklearn.linear_model import SGDRegressor  # 확률적 경사 하강법


warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

##############선형 모델 학습 ##########################################################################

dataSet = pd.read_csv(r'C:\Users\ChanYoungPark\Desktop\study_txt\nadocoding_machineLearning\ScikitLearn\LinearRegressionData.csv')
dataSet.head()

# 독립 변수 설정

X = dataSet.iloc[:, : -1].values # row, column //  : => 전부 다 ,  : -1 => 마지막 직전까지

# 종속 변수 설정
Y = dataSet.iloc[:, -1].values


reg = LinearRegression() # 선형 모델 객체 생성
reg.fit(X, Y) # 선형 모델 학습 (모델 생성)// fit => 학습 // 독립변수, 종속변수

predict_val = reg.predict(X) # 모델의 학습 결과를 출력 // X에 대한 출력 값
print(predict_val)

# 산점도
plt.scatter(X, Y, color = 'blue')
# 선 그래프
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

# => y = mx + b     => y = ( reg.coef_ )x + (reg.intercept_)



# 데이터 셋 분리


x_data = dataSet.iloc[: , : -1].values
y_data = dataSet.iloc[:, -1].values

from sklearn.model_selection import train_test_split

# 튜플 형태로 값이 4개로 분리됨
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=10) ## test size => 20 %를 테스트 셋으로 갖는다


print(x_data)
print(len(x_data))

print(x_train)
print(len(x_train))

print(x_test)
print(len(x_test))

###############분리된 데이터를 통한 모델링##########################################



from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train) ## 훈련 데이터 셋으로 모델 학습



###################데이터 시각화(훈련 셋)###############################################


# 산점도
plt.scatter(x_train, y_train, color = 'blue')
# 선 그래프
plt.plot(x_train, reg.predict(x_train), color = 'green')
# 타이틀
plt.title('Score by hours ( train set) ')
# X축 이름
plt.xlabel('hours')
# Y축 이름
plt.ylabel('score')
# 그래프 출력
plt.show()



###############데이터 시각화(테스트 셋) ###########################################



# 산점도
plt.scatter(x_test, y_test, color = 'blue')
#선 그래프
plt.plot(x_test, reg.predict(x_test), color = 'green')
# 타이틀
plt.title('Score by hours  (test set)')
# X축 이름
plt.xlabel('hours')
# Y축 이름
plt.ylabel('score')
# 그래프 출력
plt.show()


#########################모델 평가############################################


score = reg.score(x_test, y_test) # 테스트 셋을 통한 모델 평가

print(score)


#################################################################################

# 경사하강법 (Gradient Descent)

sgd_regressor = SGDRegressor(max_iter=1000, eta0=1e-3, random_state=0, verbose=1) # 확률적 경사 하강법 모델
# max_iter = 훈련 세트 반복 횟수(epoch), eta0: 학습률(learning Rate)
# 지수 표기법
# 1e-3 : 0.001 ( 10^ -3 )
# 1e+4 : 10 ^ 4 = 10000
sgd_regressor.fit(x_train, y_train) # 모델 학습


# 산점도
plt.scatter(x_train, y_train, color = 'blue')
#선 그래프
plt.plot(x_train, sgd_regressor.predict(x_train), color = 'green')
# 타이틀
plt.title('Score by hours  (train set) SGD')
# X축 이름
plt.xlabel('hours')
# Y축 이름
plt.ylabel('score')
# 그래프 출력
plt.show()

print("기울기", sgd_regressor.coef_, "y 절편 : ", sgd_regressor.intercept_)

print("평가 점수 : ", sgd_regressor.score(x_test, y_test))



