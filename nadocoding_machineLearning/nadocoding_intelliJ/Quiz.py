import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

read_csv = pd.read_csv("ScikitLearn/QuizData.csv")

total = read_csv.iloc[:, :-1].values
reception = read_csv.iloc[:, -1].values

train_total, test_total, train_reception, test_reception = train_test_split(
    total, reception, test_size=0.25, random_state=0
)


linear_regression = LinearRegression()
linear_regression.fit(train_total, train_reception)


plt.scatter(train_total, train_reception, color='blue')
plt.plot(train_total, linear_regression.predict(train_total), color='green')
plt.xlabel('total')
plt.ylabel('reception')
plt.title('Train')
plt.show()


plt.scatter(test_total, test_reception, color='blue')
plt.plot(train_total, linear_regression.predict(train_total), color='green')
plt.xlabel('total')
plt.ylabel('reception')
plt.title('Test')
plt.show()


print("Train Score : ", linear_regression.score(train_total, train_reception))
print("Test Score : ", linear_regression.score(test_total, test_reception))

print("If Total is 300 ? : ", np.around(linear_regression.predict([[300]])[0]).astype(int))
