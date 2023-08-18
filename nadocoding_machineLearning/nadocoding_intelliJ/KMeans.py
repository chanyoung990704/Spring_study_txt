# K - Means
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


read_csv = pd.read_csv("ScikitLearn/KMeansData.csv")

print(read_csv[: 5])

# 비지도학습에서는 y 존재하지 않음
x = read_csv.iloc[:, :].values


# 데이터 시각화
plt.scatter(x[:, 0], x[:, 1]) # x축 : hour , y축 : score
plt.title('Score By hours')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()



# 데이터 시각화 (축 범위 통일)
plt.scatter(x[:, 0], x[:, 1]) # x축 : hour , y축 : score
plt.title('Score By hours')
plt.xlabel('hours')
plt.ylabel('score')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.show()

# 피처 스케일링(Feature Scaling)
scaler = StandardScaler()
x = scaler.fit_transform(x)
print(x)

# 데이터 시각화 ( 스케일링된 데이터 )
plt.scatter(x[:, 0], x[:, 1]) # x축 : hour , y축 : score
plt.title('Score By hours')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()

# 최적의 K 값 찾기 (Elbow Method)
inertia_list = []
for i in range(1, 11):
    k_means = KMeans(n_clusters=i, init="k-means++", random_state=0)
    k_means.fit(x)
    inertia_list.append(k_means.inertia_)  # 각 지점으로부터 클러스터의 중심(centroid)까지의 거리의 제곱의 합


plt.plot(range(1, 11), inertia_list)
plt.title('Elbow Label')
plt.xlabel('n_cluster')
plt.ylabel('inertia')
plt.show()

# 최적의 K값으로 Kmeans 학습
n_cluster = 4 # 최적의 K 값
centers_ = k_means.cluster_centers_
k = KMeans(n_clusters=n_cluster, random_state=0)
k_fit_predict = k.fit_predict(x)
print(k_fit_predict)

# 데이터 시각화 (최적의 K)
for i in range(n_cluster):
    plt.scatter(x[k_fit_predict == i, 0], x[k_fit_predict == i, 1], s=100, edgecolor='black')
    plt.scatter(centers_[i, 0], centers_[i, 1], s=300, edgecolors='pink', color='gray', marker='s')
    plt.text(centers_[i, 0], centers_[i, 1], i, va='center', ha='center', color='red') # 텍스트 출력

plt.title('scoreByHours')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()


# 데이터 시각화(스케일링 원복)
inverse_transform = scaler.inverse_transform(x) # Feature Scaling된 데이터를 원복
print(inverse_transform[:5])
inverse_centers = scaler.inverse_transform(centers_)
print(inverse_centers)

# 데이터 시각화 (최적의 K)
for i in range(n_cluster):
    plt.scatter(inverse_transform[k_fit_predict == i, 0], inverse_transform[k_fit_predict == i, 1], s=100, edgecolor='black')
    plt.scatter(inverse_centers[i, 0], inverse_centers[i, 1], s=300, edgecolor='pink', color='gray', marker='s')
    plt.text(inverse_centers[i, 0], inverse_centers[i, 1], i, va='center', ha='center', color='red') # 텍스트 출력

plt.title('scoreByHours')
plt.xlabel('hours')
plt.ylabel('score')
plt.show()