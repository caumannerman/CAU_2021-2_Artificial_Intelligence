import numpy as np
import matplotlib.pyplot as plt

X = np.array([0.0, 1.0, 2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0, 11.0, 12.0,13.0,14.0])
y = np.array([3.0, 3.5, 5.5, 6.0, 9.0,10.0,11.0,14.0,15.0,19.0,5.0,7.0,28.0,24.0,26.0])

W = 0 # 기울기
b = 0 # 절편

lrate = 0.01 # 학습률
epochs = 3 #반복 횟수

n = float(len(X)) # 입력 데이터의 갯수

# 경사 하강법
for i in range(epochs):
    y_pred = W*X + b # 예측값
    dW = (2/n) * sum(X * (y_pred-y))
    db = (2/n) *sum(y_pred-y)
    W = W - lrate * dW # 기울기 수정
    b = b - lrate *db # 절편 수정

# 기울기와 절편을 출력한다.
print(W, b)

#예측값 만들기
y_pred = W*X + b

#입력 데이터를 그래프 상에 찍는다.
plt.scatter(X, y)

#예측값은 선 그래프로 그린다.
plt.plot([min(X), max(X)], [min(y_pred), max(y_pred)], color='red')
plt.show()
