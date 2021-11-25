import matplotlib.pylab as plt
from sklearn import linear_model

reg = linear_model.LinearRegression()

X = [[174], [152], [138], [128], [186],[165],[170],[180],[125],[185]]
y = [71, 55, 46, 38, 88,57,80,72,30,88]
reg.fit(X, y)   #학습

#predict할 x값 입력
test_x = 135
test_y = reg.predict([[test_x]])
print(test_y)

#학습 데이터와 y값을 산포도로 그린다.
plt.scatter(X, y, color='black')
# test_x와 그에 따른 output 쌍을 빨간색으로 plot
plt.scatter([[test_x]],[test_y], color='red' )

#학습 데이터를 입력으로 하여 예측값을 계산한다.
y_pred = reg.predict(X)

#학습데이터와 예측값으로 선그래프로 그린다.
# 계산된 기울기와 y절편을 가지는 직선이 그려진다.
plt.plot(X, y_pred, color='blue', linewidth=3)
plt.show()