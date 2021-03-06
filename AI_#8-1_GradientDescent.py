x = 2
learning_rate = 0.05
precision = 0.00001
max_iterations = 500

#손실 함수를 람다식으로 정의한다.
loss_func = lambda x: (x-6)**2 + 10

#그래디언트를 람다식으로 정의한다. 손실함수의 1차 미분값이다.
gradient = lambda x: 2*x - 12

#그래디언트 강하법
for i in range(max_iterations):
    x = x - learning_rate *gradient(x)
    print("손실 함수값(", x, ")=", loss_func(x))

print("최소값 = ", x)