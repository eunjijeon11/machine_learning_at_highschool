import numpy as np
import tensorflow as tf

def sigmoid(x):
    return 1/(1+np.exp(-x))

and_x = np.array([[1,1],[1,0],[0,1],[0,0]])
and_y = np.array([[1],[0],[0],[0]])

w = tf.random.normal([2], 0, 1)
b = tf.random.normal([1], 0, 1)
a = 0.1

for i in range(10000):
    error_sum = 0
    for j in range(4):
        out = sigmoid(np.sum(and_x[j]*w)+b)
        error = and_y[j][0] - out
        w = w + and_x[j] * a * error
        b = b + a * error
        error_sum += error

print(w,b)
print("x1   x2  |realY|    y     |")
for i in range(4):
    print(and_x[i][0], "  " , and_x[i][1], "   " , and_y[i][0], "  " , sigmoid(np.sum(and_x[i] * w) + b)[0])
