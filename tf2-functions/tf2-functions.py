from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import time
import sys
import os
import sklearn
import matplotlib as mpl
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
% matplotlib inline


"""define loss function"""

# define mse function used to complie


def customized_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))


model.compile(loss=customized_mse, optimizer="sgd")


"""define layers"""

layer = tf.keras.layers.Dense(100)
layer = tf.keras.layers.Dense(100, input_shape=(None, 5))
# 类似于调用函数式调用layers生成
layer(tf.zeros([10, 5]))

layer.variables  # kernel, bias
layer.trainable_variables  # 可训练变量
help(layer)  # layer methods doc


"""define your own model"""


class CustomizedDenseLayer(keras.layers.Layer):
    def __init__(self, unit, activation=None, **kwargs):
        self.units = units
        self.activation = keras.layers.Activation(activation)
        # change object of son class to object of father class
        super(CustomizedDenseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """构建所需要的参数"""
        self.kernal = self.add_weight(name="kernal",
                                      shape=(input_shape[1], self.units),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.units),
                                    initialize='zeros',
                                    trainable=True)
        # inherit build in father class.
        super(CustomizedDenseLayer, self).build(input_shape)

    def call(self, x):
        """完成正向计算"""
        return self.activation(x @ self.kernal + self.bias)


# tf.nn.softplus: log(1+e^x)
customized_softplus = keras.layers.Lamda(lambda x: tf.nn.softplus(x))
print(customized_softplus([-10., -5., 0., 5., 10.]))

# your model construct by own define dense layer,
# output layer & softplus layer
model = keras.models.Sequential([
    CustomizedDenseLayer(30, activation='relu',
                         input_shape=x_train.shape[1, :]),
    CustomizedDenseLayer(1),
    customized_softplus,
    # keras.layers.Dense(1,acitvation="softplus")
])


"""define your own difference"""


def f(x):
    return 3. * x ** 2 + 2.*x-1


def approximate_derivative(f, x, eps=1e-3):
    return (f(x+eps)-f(x-eps))/(2.*eps)


def g(x1, x2):
    return (x1 + 5) * (x2 ** 2)


def approximate_gradient(g, x1, x2, eps=1e-3):
    dg_x1 = approximate_derivative(lambda x: g(x, x2), x1, eps)
    dg_x2 = approximate_derivative(lambda x: g(x1, x), x2, eps)
    return dg_x1, dg_x2


print(approximate_gradient(g, 2., 3.))

x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
with tf.GradientTape() as tape:
    z = g(x1, x2)

dz_x1 = tape.gradient(z, x1)
print(dz_x1)
# 只能用一次

try:
    dz_x2 = tape.gradient(z, x2)
except RuntimeError as ex:
    print(ex)


with tf.GradientTape(persistent=True) as tape:
    z = g(x1, x2)

dz_x1 = tape.gradient(z, x1)
print(dz_x1)
dz_x2 = tape.gradient(z, x2)
print(dz_x1, dz_x2)

del tape


with tf.GradientTape() as tape:
    z = g(x1, x2)

dz_x1x2 = tape.gradient(z, [x1, x2])

# constant 不能求导
x1 = tf.constant(2.0)
x2 = tf.constant(3.0)

with tf.GradientTape() as tape:
    tape.watch(x1)
    tape.watch(x2)
    z = g(x1, x2)

x = tf.Variable(5.0)
with tf.GradientTape() as tape:
    z1 = 3 * x
    z2 = x**2
tape.gradient([z1, z2], x)
# 是导数的求和


# Second derivative
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)

with tf.GradientTape(persistent=True) as outer_tape:
    with tf.GradientTape(persistent=True) as inner_tape:
        z = g(x1, x2)
    inner_grads = inner_tape.gradient(z, [x1, x2])
outer_grads = [outer_tape.gradient(inner_grad, [x1, x2])
               for inner_grad in inner_grads]

print(outer_grads)
del inner_tape
del outer_tape


# default gradient descent
learning_rate = 0.1
x = tf.Variable(0.0)

for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z, x)
    x.assign_sub(learning_rate * dz_dx)
print(x)


# use with optimizer

learning_rate = 0.1
x = tf.Variable(0.0)

optimizer = keras.optimizers.SGD(lr=learning_rate)
for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z, x)
    optimizer.apply_gradients([(dz_dx, x)])
print(x)

# metric

metric = keras.metrics.MeanSquaredError()
print(metric([5.], [2.]))
print(metric([0.], [1.]))
print(metric.result())  # 有累加数据并计算

metric.reset_states()
metric([1.], [3.])
print(metric.result())

# use with keras

epochs = 100
batch_size = 32
steps_per_epoch = len(x_train_scaled) // batch_size
optimizer = keras.optimizers.SGD()
metric = keras.metrics.MeanSquaredError()


def random_batch(x, y, batch_size=32):
    idx = np.random.randint(0, len(x), size=batch_size)
    return x[idx], y[idx]


# 1. batch 遍历训练集 metric
# 1.1 自动求导
# 2. epoch结束验证集上验证 metric
for epoch in range(epochs):
    metric.reset_states()
    for step in range(steps_per_epoch):
        x_batch, y_batch = random_batch(x_train_scaled, y_train, batch_size)

        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            loss = tf.reduce_mean(
                keras.losses.mean_squared_error(y_batch, y_pred))
            metric(y_batch, y_pred)
        grads = tape.gradient(loss, model.variables)
        grads_and_vars = zip(grads, model.variables)
        optimizer.apply_gradient(grads_and_vars)
        print("\rEpoch", epoch, "train mse:", metric.result().numpy(), end="")
    y_valid_pred = model(x_valid_scaled)
    valid_loss = tf.reduce_mean(
        keras.losses.mean_squared_error(y_valid_pred, y_valid))
    print("\t," "valid mse:", valid_loss.numpy())
