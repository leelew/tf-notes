import os
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

%matplotlib inline


"""tf-basic-api"""

"""default dataType"""

# tensor for list
t = tf.constant([[1., 2., 3.], [4., 5., 6.]])

# slice of tensor
print(t)
print(t[:, 1:])
print(t[..., 1])

# ops
print(t + 10)
print(tf.square(t))
print(t @ tf.transpose(t))  # t^t'

# numpy convert tensor & tensor convert numpy
print(t.numpy())
print(np.square(t))
np_t = np.array([[1., 2., 3.], [4., 5., 6.]])
print(tf.constant(np_t))

# Scalars
t = tf.constant(2.718)
print(t.numpy())
print(t.shape)


# tensor for string
t = tf.constant("cafe")
print(t)
print(tf.strings.length(t))
print(tf.strings.length(t, unit="UTF8_CHAR"))
print(tf.strings.unicode_decode(t, "UTF8"))

# string array
t = tf.constant(["cafe", "coffee", "咖啡"])
print(tf.strings.length(t, unit="UTF8_CHAR"))
r = tf.strings.unicode_decode(t, "UTF8")
print(r)


# ragged tensor
r = tf.ragged.constant([[11, 12], [21, 22, 23], [], [41]])
print(r)
print(r[1])
print(r[1:2])

# ops on ragged tensor
r2 = tf.ragged.constant([[51, 52], [], [71]])
print(tf.concat([r, r2], axis=0))
r3 = tf.ragged.constant([[13, 14], [15], [], [42, 43]])
print(tf.concat([r, r3], axis=1))
print(r.to_tensor())  # 空闲位置用0补齐


# sparse tensor
s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]],
                    values=[1., 2., 3.],
                    dense_shape=[3, 4])  # indices必须要排好序，否则要用tf.sparse.reorder()
print(s)
print(tf.sparse.to_dense(s))

# ops on sparse tensors
s2 = s * 2.0
print(s2)

try:
    s3 = s + 1.0
except TypeError as ex:
    print(ex)

s4 = tf.constant([[10., 20.],
                  [30., 40.],
                  [50., 60.],
                  [70., 80.]])

print(tf.sparse.sparse_dense_matmul(s, s4))


# variables
v = tf.Variable([[1., 2., 3.], [4., 5., 6]])
print(v)
print(v.value())
print(v.numpy())

# assign value
v.assign(2 * v)
print(v.numpy())
v[0, 1].assign(42)
print(v.numpy())
v[1].assign([7., 8., 9.])
print(v.numpy())

try:
    v[1] = [7., 8., 9.]
except TypeError as ex:
    print(ex)
