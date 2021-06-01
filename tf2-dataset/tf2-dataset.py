
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


# tf-dataset-basic api
dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
print(dataset)

# ops
for item in dataset:
    print(item)  # tensor

# 1. repeat epoch
# 2. get batch

# repeat 3 times and got 7 batch each time
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)

# interleave:
# case: 文件名dataset -> 具体数据集
# 均匀混合

dataset2 = dataset.interleave(
    lambda v: tf.data.Dataset.from_tensor_slices(v),  # map_fn
    cycle_length=5,  # cycle_length
    block_length=5,  # block_length
)
for item in dataset2:
    print(item)

#
x = np.array([1, 2], [3, 4], [5, 6])
y = np.array(['cat', 'dog', 'fox'])

dataset3 = tf.data.Dataset.from_tensor_slices((x, y))
print(dataset3)

for item_x, item_y in dataset3:
    print(item_x.numpy(), item_y.numpy())

dataset4 = tf.data.Dataset.from_tensor_slices({"feature": x,
                                               "label": y})
for item in dataset4:
    print(item["feature"].numpy(), item["label"].numpy())
