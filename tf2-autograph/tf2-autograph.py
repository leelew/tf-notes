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


"""create your own autograph"""

# tf.function and autograph.


def scaled_elu(z, scale=1.0, alpha=1.0):
    # z>=0? scale* z: scale* alpha*tf.nn.elu(z)
    is_positive = tf.greater_equal(z, 0.0)
    return scale * tf.where(is_positive, z, alpha * tf.nn.elu(z))


print(scaled_elu(tf.constant(-3.)))
print(scaled_elu(tf.constant([-3., - 2.5])))

scaled_elu_tf = tf.function(scaled_elu)
print(scaled_elu_tf(tf.constant(-3.)))
print(scaled_elu_tf(tf.constant([-3., - 2.5])))

print(scaled_elu_tf.python_function is scaled_elu)

# tf model cost less time
%timeit scaled_elu(tf.random.normal((1000, 1000)))
%timeit scaled_elu_tf(tf.random.normal((1000, 1000)))

# 1+1/2 + 1/2^@ + ...


@tf.function
def converge_to_2(n_iters):
    total = tf.constant(0.)
    increment = tf.constant(1.)
    for _ in range(n_iters):
        total += increment
        increment /= 2.0
    return total


print(converge_to_2(20))


# to_code generated code, to_graph generate graph
def display_tf_code(func):
    code = tf.autograph.to_code(func)
    from IPython.display import display, Markdown
    display(Markdown('```python\n{}\n```'.format(code)))


display_tf_code(scaled_elu)
display_tf_code(converge_to_2)

# variable used in func must be initialize outside
var = tf.Variable(0.)


@tf.function
def add_21():
    return var.assign_add(21)


print(add_21())

# input_signature could constrait the type,shape of input.
# only use input_signature, tf could save this model using get_concrete_function.


@tf.function(input_signature=[tf.TensorSpec=([None], tf.int32, name='x')])
def cube(z):
    return tf.pow(z, 3)


print(cube(tf.constant([1., 2., 3.])))
print(cube(tf.constant([1, 2, 3])))

# @tf.function py func -> graph
# get_concrete_function -> add input signature -> SavedModel

cube_func_int32 = cube.get_concrete_function(
    tf.TensorSpec([None], tf.int32, name='x'))
print(cube_func_int32)

print(cube_func_int32 is cube.get_concrete_function(
    tf.TensorSpec([5], tf.int32, name='x')))

print(cube_func_int32 is cube.get_concrete_function(
    tf.constant([1, 2, 3])))

print(cube_func_int32.graph)  # function graph

print(cube_func_int32.graph.get_operations())  # function graph operation


# target function graph operation
pow_op = cube_func_int32.graph.get_operations()[2]
print(pow_op)

print(list(pow_op.inputs))
print(list(pow_op.outputs))

# get ops x
cube_func_int32.graph.get_operation_by_name("x")

# get tensor of x
cube_func_int32.graph.get_tensor_by_name("x:0")

# get note
cube_func_int32.graph.as_graph_def()
