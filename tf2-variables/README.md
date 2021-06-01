## TF-variables-API

### Notes:

1. `tf.constant`could convert to numpy.
2. **ragged tensor** anology with cell in MATLAB, could contain the vector with different shape and type.
3. `tf.ragged.constant.to_tensor` could implement the missing value with 0.
4. `tf.SparseTensor`construct sparse tensor, and `tf.sparse.to_dense()`could turn sparse tensor to dense tensor, which reduce the memory usage. The param `indices` must be in order, otherwise, `tf.sparse.reorder`must be used to construct spare tensor. Sparse tensor couldn't plus int and float.
5. ''ResourceVariable'' object doesn't support item assignment, use `variable.assign()`instead.