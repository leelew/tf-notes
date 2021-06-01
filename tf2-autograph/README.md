# tf.autograph

## `@tf.function`
1. Decorater `@tf.function` compile func into a graph, do **graph mode**, which cost less time than **eager mode**.
2. Any funcs called by annotated func will be run in graph mode.
3. If  graphs execute only few expensive ops (e.g., conv), you may not see much speed up.
4. If you use data-dependent control flow like `if`,`for`, AutoGraph will convert them to appropriate TensorFlow ops (e.g., `tf.cond()`)
5. AutoGraph give a low-level API to look at the generated code.

	```python
	print(tf.autograph.to_code())
	```
6. variable used in func must be initialize outside
7. `input_signature` could constrait the type, shape of input. only use `input_signature`, tf could save this model using `get_concrete_function`.
8. `@tf.function` py func -> graph
	 `get_concrete_function` -> add `input signature` -> SavedModel
