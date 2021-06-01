## basic API and integrate Keras

1. Define your own loss function.  **`tf.reduce_mean`**

2. Define your own layer.(including dense, activation)
	1) using class inherit father class `keras.layers.Layer`, and construct dense layers in your favor.
	2) using `tf.keras.layers.Dense()`.
	
	**Notes**
	```
	in TF2, dense layers could be generate by func-type, such as 
	layer = tf.keras.layers.Dense(100)
	layer(tf.zeros([10,5]))
	```
3. Define your own model.
	```
	model = keras.models.Sequential([
    CustomizedDenseLayer(),# your own layers
    CustomizedDenseLayer(1),# your own output layer
    customized_softplus,# your own activation layer
   ```
4. Define your own diff-calculate method.
	`tf.GradientTape()` only could use once. if you want to caculate gradient dz/dx1, dz/dx2, parameter persistence=True must be used. and in each iter, del tape is necessary.
	if you want to interact your own diff method with Keras, `optimizer.apply_gradients()`could achieve.