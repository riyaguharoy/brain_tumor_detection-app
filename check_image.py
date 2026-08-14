import tensorflow as tf
model = tf.keras.models.load_model("brain_tumor_model.h5")
model.summary()
print(model.input_shape)