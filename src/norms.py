def stable_norm(t, axis, keepdims=False):
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(t), axis=axis, keepdims=keepdims), tf.keras.backend.epsilon()))

def l1_norm(t, axis, keepdims=False):
    return tf.reduce_sum(tf.abs(t), axis=axis, keepdims=keepdims)
