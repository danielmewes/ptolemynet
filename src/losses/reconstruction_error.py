def compute_full_reconstruction_error(y_true, y_pred):
    y_true = tf.cast(tf.stop_gradient(y_true), tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Like cosine similarity, but weighted by the norm of the true vectors
    similarity = tf.reduce_sum(y_true * tf.math.l2_normalize(y_pred, axis=-1, epsilon=tf.keras.backend.epsilon()), axis=-1)
    weights = stable_norm(y_true, axis=-1)
    
    return tf.reduce_mean(weights) - tf.reduce_mean(similarity)

def compute_partial_reconstruction_error(y_true, y_pred):
    y_true = tf.cast(tf.stop_gradient(y_true), tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Compare the L1 norm size change when adding y_pred vectors to y_true vectors
    norm_y_true = y_true / tf.maximum(l1_norm(y_true, axis=-1, keepdims=True), tf.keras.backend.epsilon())
    norm_y_pred = y_pred / tf.maximum(l1_norm(y_pred, axis=-1, keepdims=True), tf.keras.backend.epsilon())
    partial_loss = 2.0 - l1_norm(norm_y_true + norm_y_pred, axis=-1)
    
    return tf.reduce_mean(partial_loss)
