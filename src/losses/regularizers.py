def compute_scale_loss(state):
    state = tf.cast(state, tf.float32)
    
    scale_loss = 1.0 - tf.math.tanh(stable_norm(state, axis=-1))
    
    return tf.reduce_mean(scale_loss)

def mean_l1_activity(t):
    t = tf.cast(t, tf.float32)
    return tf.reduce_mean(tf.reduce_sum(tf.abs(t), axis=-1))
