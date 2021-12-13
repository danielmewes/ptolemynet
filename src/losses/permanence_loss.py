def compute_change_non_sparseness(prev_state, cur_state):
    diff = cur_state - prev_state
    l1 = l1_norm(diff, axis=-1)
    # Minimum is to avoid negative values due to the espilon in stable_norm
    l2 = tf.minimum(l1, stable_norm(diff, axis=-1))
    
    bias = 0.1

    non_sparseness = (l1 + bias) / (l2 + bias)
    
    return non_sparseness

def compute_permanence_loss(prev_upstream_state, cur_upstream_state, prev_reconstruction, cur_reconstruction):
    prev_upstream_state = tf.cast(prev_upstream_state, tf.float32)
    cur_upstream_state = tf.cast(cur_upstream_state, tf.float32)
    prev_reconstruction = tf.cast(tf.stop_gradient(prev_reconstruction), tf.float32)
    cur_reconstruction = tf.cast(tf.stop_gradient(cur_reconstruction), tf.float32)
    
    upstream_non_sparseness = tf.reduce_mean(
        compute_change_non_sparseness(prev_upstream_state, cur_upstream_state),
        axis=-1,
    )
    reconstruction_non_sparseness = tf.reduce_mean(
        compute_change_non_sparseness(prev_reconstruction, cur_reconstruction),
        axis=-1,
    )
    
    non_sparseness_ratio = upstream_non_sparseness / reconstruction_non_sparseness
    
    return tf.reduce_mean(non_sparseness_ratio)
