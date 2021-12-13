class ColumnSelectorLayer(tf.keras.layers.Layer):
    def __init__(self, num_column_instances):
        super(ColumnSelectorLayer, self).__init__()
    
        self.num_column_instances = num_column_instances
        
    def build(self, input_shape):
        self.upstream_dim = input_shape[-1]
        
        # One query per column instance
        self.queries = self.add_weight(
            shape=(self.num_column_instances, self.upstream_dim),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=np.sqrt(2.0 / self.upstream_dim)),
            name='selector_queries',
            use_resource=True,
            trainable=True,
        )
        
        super(ColumnSelectorLayer, self).build(input_shape)
    
    def call(self, upstream_state):
        # Shape: (..., num_samples, upstream_dim), (..., num_column_instances, upstream_dim) -> (..., num_column_instances, num_samples)
        salience = tf.einsum('...SD,...ID->...IS', upstream_state, self.queries)

        salience = tf.nn.softmax(salience, axis=-1)
        
        # Shape: (..., num_column_instances, num_samples), (..., num_samples, upstream_dim) -> (..., num_column_instances, upstream_dim)
        selected_inputs = tf.einsum('...IS,...SD->...ID', salience, upstream_state)
        
        return selected_inputs
