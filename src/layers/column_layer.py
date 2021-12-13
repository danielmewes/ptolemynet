class ColumnLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        upstream_dim,
        num_column_instances,
        position_dim,
        context_fan_in,
        context_num_samples,
        context_initial_sharpness,
        train_context,
        noise_rate,
        bypass_rate,
    ):
        super(ColumnLayer, self).__init__()

        self.upstream_dim = upstream_dim
        self.num_column_instances = num_column_instances
        self.position_dim = position_dim
        self.context_fan_in = context_fan_in
        self.context_num_samples = context_num_samples
        self.context_initial_sharpness = context_initial_sharpness
        self.train_context = train_context
        self.noise_rate = noise_rate
        self.bypass_rate = bypass_rate
    
    def build(self, input_shape):        
        self.context_aggregator = ContextAggregationLayer(
            context_fan_in=self.context_fan_in,
            position_dim=self.position_dim,
            num_samples=self.context_num_samples,
            initial_sharpness=self.context_initial_sharpness,
            train_context=self.train_context,
            dtype='mixed_float16',
        )
        
        self.permanence_separator = PermanenceSeparatorLayer(
            upstream_dim=self.upstream_dim,
            position_dim=self.position_dim,
            bypass_rate=self.bypass_rate,
            noise_rate=self.noise_rate,
        )
        
        self.column_selector = ColumnSelectorLayer(
            num_column_instances=self.num_column_instances,
        )
        
        super(ColumnLayer, self).build(input_shape)
    
    def reconstruct(self, upstream_state):
        reconstruction = self.permanence_separator.reconstruct(upstream_state)
        reconstruction = self.context_aggregator.reconstruct(reconstruction)
        return reconstruction
    
    def call(self, centered_input, training, is_bootstrapping=False):
        enriched_input = self.context_aggregator(centered_input)
        
        full_upstream_state, full_reconstruction = self.permanence_separator(
            enriched_input,
            is_bootstrapping=is_bootstrapping,
        )
        full_reconstruction = self.context_aggregator.reconstruct(full_reconstruction)

        upstream_state = self.column_selector(full_upstream_state)
        
        # Restore unit length for the selected upstream states
        upstream_state = tf.math.l2_normalize(upstream_state, axis=-1, epsilon=tf.keras.backend.epsilon())
        
        reconstruction = self.permanence_separator.reconstruct(upstream_state, training=training)
        reconstruction = self.context_aggregator.reconstruct(reconstruction)
        
        return [upstream_state, reconstruction, full_reconstruction]
