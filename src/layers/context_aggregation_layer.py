class ContextAggregationLayer(tf.keras.layers.Layer):
    def __init__(self, context_fan_in, position_dim, num_samples, initial_sharpness, train_context, dtype=None):
        super(ContextAggregationLayer, self).__init__(dtype=dtype)

        self.context_fan_in = context_fan_in
        self.position_dim = position_dim
        self.num_samples = num_samples
        self.initial_sharpness = initial_sharpness
        self.train_context = train_context
    
    def build(self, input_shape):
        self.downstream_dim = input_shape[-1]
        self.downstream_num = input_shape[-2]

        # Encodes the positions of downstream columns
        # Positions are within a unit ball.
        self.downstream_positions = self.add_weight(
            shape=(self.downstream_num, self.position_dim),
            initializer=self.initialize_positions,
            name='downstream_positions',
            trainable=True,
            use_resource=True,
            constraint=tf.keras.constraints.MaxNorm(max_value=1, axis=1),
        )
        
        # These rotations form the "neighborhood" of a given position.    
        # Shape: position_dim -> position_dim * context_fan_in
        self.context_queries = Dense(
            self.position_dim * self.context_fan_in,
            use_bias=False,
            kernel_initializer=self.initialize_rotations,
            name='context_queries',
            trainable=self.train_context,
            kernel_constraint=self.constrain_to_rotations,
        )
        
        self.context_sharpness = self.add_weight(
            shape=(self.context_fan_in),
            initializer=tf.keras.initializers.Constant(value=self.initial_sharpness),
            name='context_sharpness',
            trainable=True,
            use_resource=True,
        )

        super(ContextAggregationLayer, self).build(input_shape)
    
    def initialize_positions(self, shape, dtype):
        rn = tf.keras.initializers.RandomNormal()(shape, dtype=dtype)
        # Initializing to smaller values initially helps positions converge faster.
        return 0.1 * tf.keras.constraints.UnitNorm(axis=1)(rn)
    
    def constrain_to_rotations(self, a):
        rs = tf.split(a, self.context_fan_in, axis=1)
        constrained_rs = []
        for r in rs:
            s, u, v = tf.linalg.svd(r)
            vh = tf.linalg.adjoint(v)
            constrained_rs.append(tf.matmul(u, vh))
            
        return tf.concat(constrained_rs, axis=1)
    
    def initialize_rotations(self, shape, dtype):
        identity = tf.keras.initializers.Identity()((self.position_dim, self.position_dim), dtype=dtype)
        rotations = []
        # The first rotation is always the identity (i.e. the seed value itself)
        rotations.append(identity)
        for i in range(1, self.context_fan_in):
            rotation = tf.keras.initializers.Orthogonal(gain=0.1)((self.position_dim, self.position_dim), dtype=dtype)
            rotations.append(identity + rotation)
        rotations = tf.concat(rotations, axis=1)
        return self.constrain_to_rotations(rotations)
    
    def sample_seed_indexes(self):
        # This tensorflow sampling method is inspired by https://stackoverflow.com/a/54755281
        # It approximates Numpy's np.random.choice with replace=False
        # Sampling different seed indexes for each batch helps the positions and contexts
        # to converge better to true neighborhoods. I was previously using a fixed sampling
        # throughout the whole training, and it led to the seed positions to get stuck
        # in specific values that didn't resemble their neighbors.

        uniform_distribution = tf.random.uniform([self.downstream_num], minval=0.0, maxval=1.0)
        _, top_indexes = tf.nn.top_k(uniform_distribution, self.num_samples)

        return top_indexes
    
    def reconstruct(self, full_context):        
        # Position shape: (..., num_columns, position_dim)
        # Value shape: (..., num_columns, context_fan_in * downstream_dim)
        position, values = tf.split(
            full_context,
            [
                self.position_dim,
                self.context_fan_in * self.downstream_dim,
            ],
            axis=-1,
        )
        
        # Restore normalization
        values *= np.sqrt(self.context_fan_in)

        # Shape: (context_fan_in, ..., num_columns, downstream_dim)
        values = tf.stack(tf.split(values, self.context_fan_in, axis=-1), axis=0)

        # Shape: (..., num_columns, context_fan_in * position_dim)
        queries = self.context_queries(position)
        
        # Shape: (context_fan_in, ..., num_columns, position_dim)
        queries = tf.stack(tf.split(queries, self.context_fan_in, axis=-1), axis=0)
        
        # Shape: (context_fan_in)
        exp_sharpness = tf.exp(self.context_sharpness)
        
        # Needed so we can call reconstruct directly when in mixed precision
        cast_downstream_positions = tf.cast(self.downstream_positions, self.compute_dtype)
        exp_sharpness = tf.cast(exp_sharpness, self.compute_dtype)
        queries = tf.cast(queries, self.compute_dtype)
        values = tf.cast(values, self.compute_dtype)
        
        # Process heads in sequence to reduce memory footprint
        reconstruction_shape = tf.concat([tf.shape(full_context)[0:-2], [self.downstream_num, self.downstream_dim]], axis=0)
        reconstruction = tf.zeros(reconstruction_shape, dtype=full_context.dtype)
        for head_idx in tf.range(0, self.context_fan_in):
            # Shape: (..., num_columns, position_dim)
            head_query = queries[head_idx]
            # Shape: (..., num_columns, downstream_dim)
            head_values = values[head_idx]
            # Shape: (1)
            head_sharpness = exp_sharpness[head_idx]
            # Shape: (..., num_samples, downstream_dim)
            head_output = self.compute_reconstruction_head_output(
                cast_downstream_positions,
                head_values,
                head_query,
                head_sharpness,
            )
            reconstruction = reconstruction + tf.cast(head_output, full_context.dtype)

        return reconstruction
    
    def call(self, downstream_input):
        # Shape: (num_samples, position_dim)
        seed_positions = tf.gather(self.downstream_positions, self.sample_seed_indexes(), axis=-2)
        
        # Shape: (num_samples, context_fan_in * position_dim)
        queries = self.context_queries(seed_positions)
        
        # Shape: (context_fan_in, num_samples, position_dim)
        queries = tf.stack(tf.split(queries, self.context_fan_in, axis=-1), axis=0)
        
        # Shape: (context_fan_in)
        exp_sharpness = tf.exp(self.context_sharpness)
        
        # Cast for mixed precision
        queries = tf.cast(queries, self.compute_dtype)
        exp_sharpness = tf.cast(exp_sharpness, self.compute_dtype)
        
        # Process heads in sequence to reduce memory footprint
        head_buffer = tf.TensorArray(dtype=self.compute_dtype, size=self.context_fan_in, clear_after_read=True)
        for head_idx in tf.range(0, self.context_fan_in):
            # Shape: (..., num_samples, position_dim)
            head_query = queries[head_idx]
            # Shape: (1)
            head_sharpness = exp_sharpness[head_idx]
            # Shape: (..., num_samples, downstream_dim)
            head_output = self.compute_attention_head_output(self.downstream_positions, downstream_input, head_query, head_sharpness)
            head_buffer = head_buffer.write(head_idx, head_output)

        # Restore normalization
        head_values = head_buffer.stack() / np.sqrt(self.context_fan_in)
        head_values = tf.unstack(head_values)
        
        # Shape: (..., num_samples, position_dim + context_fan_in * downstream_dim)
        expanded_seed_positions = tf.broadcast_to(
            seed_positions,
            tf.concat([tf.shape(head_values[0])[0:-1], [self.position_dim]], axis=0),
        )
        full_context = tf.concat([expanded_seed_positions] + head_values, axis=-1)

        return full_context
    
    @staticmethod
    @tf.function(jit_compile=True)
    #@tf.recompute_grad
    def compute_attention_head_output(downstream_positions, downstream_input, query, sharpness):
        # Shape: (downstream_num, position_dim), (num_samples, position_dim) -> (num_samples, downstream_num)
        attention_similarities = sharpness * tf.einsum('ND,SD->SN', downstream_positions, query)

        # Shape: (num_samples, downstream_num) -> (num_samples, downstream_num)
        attention_scores = tf.nn.softmax(attention_similarities, axis=-1)
        
        # Shape: (..., downstream_num, downstream_dim), (num_samples, downstream_num) -> (..., num_samples, downstream_dim)
        head_output = tf.einsum('...ND,SN->...SD', downstream_input, attention_scores)
        
        return head_output
    
    @staticmethod
    @tf.function(jit_compile=True)
    @tf.recompute_grad
    def compute_reconstruction_head_output(downstream_positions, values, query, sharpness):
        # Shape: (downstream_num, position_dim), (..., num_columns, position_dim) -> (..., num_columns, downstream_num)
        attention_similarities = sharpness * tf.einsum('ND,...SD->...SN', downstream_positions, query)
        
        # Shape: (..., num_columns, downstream_num) -> (..., num_columns, downstream_num)
        attention_scores = tf.nn.softmax(attention_similarities, axis=-1)
        
        # Shape: (..., num_columns, downstream_dim), (..., num_columns, downstream_num) -> (..., downstream_num, downstream_dim)
        head_reconstruction = tf.einsum('...CD,...CN->...ND', values, attention_scores)
        
        return head_reconstruction
