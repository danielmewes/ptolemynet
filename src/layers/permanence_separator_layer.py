class PermanenceSeparatorLayer(tf.keras.layers.Layer):
    def __init__(self, upstream_dim, position_dim, bypass_rate, noise_rate):
        super(PermanenceSeparatorLayer, self).__init__()
        
        self.upstream_dim = upstream_dim
        self.position_dim = position_dim
        self.bypass_rate = bypass_rate
        self.noise_rate = noise_rate
    
    def build(self, input_shape):
        self.internal_dim = input_shape[-1]

        # A note on kernel intitializers: Because we renormalize vectors after encoding and decoding them,
        # the usual He normal / Xavier normalization strategies don't quite apply here. We instead just
        # initialize with a normal distribution at a small stddev.
        self.separation_layers = [
            Dense(
                np.ceil(2 * self.internal_dim / 8.0) * 8,
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2=1e-5),
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
                name='separation_layer_1',
            ),
            LeakyReLU(alpha=0.1),
            Dense(
                np.ceil(2 * self.internal_dim / 8.0) * 8,
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2=1e-5),
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
                name='separation_layer_2',
            ),
            LeakyReLU(alpha=0.1),
            Dense(
                self.upstream_dim,
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2=1e-5),
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
                name='separation_layer_3',
            )
        ]
        
        self.reconstruction_layers = [
            Dense(
                np.ceil(2 * self.internal_dim / 8.0) * 8,
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2=1e-5),
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
                name='reconstruction_layer_1',
            ),
            LeakyReLU(alpha=0.1),
            Dense(
                np.ceil(2 * self.internal_dim / 8.0) * 8,
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2=1e-5),
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
                name='reconstruction_layer_2',
            ),
            LeakyReLU(alpha=0.1),
            Dense(
                self.internal_dim,
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(l2=1e-5),
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
                name='reconstruction_layer_3',
            )
        ]
        
        if self.noise_rate:
            self.noise = GaussianNoise(stddev=self.noise_rate/np.sqrt(self.upstream_dim))
        else:
            self.noise = None
        
        super(PermanenceSeparatorLayer, self).build(input_shape)
     
    @tf.function(jit_compile=True)
    def reconstruct(self, separated_input, training=False):
        # Inject noise to increase robustness.
        if self.noise:
            separated_input = self.noise(separated_input, training=training)
            separated_input = tf.math.l2_normalize(separated_input, axis=-1, epsilon=tf.keras.backend.epsilon())
        
        enriched_input = separated_input
        for l in self.reconstruction_layers:
            enriched_input = l(enriched_input)
        
        return enriched_input
    
    def call(self, enriched_input, is_bootstrapping, training):        
        separated_input = enriched_input
        for l in self.separation_layers:
            separated_input = l(separated_input)
        
        # Normalize upstream states
        normalized_separated_input = tf.math.l2_normalize(separated_input, axis=-1, epsilon=tf.keras.backend.epsilon())
        
        # Try to keep the state sparse if we can - this makes the work of upstream layers easier, and encourages
        # the encoder to identify distinct recurring patterns in the inputs.
        activity_reg = mean_l1_activity(separated_input)
        self.add_metric(activity_reg, 'mean_l1_activity')
        self.add_loss(0.01 * activity_reg)
        
        reconstruction = self.reconstruct(normalized_separated_input, training=training)
        
        # Add a local reconstruction loss just for the separator mapping. This makes convergence of the overall
        # network more reliable and efficient.
        # For our loss here, we want to treat the input as a ground truth. Hence we stop gradient propagation along the
        # input. (In particular, we do *not* want to encourage the input to change to make it easy for us to fit better
        # in this local loss. There will be a separate global reconstruction loss at the end to make sure end to end
        # reconstruction works.)
        no_gradient_enriched_input = tf.stop_gradient(enriched_input)
        # Require full reconstruction for the position during bootstrapping to speed up convergence of positions.
        # After bootstrapping, we relax this constraint so the network can choose how much precision is really
        # needed for a given part of the position.
        position_reconstruction_error = compute_full_reconstruction_error(
            no_gradient_enriched_input[:, :, :, 0:self.position_dim],
            reconstruction[:, :, :, 0:self.position_dim],
        )
        self.add_metric(position_reconstruction_error, 'position_reconstruction_error')
        self.add_loss(0.5 * tf.cast(is_bootstrapping, tf.float32) * position_reconstruction_error)
        
        # The purpose of the scale loss is to add numeric stability by shrinking the factor
        # by which the normalization of the results will need to scale up the output of the
        # relu layers. It also ensures that the l2 regularization of the layer weights
        # has the desired effect of keeping the reconstruction robust under small upstream state
        # changes.
        upstream_scale_loss = compute_scale_loss(separated_input)
        self.add_metric(upstream_scale_loss, 'upstream_scale_loss')
        self.add_loss(0.01 * upstream_scale_loss)
        reconstruction_scale_loss = compute_scale_loss(reconstruction[:, :, :, self.position_dim:])
        self.add_metric(reconstruction_scale_loss, 'reconstruction_scale_loss')
        self.add_loss(0.01 * reconstruction_scale_loss)

        # During bootstrapping, add a "cheat" bypass to better bootstrap the locality mapping
        if training:
            bypass_rate = tf.cast(is_bootstrapping, tf.float32) * self.bypass_rate
            m = tf.shape(enriched_input)[0]
            toss = tf.random.uniform(shape=[m, 1, 1, 1], minval=0.0, maxval=1.0)
            bypass_mask = tf.cast(toss < bypass_rate, dtype=self.compute_dtype)
            reconstruction = bypass_mask * enriched_input + (1.0 - bypass_mask) * reconstruction
        
        return [normalized_separated_input, reconstruction]
