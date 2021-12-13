class PtolemyLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_channels,
        num_instances_per_channel,
        context_fan_in,
        upstream_dim,
        position_dim,
        stride=1,
        permanence_loss_rho=0.01,
        initial_sharpness=3.0,
        noise_rate=0.1,
        bypass_rate=0.5,
        train_context=True,
    ):
        super(PtolemyLayer, self).__init__()
        
        self.num_channels = num_channels
        self.num_instances_per_channel = num_instances_per_channel
        self.context_fan_in = context_fan_in
        self.upstream_dim = upstream_dim
        self.position_dim = position_dim
        self.stride = stride
        self.permanence_loss_rho = permanence_loss_rho
        self.initial_sharpness = initial_sharpness
        self.noise_rate = noise_rate
        self.bypass_rate = bypass_rate
        self.train_context = train_context
        
        self.is_bootstrapping = tf.Variable(
            name='is_bootstrapping',
            initial_value=True,
            trainable=False,
            shape=(),
            dtype=tf.bool,
        )
    
    def get_config(self):
        return {
            'num_channels': self.num_channels,
            'num_instances_per_channel': self.num_instances_per_channel,
            'context_fan_in': self.context_fan_in,
            'upstream_dim': self.upstream_dim,
            'position_dim': self.position_dim,
            'stride': self.stride,
            'permanence_loss_rho': self.permanence_loss_rho,
            'initial_sharpness': self.initial_sharpness,
            'noise_rate': self.noise_rate,
            'bypass_rate': self.bypass_rate,
            'train_context': self.train_context,
        }
    
    def build(self, input_shape):
        self.downstream_num = input_shape[-2]
        self.downstream_dim = input_shape[-1]
        
        num_samples = int(self.downstream_num / self.stride)
        
        self.columns = []
        for c in range(0, self.num_channels):
            self.columns.append(
                ColumnLayer(
                    upstream_dim=self.upstream_dim,
                    num_column_instances=self.num_instances_per_channel,
                    position_dim=self.position_dim,
                    context_fan_in=self.context_fan_in,
                    context_num_samples=num_samples,
                    context_initial_sharpness=self.initial_sharpness,
                    train_context=self.train_context,
                    noise_rate=self.noise_rate,
                    bypass_rate=self.bypass_rate,
                )
            )
        
        self.bias = self.add_weight(
            shape=(self.downstream_num, self.downstream_dim),
            trainable=False,
            initializer=tf.keras.initializers.Zeros(),
            use_resource=True,
            name='bias',
        )
        
        super(PtolemyLayer, self).build(input_shape)
    
    @tf.function(jit_compile=True)
    def reconstruct(self, upstream_state):
        upstream_states = tf.split(upstream_state, self.num_channels, axis=-2)
        channel_reconstructions = []
        for c in range(0, self.num_channels):
            channel_reconstruction = self.columns[c].reconstruct(upstream_states[c])
            channel_reconstructions.append(channel_reconstruction)
            
        reconstruction = self.reconstruct_from_channel_reconstructions(channel_reconstructions)
        
        return reconstruction
    
    def reconstruct_from_channel_reconstructions(self, channel_reconstructions):       
        reconstruction = tf.math.accumulate_n(channel_reconstructions)

        reconstruction += 0.01 * tf.cast(self.bias, self.compute_dtype)
    
        # Normalize reconstruction
        reconstruction = tf.math.l2_normalize(reconstruction, axis=-1, epsilon=tf.keras.backend.epsilon())
        
        return reconstruction
    
    def call(self, downstream_input, training=False):
        # Update bias to be the mean of the inputs
        batch_mean = tf.reduce_mean(tf.reduce_mean(tf.cast(downstream_input, tf.float32), axis=0), axis=0)
        if training:
            self.bias.assign(0.9 * tf.cast(self.bias, tf.float32) + 0.1 * batch_mean)
        
        upstream_states = []
        channel_reconstructions = []
        channel_full_reconstructions = []
        for c in range(0, self.num_channels):
            upstream_state, channel_reconstruction, channel_full_reconstruction = self.columns[c](
                downstream_input,
                is_bootstrapping=self.is_bootstrapping,
            )
 
            upstream_states.append(upstream_state)
            channel_reconstructions.append(channel_reconstruction)
            channel_full_reconstructions.append(channel_full_reconstruction)
        
        upstream_state = tf.concat(upstream_states, axis=-2)
        reconstruction = self.reconstruct_from_channel_reconstructions(channel_reconstructions)
        full_reconstruction = self.reconstruct_from_channel_reconstructions(channel_full_reconstructions)

        if self.permanence_loss_rho:
            # Currently only works for pairs
            permanence_loss = compute_permanence_loss(
                upstream_state[:, 0],
                upstream_state[:, 1],
                reconstruction[:, 0],
                reconstruction[:, 1],
            )
            self.add_metric(permanence_loss, 'permanence_loss')
            self.add_loss(self.permanence_loss_rho * (1.0 - tf.cast(self.is_bootstrapping, tf.float32)) * permanence_loss)
        
        full_reconstruction_error = compute_full_reconstruction_error(downstream_input, full_reconstruction)
        self.add_metric(full_reconstruction_error, 'full_reconstruction_error')
        self.add_loss(tf.cast(self.is_bootstrapping, tf.float32) * full_reconstruction_error)

        reconstruction_error = compute_full_reconstruction_error(downstream_input, reconstruction)
        self.add_metric(reconstruction_error, 'reconstruction_error')
        self.add_loss(reconstruction_error)

        return [upstream_state, reconstruction]
