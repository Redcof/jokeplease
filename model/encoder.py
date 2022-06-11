class Encoder(Model):
    def __init__(self, embed_dim):
        super(Encoder, self).__init__()
        self.dense = tf.keras.layers.Dense(embed_dim, activation="relu")  # build your Dense layer with relu activation

    def call(self, features):
        features = self.dense(features)  # extract the features from the image shape: (batch, 8*8, embed_dim)
        return features
