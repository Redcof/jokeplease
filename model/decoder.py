from tensorflow.python.keras import Model

from model.attention_model import Attention_model
import tensorflow as tf


class Decoder(Model):
    def __init__(self, embed_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units = units
        self.attention = Attention_model(self.units)  # iniitalise your Attention model with units
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim)  # build your Embedding layer
        self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.d1 = tf.keras.layers.Dense(self.units)  # build your Dense layer
        self.d2 = tf.keras.layers.Dense(vocab_size)  # build your Dense layer

    def call(self, x=None, features=None, hidden=None):
        context_vector, attention_weights = self.attention(features,
                                                           hidden)  # create your context vector & attention weights from attention model
        embed = self.embed(x)  # embed your input to shape: (batch_size, 1, embedding_dim)
        embed = tf.concat([tf.expand_dims(context_vector, 1), embed],
                          axis=-1)  # Concatenate your input with the context vector from attention layer. Shape: (batch_size, 1, embedding_dim + embedding_dim)
        output, state = self.gru(
            embed)  # Extract the output & hidden state from GRU layer. Output shape : (batch_size, max_length, hidden_size)
        output = self.d1(output)
        output = tf.reshape(output, (-1, output.shape[2]))  # shape : (batch_size * max_length, hidden_size)
        output = self.d2(output)  # shape : (batch_size * max_length, vocab_size)

        return output, state, attention_weights

    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
