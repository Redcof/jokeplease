import pathlib

import numpy as np

from model.decoder import Decoder
from model.encoder import Encoder
import tensorflow as tf

embedding_dim = 256
units = 512
vocab_size = 5001  # top 5,000 words +1

BATCH_SIZE = 16  # 31 # 93 # 279
BUFFER_SIZE = 1000
AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_SHAPE = (299, 299)

attention_features_shape = 64

current_path = pathlib.Path(__file__).parents[0]


class Evaluate(object):

    def __init__(self, tokenizer, max_length=39):
        self.decoder = Decoder(embedding_dim, units, vocab_size)
        self.decoder.load_weights(str(current_path / 'weights/decoder.tf'))

        self.encoder = Encoder(embedding_dim)
        self.encoder.load_weights(str(current_path / 'weights/encoder.tf'))

        # prepare image feature extraction model
        image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

        new_input = image_model.input  # write code here to get the input of the image_model
        hidden_layer = image_model.layers[-1].output  # write code here to get the output of the image_model

        self.image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

        self.tokenizer = tokenizer
        self.max_length = max_length

    # This function returns pre-processed image
    def load_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMG_SHAPE)  # reshape
        image = tf.keras.applications.inception_v3.preprocess_input(image)
        return image  # , image_path

    # this function extract inceptionv3 features from the given image.
    def extract_inceptionv3_features(self, img):
        # extract feature
        batch_features = self.image_features_extract_model(img)
        # reshape feature
        batch_features = tf.reshape(batch_features, (BATCH_SIZE, 8 * 8, 2048))
        return batch_features

    def evaluate(self, image):
        """Greedy search decoder"""
        attention_plot = np.zeros((self.max_length, attention_features_shape))

        hidden = self.decoder.reset_state(batch_size=1)

        temp_input = tf.expand_dims(self.load_image(image),
                                    0)  # process the input image to desired format before extracting features
        img_tensor_val = self.image_features_extract_model(
            temp_input)  # Extract features using our feature extraction model
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        features = self.encoder(img_tensor_val)  # extract the features by passing the input to encoder

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(self.max_length):
            # getting probability distribution
            predictions, hidden, attention_weights = self.decoder(dec_input, features,
                                                                  hidden)  # get the output from decoder
            attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

            # extract the predicted id(embedded value) which carries the max value
            # taking token-id with largest probability
            predicted_id = tf.argmax(predictions[0]).numpy()

            # createing result sentense
            result.append(self.tokenizer.index_word[
                              predicted_id])  # map the id to the word from tokenizer and append the value to the result list

            if self.tokenizer.index_word[predicted_id] == '<end>':
                break

            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        return result
