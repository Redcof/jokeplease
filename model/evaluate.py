max_length = 39
attention_features_shape = 64

from model.decoder import Decoder
from model.encoder import Encoder

embedding_dim = 256
units = 512
vocab_size = 5001  # top 5,000 words +1

decoder = Decoder(embedding_dim, units, vocab_size)
decoder.load_weights('weigths/decoder.tf')

encoder = Encoder(embedding_dim)
encoder.load_weights('weigths/encoder.tf')


def evaluate(image):
    """Greedy search decoder"""
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image),
                                0)  # process the input image to desired format before extracting features
    img_tensor_val = image_features_extract_model(temp_input)  # Extract features using our feature extraction model
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)  # extract the features by passing the input to encoder

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        # getting probability distribution
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)  # get the output from decoder
        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        # extract the predicted id(embedded value) which carries the max value
        # taking token-id with largest probability
        predicted_id = tf.argmax(predictions[0]).numpy()

        # createing result sentense
        result.append(tokenizer.index_word[
                          predicted_id])  # map the id to the word from tokenizer and append the value to the result list

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot, predictions

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot, predictions
