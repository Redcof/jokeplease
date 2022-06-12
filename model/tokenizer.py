import pathlib
import re
import tensorflow as tf

current_path = pathlib.Path(__file__).parents[0]


class FlikerTokenizer(object):
    # Create the vocabulary & the counter for the captions

    def __init__(self):
        with open(current_path / "weights/captions.txt", "r") as fp:
            lines = fp.readlines()
            annotations = [line.split(",")[1].strip(" ") for line in lines]

        # add the <start> & <end> token to all those captions as well
        annotations = [f"<start> {x} <end>" for x in annotations]

        # create the tokenizer
        top_freq_words = 5000

        special_chars = r'!"#$%&()*+.,-/:;=?@[\]^_`{|}~ '
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_freq_words,
                                                          oov_token="UNK",
                                                          filters=special_chars)
        # fit captions
        tokenizer.fit_on_texts(annotations)

        # Adding PAD to tokenizer list
        tokenizer.word_index['PAD'] = 0
        tokenizer.index_word[0] = 'PAD'

        cap_seqs = tokenizer.texts_to_sequences(annotations)
        # Pad each vector
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(cap_seqs,
                                                                   padding='post')
        self.tokenizer = tokenizer
        self.cap_vector = cap_vector
        self.max_length = 39
