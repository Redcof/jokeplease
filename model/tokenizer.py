import pathlib
import re
import tensorflow as tf

import pandas as pd

current_path = pathlib.Path(__file__).parents[0]


class FlikerTokenizer(object):
    # Create the vocabulary & the counter for the captions

    def __init__(self):
        df = pd.read_csv(current_path / "weights/captions.txt")
        annotations = df['caption']  # write your code here

        # add the <start> & <end> token to all those captions as well
        annotations = annotations.apply(lambda x: f"<start> {x} <end>")
        df['Captions'] = annotations

        # create the tokenizer
        top_freq_words = 5000

        special_chars = r'!"#$%&()*+.,-/:;=?@[\]^_`{|}~ '
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_freq_words,
                                                          oov_token="UNK",
                                                          filters=special_chars)
        # fit captions
        tokenizer.fit_on_texts(df.Captions)

        # Adding PAD to tokenizer list
        tokenizer.word_index['PAD'] = 0
        tokenizer.index_word[0] = 'PAD'

        cap_seqs = tokenizer.texts_to_sequences(df.Captions)
        # Pad each vector
        cap_vector = tf.keras.preprocessing.sequence.pad_sequences(cap_seqs,
                                                                   padding='post')
        self.tokenizer = tokenizer
        self.cap_vector = cap_vector
        self.max_length = 39
