import os
import time
import utils
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from model.LSTM_CRF import LSTM_CRF
# from model.BiLSTM_CRF2 import BiLSTM_CRF2
from Config import Config

import loader
from loader import word_mapping, char_mapping, tag_mapping
from loader import prepare_dataset
from loader import augment_with_pretrained

from utils import save_mappings
from utils import create_input
tf.set_random_seed(1337)

con=Config()

# Load sentences
print("preparing data")

train_sentences = loader.load_sentences(con.dataset['traindata'], con.lower, con.zeros)
dev_sentences = loader.load_sentences(con.dataset['devdata'], con.lower, con.zeros)
test_sentences = loader.load_sentences(con.dataset['testdata'], con.lower, con.zeros)

dev_oov_sentences = loader.load_sentences(con.dev_oov, con.lower, con.zeros)
test_oov_sentences = loader.load_sentences(con.test_oov, con.lower, con.zeros)

# Create a dictionary / mapping of words
# If we use pretrained embeddings, we add them to the dictionary.
if con.model_para['emb_path'] != None:
    dico_words_train = word_mapping(train_sentences, con.lower)[0]
    dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        con.model_para['emb_path'],
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sentences + test_sentences])
        )
    )
    embedding_matrix = loader.get_lample_embedding(con.model_para['emb_path'], id_to_word, con.model_para['input_dim'])
else:
    dico_words, word_to_id, id_to_word = word_mapping(train_sentences, con.lower)
    dico_words_train = dico_words
    embedding_matrix = None


# Create a dictionary and a mapping for words / POS tags / tags
dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

# Index data
train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, con.lower
)
dev_data = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, con.lower
)
test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, con.lower
)
dev_oov_data = prepare_dataset(
    dev_oov_sentences, word_to_id, char_to_id, tag_to_id, con.lower
)
test_oov_data = prepare_dataset(
    test_oov_sentences, word_to_id, char_to_id, tag_to_id, con.lower
)

n_words = len(id_to_word)
n_chars = len(id_to_char)
n_tags = len(id_to_tag)

word_len = 37

singletons = set([word_to_id[k] for k, v
                  in dico_words_train.items() if v == 1])

print("%i / %i / %i / %i / %i sentences in train / dev / test / dev_oov / test_oov." % (
    len(train_data), len(dev_data), len(test_data), len(dev_oov_data), len(test_oov_data)))


# Save the mappings to disk
print('Saving the mappings to disk...')
save_mappings(con, id_to_word, id_to_char, id_to_tag)

print("building model")

num_steps=con.maxlen
num_epochs=con.epochs

# os._exit(0)

gpu_config = tf.ConfigProto(device_count={"CPU": 8}, # limit to num_cpu_core CPU usage
    inter_op_parallelism_threads = 8,
    intra_op_parallelism_threads = 8,
    allow_soft_placement=True,
    log_device_placement=False
)

gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.3


with tf.Session(config=gpu_config) as sess:
    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        model = LSTM_CRF(num_steps=num_steps, word_len=word_len, num_epochs=num_epochs, embedding_matrix=embedding_matrix, singletons=singletons)
        # model.build()
        
    print("training model")
    sess.run(tf.global_variables_initializer())
    model.train(sess, train_data, dev_data, test_data, dev_oov_data, test_oov_data)

