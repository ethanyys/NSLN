import codecs
import os, types
import math
import utils
import numpy as np
import random
import tensorflow as tf
from tensorflow.contrib import rnn
from Config import Config
from inference.CRF import CRF

from utils import reload_mappings
tf.set_random_seed(1337)

class CharWordBiLSTM(object):
    def __init__(self, num_words, num_chars, num_classes, num_steps, word_len, embedding_matrix=None):
        # Parameter
        self.config = Config()
        self.dropout_rate = self.config.model_para['dropout_rate']
        self.batch_size = self.config.model_para['batch_size']
        self.num_layers = self.config.model_para['lstm_layer_num']
        self.input_dim = self.config.model_para['input_dim']
        self.hidden_dim = self.config.model_para['hidden_dim']
        self.char_input_dim = self.config.model_para['char_input_dim']
        self.char_hidden_dim = self.config.model_para['char_hidden_dim']
        self.use_pa_learning = self.config.model_para['use_pa_learning']
        
        self.embedding_matrix = embedding_matrix
        
        self.word_len = word_len
        self.num_steps = num_steps
        self.num_words = num_words
        self.num_chars = num_chars
        self.num_classes = num_classes
        
        
        self.char_inputs = tf.placeholder(tf.int32, [None, self.word_len])

        with tf.variable_scope("character-based-emb"):
            # char embedding
            self.char_embedding = tf.get_variable("char_emb", [self.num_chars, self.char_input_dim])

            self.char_inputs_emb = tf.nn.embedding_lookup(self.char_embedding, self.char_inputs)
            self.char_inputs_emb = tf.transpose(self.char_inputs_emb, [1, 0, 2])
            self.char_inputs_emb = tf.reshape(self.char_inputs_emb, [-1, self.char_input_dim])
            self.char_inputs_emb = tf.split(self.char_inputs_emb, self.word_len, 0)
            
        # char forward and backward
        with tf.variable_scope("char-bi-lstm"):
            # char lstm cell
            char_lstm_cell_fw = rnn.LSTMCell(self.char_hidden_dim)
            char_lstm_cell_bw = rnn.LSTMCell(self.char_hidden_dim)

            # get the length of each word
            self.word_length = tf.reduce_sum(tf.sign(self.char_inputs), reduction_indices=1)
            self.word_length = tf.cast(self.word_length, tf.int32)

            char_outputs, f_output, r_output = tf.contrib.rnn.static_bidirectional_rnn(
                char_lstm_cell_fw, 
                char_lstm_cell_bw,
                self.char_inputs_emb, 
                dtype=tf.float32,
                sequence_length=self.word_length
            )
        final_word_output = tf.concat([f_output.h, r_output.h], -1)

        self.word_lstm_last_output = tf.reshape(final_word_output, [-1, self.num_steps, self.char_hidden_dim*2])
        
        # '''
        #     word input
        # '''
        with tf.variable_scope("word-based-emb"):
            self.inputs = tf.placeholder(tf.int32, [None, self.num_steps])
            # self.targets = tf.placeholder(tf.int32, [None, self.num_steps])
            if self.use_pa_learning:
                self.targets = tf.placeholder(tf.float32, [None, self.num_steps+2, self.num_classes+1])
            else:
                self.targets = tf.placeholder(tf.int32, [None, self.num_steps])
            self.targets_transition = tf.placeholder(tf.int32, [None])
            self.keep_prob = tf.placeholder(tf.float32)

            if embedding_matrix is not None:
                self.embedding = tf.Variable(embedding_matrix, trainable=True, name="word_emb", dtype=tf.float32)
            else:
                self.embedding = tf.get_variable("word_emb", [self.num_words, self.input_dim])

            self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.inputs)
            self.inputs_emb = tf.concat([self.inputs_emb, self.word_lstm_last_output], -1)

            self.inputs_emb = tf.nn.dropout(self.inputs_emb, self.keep_prob)
            self.inputs_emb = tf.transpose(self.inputs_emb, [1, 0, 2])
            self.inputs_emb = tf.reshape(self.inputs_emb, [-1, self.input_dim+self.char_hidden_dim*2])
            self.inputs_emb = tf.split(self.inputs_emb, self.num_steps, 0)

            # word lstm cell
            lstm_cell_fw = rnn.LSTMCell(self.hidden_dim)
            lstm_cell_bw = rnn.LSTMCell(self.hidden_dim)

            # get the length of each sample
            self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
            self.length = tf.cast(self.length, tf.int32) 
        
        # forward and backward
        with tf.variable_scope("word-bi-lstm"):
            self.outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                lstm_cell_fw, 
                lstm_cell_bw,
                self.inputs_emb, 
                dtype=tf.float32,
                sequence_length=self.length
            )

        # bidirect concat
        final_outputs = tf.reshape(tf.concat(self.outputs, 1), [-1, self.hidden_dim * 2])
        tanh_layer_w = tf.get_variable("tanh_layer_w", [self.hidden_dim * 2, self.hidden_dim])
        tanh_layer_b = tf.get_variable("tanh_layer_b", [self.hidden_dim])
        self.final_outputs = tf.tanh(tf.matmul(final_outputs, tanh_layer_w) + tanh_layer_b)

        
 
    # def add_placeholders(self):
#         '''
#             char input = sen_batch * sen_len
#         '''
#         self.char_inputs = tf.placeholder(tf.int32, [None, self.word_len])
#         '''
#             word input
#         '''
#         self.inputs = tf.placeholder(tf.int32, [None, self.num_steps])
#         self.targets = tf.placeholder(tf.int32, [None, self.num_steps])
#         self.targets_transition = tf.placeholder(tf.int32, [None])
#         self.keep_prob = tf.placeholder(tf.float32)

#     def add_lookup_op(self):
#         with tf.variable_scope("character-based-emb"):
#             # char embedding
#             self.char_embedding = tf.get_variable("char_emb", [self.num_chars, self.char_input_dim])
#             self.char_inputs_emb = tf.nn.embedding_lookup(self.char_embedding, self.char_inputs)
            
#         with tf.variable_scope("word-based-emb"):
#             if self.embedding_matrix is not None:
#                 self.embedding = tf.Variable(self.embedding_matrix, trainable=True, name="word_emb", dtype=tf.float32)
#             else:
#                 self.embedding = tf.get_variable("word_emb", [self.num_words, self.input_dim])
#             self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.inputs)

#     def add_feature_extractor_op(self):
#         with tf.variable_scope("char_bi-lstm"):
#             self.char_inputs_emb = tf.transpose(self.char_inputs_emb, [1, 0, 2])
#             self.char_inputs_emb = tf.reshape(self.char_inputs_emb, [-1, self.char_input_dim])
#             self.char_inputs_emb = tf.split(self.char_inputs_emb, self.word_len, 0)

#             # char lstm cell
#             char_lstm_cell_fw = rnn.LSTMCell(self.char_hidden_dim)
#             char_lstm_cell_bw = rnn.LSTMCell(self.char_hidden_dim)

#             # get the length of each word
#             self.word_length = tf.reduce_sum(tf.sign(self.char_inputs), reduction_indices=1)
#             self.word_length = tf.cast(self.word_length, tf.int32)
            
#             char_outputs, f_output, r_output = tf.contrib.rnn.static_bidirectional_rnn(
#                 char_lstm_cell_fw, 
#                 char_lstm_cell_bw,
#                 self.char_inputs_emb, 
#                 dtype=tf.float32,
#                 sequence_length=self.word_length
#             )
#             final_word_output = tf.concat([f_output.h, r_output.h], -1)
#             self.word_lstm_last_output = tf.reshape(final_word_output, [-1, self.num_steps, self.char_hidden_dim*2])
            
#         with tf.variable_scope("word_bi-lstm"):
#             self.inputs_emb = tf.concat([self.inputs_emb, self.word_lstm_last_output], -1)
#             self.inputs_emb = tf.nn.dropout(self.inputs_emb, self.keep_prob)
#             self.inputs_emb = tf.transpose(self.inputs_emb, [1, 0, 2])
#             self.inputs_emb = tf.reshape(self.inputs_emb, [-1, self.input_dim+self.char_hidden_dim*2])
#             # self.inputs_emb = tf.reshape(self.inputs_emb, [-1, self.input_dim])

#             self.inputs_emb = tf.split(self.inputs_emb, self.num_steps, 0)

#             # word lstm cell
#             lstm_cell_fw = rnn.LSTMCell(self.hidden_dim)
#             lstm_cell_bw = rnn.LSTMCell(self.hidden_dim)

#             # get the length of each sample
#             self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
#             self.length = tf.cast(self.length, tf.int32) 
            
#             self.outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
#                 lstm_cell_fw, 
#                 lstm_cell_bw,
#                 self.inputs_emb, 
#                 dtype=tf.float32,
#                 sequence_length=self.length
#             )
    
#         with tf.variable_scope("bidirect-concat"):
#             final_outputs = tf.reshape(tf.concat(self.outputs, 1), [-1, self.hidden_dim * 2])
#             tanh_layer_w = tf.get_variable("tanh_layer_w", [self.hidden_dim * 2, self.hidden_dim])
#             tanh_layer_b = tf.get_variable("tanh_layer_b", [self.hidden_dim])
#             self.final_outputs = tf.tanh(tf.matmul(final_outputs, tanh_layer_w) + tanh_layer_b)

    # def forward(self):
    #     self.add_placeholders()
    #     self.add_lookup_op()
    #     self.add_feature_extractor_op()
        # return self.final_outputs, self.length
        
        
        