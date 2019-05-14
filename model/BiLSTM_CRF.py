import codecs
import os, types
import math
import utils
import numpy as np
import random
import tensorflow as tf
#from tensorflow.models.rnn import rnn, rnn_cell
import tensorflow as tf
from tensorflow.contrib import rnn
from Config import Config

from utils import reload_mappings

class BiLSTM_CRF(object):
    def __init__(self, num_steps=200, word_len =50, num_epochs=100, embedding_matrix=None, singletons=None, is_crf=True, weight=False):
        # Parameter
        self.max_f1 = 0
        self.overbest = 0
        self.config = Config()
        self.learning_rate = self.config.model_para['lr']
        self.dropout_rate = self.config.model_para['dropout_rate']
        self.batch_size = self.config.model_para['batch_size']
        self.num_layers = self.config.model_para['lstm_layer_num']
        self.input_dim = self.config.model_para['input_dim']
        self.hidden_dim = self.config.model_para['hidden_dim']
        self.char_input_dim = self.config.model_para['char_input_dim']
        self.char_hidden_dim = self.config.model_para['char_hidden_dim']
        self.num_epochs = num_epochs
        
        self.word_to_id, self.id_to_word = reload_mappings(self.config.map_dict['word2id'])
        self.char_to_id, self.id_to_char = reload_mappings(self.config.map_dict['char2id'])
        self.tag_to_id, self.id_to_tag = reload_mappings(self.config.map_dict['tag2id'])
        
        self.word_len = word_len
        self.num_steps = num_steps
        self.num_words = len(self.word_to_id)
        self.num_chars = len(self.char_to_id)
        self.num_classes = len(self.tag_to_id)
        
        self.singletons = singletons

        '''
            char input:
        '''
        # char_batch = sen_batch * sen_len
        self.char_inputs = tf.placeholder(tf.int32, [None, self.word_len])

        with tf.variable_scope("character-based-emb"):
            # char embedding
            self.char_embedding = tf.get_variable("char_emb", [self.num_chars, self.char_input_dim])

            self.char_inputs_emb = tf.nn.embedding_lookup(self.char_embedding, self.char_inputs)
            self.char_inputs_emb = tf.transpose(self.char_inputs_emb, [1, 0, 2])
            self.char_inputs_emb = tf.reshape(self.char_inputs_emb, [-1, self.char_input_dim])
            self.char_inputs_emb = tf.split(self.char_inputs_emb, self.word_len, 0)

            # char lstm cell
            char_lstm_cell_fw = rnn.LSTMCell(self.char_hidden_dim)
            char_lstm_cell_bw = rnn.LSTMCell(self.char_hidden_dim)

            # get the length of each word
            self.word_length = tf.reduce_sum(tf.sign(self.char_inputs), reduction_indices=1)
            self.word_length = tf.cast(self.word_length, tf.int32)

            # char forward and backward
            with tf.variable_scope("char_bi-lstm"):
                char_outputs, f_output, r_output = tf.contrib.rnn.static_bidirectional_rnn(
                    char_lstm_cell_fw, 
                    char_lstm_cell_bw,
                    self.char_inputs_emb, 
                    dtype=tf.float32,
                    sequence_length=self.word_length
                )
            final_word_output = tf.concat([f_output.h, r_output.h], -1)

            self.word_lstm_last_output = tf.reshape(final_word_output, [-1, self.num_steps, self.char_hidden_dim*2])
        
        '''
            word input
        '''
        self.inputs = tf.placeholder(tf.int32, [None, self.num_steps])
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
        # self.inputs_emb = tf.reshape(self.inputs_emb, [-1, self.input_dim])

        self.inputs_emb = tf.split(self.inputs_emb, self.num_steps, 0)

        # word lstm cell
        lstm_cell_fw = rnn.LSTMCell(self.hidden_dim)
        lstm_cell_bw = rnn.LSTMCell(self.hidden_dim)

        # get the length of each sample
        self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
        self.length = tf.cast(self.length, tf.int32) 
        
        # forward and backward
        with tf.variable_scope("bi-lstm"):
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
        final_outputs = tf.tanh(tf.matmul(final_outputs, tanh_layer_w) + tanh_layer_b)
        
        self.softmax_w = tf.get_variable("softmax_w", [self.hidden_dim, self.num_classes])
        self.softmax_b = tf.get_variable("softmax_b", [self.num_classes])
        self.logits = tf.matmul(final_outputs, self.softmax_w) + self.softmax_b
        
        if not is_crf:
            pass
        else:
            self.tags_scores = tf.reshape(self.logits, [self.batch_size, self.num_steps, self.num_classes])
            self.transitions = tf.get_variable("transitions", [self.num_classes + 1, self.num_classes + 1])
            dummy_val = -1000
            class_pad = tf.Variable(dummy_val * np.ones((self.batch_size, self.num_steps, 1)), dtype=tf.float32)
            self.observations = tf.concat([self.tags_scores, class_pad], 2)
            
            begin_vec = tf.Variable(np.array([[dummy_val] * self.num_classes + [0] for _ in range(self.batch_size)]), trainable=False, dtype=tf.float32)
            end_vec = tf.Variable(np.array([[0] + [dummy_val] * self.num_classes for _ in range(self.batch_size)]), trainable=False, dtype=tf.float32) 
            begin_vec = tf.reshape(begin_vec, [self.batch_size, 1, self.num_classes + 1])
            end_vec = tf.reshape(end_vec, [self.batch_size, 1, self.num_classes + 1])
            
            self.observations = tf.concat([begin_vec, self.observations, end_vec], 1)
            
            self.mask = tf.cast(tf.reshape(tf.sign(self.targets),[self.batch_size * self.num_steps]), tf.float32)
            
            # point score
            self.point_score = tf.gather(tf.reshape(self.tags_scores, [-1]), tf.range(0, self.batch_size * self.num_steps) * self.num_classes + tf.reshape(self.targets,[self.batch_size * self.num_steps]))
            self.point_score *= self.mask

            # transition score
            self.trans_score = tf.gather(tf.reshape(self.transitions, [-1]), self.targets_transition)
        
            # real score
            self.target_path_score = tf.reduce_sum(self.point_score) + tf.reduce_sum(self.trans_score)  

            # all path score
            self.total_path_score, self.max_scores, self.max_scores_pre  = self.forward(self.observations, self.transitions, self.length)
            # loss
            self.loss = - (self.target_path_score - self.total_path_score)
            
        #1 summary
        self.train_summary = tf.summary.scalar("loss", self.loss)
        self.val_summary = tf.summary.scalar("loss", self.loss)        
        
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def logsumexp(self, x, axis=None):
        x_max = tf.reduce_max(x, reduction_indices=axis, keepdims=True)
        x_max_ = tf.reduce_max(x, reduction_indices=axis)
        return x_max_ + tf.log(tf.reduce_sum(tf.exp(x - x_max), reduction_indices=axis))

    def forward(self, observations, transitions, length, is_viterbi=True, return_best_seq=True):
        length = tf.reshape(length, [self.batch_size])
        transitions = tf.reshape(tf.concat([transitions] * self.batch_size, 0), [self.batch_size, self.num_classes+1, self.num_classes+1])
        observations = tf.reshape(observations, [self.batch_size, self.num_steps + 2, self.num_classes+1, 1])
        observations = tf.transpose(observations, [1, 0, 2, 3])
        
        previous = observations[0, :, :, :]
        max_scores = []
        max_scores_pre = []
        alphas = [previous]

        for t in range(1, self.num_steps + 2):
            previous = tf.reshape(previous, [self.batch_size, self.num_classes+1, 1])
            current = tf.reshape(observations[t, :, :, :], [self.batch_size, 1, self.num_classes+1])
            alpha_t = previous + current + transitions
            if is_viterbi:
                max_scores.append(tf.reduce_max(alpha_t, reduction_indices=1))
                max_scores_pre.append(tf.argmax(alpha_t, axis=1))
            alpha_t = tf.reshape(self.logsumexp(alpha_t, axis=1), [self.batch_size, self.num_classes+1, 1])
            alphas.append(alpha_t)
            previous = alpha_t
            

        alphas = tf.reshape(tf.concat(alphas, 0), [self.num_steps + 2, self.batch_size, self.num_classes+1, 1])
        alphas = tf.transpose(alphas, [1, 0, 2, 3])
        alphas = tf.reshape(alphas, [self.batch_size * (self.num_steps + 2), self.num_classes+1, 1])

        last_alphas = tf.gather(alphas, tf.range(0, self.batch_size) * (self.num_steps + 2) + length)
        last_alphas = tf.reshape(last_alphas, [self.batch_size, self.num_classes+1, 1])

        max_scores = tf.reshape(tf.concat(max_scores, 0), (self.num_steps + 1, self.batch_size, self.num_classes+1))
        max_scores_pre = tf.reshape(tf.concat(max_scores_pre, 0), (self.num_steps + 1, self.batch_size, self.num_classes+1))
        max_scores = tf.transpose(max_scores, [1, 0, 2])
        max_scores_pre = tf.transpose(max_scores_pre, [1, 0, 2])

        return tf.reduce_sum(self.logsumexp(last_alphas, axis=1)), max_scores, max_scores_pre

    
    def getTransition(self, y_train_batch):
        transition_batch = []
        for m in range(len(y_train_batch)):
            y = [self.num_classes] + list(y_train_batch[m]) + [0]
            for t in range(len(y)):
                if t + 1 == len(y):
                    continue
                i = y[t]
                j = y[t + 1]
                #if i == 0:
                if j == 0:
                    break
                transition_batch.append(i * (self.num_classes+1) + j)
        transition_batch = np.array(transition_batch)
        return transition_batch
    

    def train(self, sess, train_data, dev_data, test_data, dev_oov_data, test_oov_data):
        merged =  tf.summary.merge_all()
        
        num_iterations = int(math.ceil(1.0 * len(train_data) / self.batch_size))
        
        for epoch in range(self.num_epochs):
            # random.shuffle(train_data)
            print("current epoch: %d" % (epoch))
            cnt = 0
            for iteration in range(num_iterations):
                X_words_train_batch, X_char_train_batch, y_train_batch = utils.next_batch(train_data, self.config.maxlen, self.word_len, self.singletons, start_index=iteration * self.batch_size, batch_size=self.batch_size)
                transition_batch = self.getTransition(y_train_batch)
                
                _, loss_train, max_scores, max_scores_pre, length, train_summary =\
                    sess.run([
                        self.optimizer,
                        self.loss,
                        self.max_scores,
                        self.max_scores_pre,
                        self.length,
                        self.train_summary
                    ],
                    feed_dict={
                        self.targets_transition:transition_batch,
                        self.char_inputs:X_char_train_batch,
                        self.inputs:X_words_train_batch,
                        self.targets:y_train_batch,
                        self.keep_prob:1-self.dropout_rate
                    })
                
                if iteration % 30 == 0:
                    cnt += 1
                    print("epoch: %d, iteration: %d,training loss: %5d" % (epoch ,iteration, loss_train))
                    dev_eval = self.test(sess, dev_data, epoch, cnt, istest = False, isoov = False)
                    test_eval = self.test(sess, test_data, epoch, cnt, istest = True, isoov = False)
                    oov_dev_eval = self.test(sess, dev_oov_data, epoch, cnt, istest = False, isoov = True)
                    oov_test_eval = self.test(sess, test_oov_data, epoch, cnt, istest = True, isoov = True)
                    
                    if self.overbest == 1:
                        self.overbest = 0
                        with open('tmp/best_score','a')as fw:
                            fw.write('epoch:  '+str(epoch)+'  '+'iteration:  '+str(cnt)+'  dev:  '+str(self.max_f1)+'  test:  '+test_eval.split("FB1:")[-1].strip()+'  dev_oov:  '+oov_dev_eval.split("FB1:")[-1].strip()+'  test_oov:  '+oov_test_eval.split("FB1:")[-1].strip()+'\n')


    def viterbi(self, max_scores, max_scores_pre, length, predict_size=128):
        best_paths = []
        for m in range(predict_size):
            path = []
            last_max_node = np.argmax(max_scores[m][length[m]])
            for t in range(1, length[m] + 1)[::-1]:
                last_max_node = max_scores_pre[m][t][last_max_node]
                path.append(last_max_node)
            path = path[::-1]
            best_paths.append(path)
        return best_paths


    
    def evaluate(self, y_true, y_pred, id2char, id2label, epoch, cnt, istest, isoov):
        precision = -1.0
        recall = -1.0
        f1 = -1.0
        
        eval_script = 'tmp/conlleval'
        output_path = 'tmp/evaluate.txt'
        scores_path = 'tmp/score.txt'
        
        with open(output_path,'w')as f:
            for i in range(len(y_true)):
                for j in range(len(y_true[i])):
                    f.write(id2label[y_true[i][j]]+' '+id2label[y_true[i][j]]+' '+id2label[y_true[i][j]]+' '+id2label[y_pred[i][j]]+'\n')
                f.write('\n')
        
        os.system("perl %s < %s > %s" % (eval_script, output_path, scores_path))
        eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
        if istest:
            score_test = 'tmp/score_test'
        else:
            score_test = 'tmp/score_dev'
        if isoov:
            score_test += '_oov'
        with open(score_test,'a')as fw:
            fw.write('epoch:  '+str(epoch)+'  '+'iteration:  '+str(cnt)+'  '+eval_lines[1]+'\n')
        
        return eval_lines[1]


    def test(self, sess, test_data, epoch, cnt, istest = False, isoov = False):
        num_iterations = int(math.ceil(1.0 * len(test_data) / self.batch_size))
        preds = []
        y_true = []
        for iteration in range(num_iterations):
            X_words_test_batch, X_char_test_batch, y_test_batch = utils.next_batch(test_data, self.config.maxlen, self.word_len, self.singletons, start_index=iteration * self.batch_size, batch_size=self.batch_size, is_train=False)
            
            length, max_scores, max_scores_pre = sess.run([self.length, self.max_scores, self.max_scores_pre], feed_dict={self.char_inputs:X_char_test_batch, self.inputs:X_words_test_batch, self.keep_prob:1})
            
            if iteration == num_iterations - 1:
                left_size = self.batch_size - (num_iterations * self.batch_size - len(test_data))                
                predicts = self.viterbi(max_scores, max_scores_pre, length, self.batch_size)        
                preds.extend(predicts[:left_size])
                y_true.extend(y_test_batch[:left_size])
            else:
                predicts = self.viterbi(max_scores, max_scores_pre, length, self.batch_size)
                preds.extend(predicts)
                y_true.extend(y_test_batch)
        
        result = self.evaluate(y_true, preds, self.id_to_word, self.id_to_tag, epoch, cnt, istest, isoov)
        
        if float(result.split("FB1:")[-1].strip()) > self.max_f1 and not istest and not isoov:
            #saver = tf.train.Saver()
            self.overbest = 1
            self.max_f1 = float(result.split("FB1:")[-1].strip())
            #save_path = saver.save(sess, self.config.modelpath, global_step = epoch)
            print("saved the best model with f1:  ", self.max_f1)
        return result
