import codecs
import os, types
import math
import utils
import numpy as np
import random
import tensorflow as tf
#from tensorflow.models.rnn import rnn, rnn_cell
import tensorflow as tf
from inference.crf import CRF
from .charwordbilstm import CharWordBiLSTM
from tensorflow.contrib import rnn
from Config import Config

from utils import reload_mappings

class LSTM_CRF(object):
    def __init__(self, num_steps=200, word_len =50, num_epochs=100, embedding_matrix=None, singletons=None, is_crf=True, weight=False):
        # Parameter
        self.max_f1 = 0
        self.overbest = 0
        self.config = Config()
        self.learning_rate = self.config.model_para['lr']
        self.dropout_rate = self.config.model_para['dropout_rate']
        self.batch_size = self.config.model_para['batch_size']
        self.use_crf = self.config.model_para['use_crf']
        self.use_pa_learning = self.config.model_para['use_pa_learning']
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
        
        # build model gragh
        self.graph = CharWordBiLSTM(self.num_words, self.num_chars, self.num_classes, self.num_steps, self.word_len, embedding_matrix)
        # self.graph.forward()

        if self.use_crf:
            self.crf = CRF(self.batch_size, self.num_classes, self.num_steps, self.graph.final_outputs)
            if self.use_pa_learning:
                self.loss = self.crf.neg_log_likelihood_pa_loss(self.graph.targets, self.graph.length)    
            else:
                self.loss = self.crf.neg_log_likelihood_loss(self.graph.targets, self.graph.targets_transition, self.graph.length)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    
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
                if self.use_pa_learning:
                    X_words_train_batch, X_char_train_batch, y_train_batch = utils.next_pa_batch(train_data, self.num_classes, self.config.maxlen, self.word_len, self.singletons, start_index=iteration * self.batch_size, batch_size=self.batch_size)
                else:
                    X_words_train_batch, X_char_train_batch, y_train_batch = utils.next_batch(train_data, self.config.maxlen, self.word_len, self.singletons, start_index=iteration * self.batch_size, batch_size=self.batch_size)
                    transition_batch = self.getTransition(y_train_batch)
                
                # _, loss_train, max_scores, max_scores_pre, length, target_path_score, total_path_score  =\
                # _, loss_train, max_scores, max_scores_pre, length, PA_path_score, total_path_score =\
                _, loss_train, max_scores, max_scores_pre, length =\
                    sess.run([
                        self.optimizer,
                        self.loss,
                        self.crf.max_scores,
                        self.crf.max_scores_pre,
                        self.graph.length,
                        # self.crf.PA_path_score,
                        # self.crf.target_path_score,
                        # self.crf.total_path_score
                    ],
                    feed_dict={
                        # self.graph.targets_transition:transition_batch,
                        self.graph.char_inputs:X_char_train_batch,
                        self.graph.inputs:X_words_train_batch,
                        self.graph.targets:y_train_batch,
                        self.graph.keep_prob:1-self.dropout_rate
                    })
                # print(str(target_path_score), str(total_path_score))
                # print(str(PA_path_score), str(total_path_score))
                
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
            
            length, max_scores, max_scores_pre = sess.run([self.graph.length, self.crf.max_scores, self.crf.max_scores_pre], feed_dict={self.graph.char_inputs:X_char_test_batch, self.graph.inputs:X_words_test_batch, self.graph.keep_prob:1})
            
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
