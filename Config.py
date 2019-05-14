# -*- coding: utf-8 -*-

class Config():
    def __init__(self):
        self.dataset = {
            'traindata': r'data/conll2003/train2.txt',
            'devdata': r'data/conll2003/valid.txt',
            'testdata': r'data/conll2003/test.txt',
            # 'traindata': r'data/ec_pa/train',
            # 'traindata': r'data/ec_pa/ds_pa',
            # 'devdata': r'data/ec_pa/dev',
            # 'testdata': r'data/ec_pa/test',
        }
        self.dev_oov = r'data/conll2003/valid_oov.txt'
        self.test_oov = r'data/conll2003/test_oov.txt'
        # self.dev_oov = r'data/ec_pa/dev'
        # self.test_oov = r'data/ec_pa/test'
        self.map_dict = {
            'word2id': r'resource/mapping/word2id',
            'char2id': r'resource/mapping/char2id',
            'tag2id': r'resource/mapping/tag2id',
        }
        self.maxlen = 125
        self.modelpath = r''
        self.modeldir = r''
        self.zeros = 1
        self.lower = 0 
        self.model_para = {
            'lr': 0.002,
            'dropout_rate': 0.5,
            'batch_size': 32,
            'lstm_layer_num': 1,
            'input_dim': 100,
            'char_input_dim': 25,
            'hidden_dim': 100,
            'char_hidden_dim': 25,
            'emb_path': None,
            'average_batch_loss': 0,
            'use_crf': True,
            'use_pa_learning': True,
        }
        self.epochs = 1

