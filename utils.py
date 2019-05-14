# -*- coding: UTF-8 -*-
import re
import os
import types
import csv
import time
import pickle
import numpy as np
import pandas as pd
import random
from Config import Config


np.random.seed(1337)
config=Config()


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def loadMap(token2id_filepath):
    if not os.path.isfile(token2id_filepath):
        print("file not exist, building map")
        buildMap()
    token2id = {}
    id2token = {}
    with open(token2id_filepath) as infile:
        for row in infile:
            row = row.rstrip()
            token = row.split('\t')[0]
            token_id = int(row.split('\t')[1])
            token2id[token] = token_id
            id2token[token_id] = token
    return token2id, id2token


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def save_mappings(config, id_to_word, id_to_char, id_to_tag):
    """
    We need to save the mappings if we want to use the model later.
    """
    with open(config.map_dict['word2id'], 'w') as f:
        for idx in id_to_word:
            f.write(id_to_word[idx] + "\t" + str(idx)  + "\r\n")
                
    with open(config.map_dict['char2id'], 'w') as f:
        for idx in id_to_char:
            f.write(id_to_char[idx] + "\t" + str(idx)  + "\r\n")

    with open(config.map_dict['tag2id'], 'w') as f:
        for idx in id_to_tag:
            f.write(id_to_tag[idx] + "\t" + str(idx)  + "\r\n")

                
def reload_mappings(path):
    """
    Load mappings from disk.
    """
    token_to_id = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            # token = line.rstrip().split()
            token = line.rstrip().split('\t')
            token_to_id[token[0]] = int(token[1])
    id_to_token = {value:key for key,value in token_to_id.items()}
    
    return token_to_id, id_to_token
            

def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(1)
        else:
            new_words.append(word)
    return new_words


def pad_word_chars(words, word_len, maxlen):
    """
    Pad the characters of the words in a sentence.
    Input:
        - list of lists of ints (list of words, a word being a list of char indexes)
    Output:
        - padded list of lists of ints
        - list of ints corresponding to the index of the last character of each word
    """
    max_length = word_len
    char = []
    for word in words:
        padding = [0] * (max_length - len(word))
        char.append(word + padding)
    tmp = [0] * word_len
    for i in range(maxlen - len(words)):
        char.append(tmp)
    return char


def padding(sample, maxlen):
    return sample + [0 for _ in range(maxlen - len(sample))]
    

def create_input(data, maxlen, word_len, add_label, singletons=None, is_train=True):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    words = padding(data['words'], maxlen)
    chars = data['chars']
    caps = data['caps']
    if singletons is not None:
        words = insert_singletons(words, singletons)
    # if parameters['cap_dim']:
    #     caps = data['caps']
    char = pad_word_chars(chars, word_len, maxlen)
    input = []
    input.append(words)
    input.append(char)
    input.append(caps)
    if add_label:
        if is_train:
            input.append(padding(data['tags'], maxlen))               
        else:
            input.append(data['tags'])
    return input


def next_batch(data, maxlen, word_len, singletons, start_index, batch_size=128, is_train = True):
    last_index = start_index + batch_size
    batch_data = list(data[start_index:min(last_index, len(data))])
    if last_index > len(data):
        left_size = last_index - (len(data))
        for i in range(left_size):
            index = np.random.randint(len(data))
            batch_data.append(data[index])
    # create input for each sentence
    X_words_batch = []
    X_char_batch = []
    y_batch = []
    
    for sample in batch_data:
        sample_input = create_input(sample, maxlen, word_len, True, singletons, is_train)
        X_words_batch.append(sample_input[0])
        X_char_batch.append(sample_input[1])
        y_batch.append(sample_input[-1])
        
    X_words_batch = np.array(X_words_batch)
    X_chars = []
    for sample in X_char_batch:
        X_chars.extend(sample)
    X_char_batch = np.array(X_chars)
    y_batch = np.array(y_batch)
    return X_words_batch, X_char_batch, y_batch


'''
    2019/4/23: debug prepare pa_targets input
'''
def prepare_pa_targets(sample, num_classes):
    pa_targets = []
    
    padding_hot = [0 for i in range(num_classes+1)]
    padding_hot[0] = 1
    head_hot = [0 for i in range(num_classes+1)]
    head_hot[num_classes] = 1
    
    pa_targets.append(head_hot)
    for i in range(len(sample)):
        tmp = [0 for i in range(num_classes+1)]
        tmp[sample[i]] = 1
        pa_targets.append(tmp)
    pa_targets.append(padding_hot)
    return pa_targets


def next_pa_batch(data, num_classes, maxlen, word_len, singletons, start_index, batch_size=128, is_train = True):
    last_index = start_index + batch_size
    batch_data = list(data[start_index:min(last_index, len(data))])
    if last_index > len(data):
        left_size = last_index - (len(data))
        for i in range(left_size):
            index = np.random.randint(len(data))
            batch_data.append(data[index])
    # create input for each sentence
    X_words_batch = []
    X_char_batch = []
    y_batch = []
    
    for sample in batch_data:
        sample_input = create_input(sample, maxlen, word_len, True, singletons, is_train)
        X_words_batch.append(sample_input[0])
        X_char_batch.append(sample_input[1])
        y_batch.append(prepare_pa_targets(sample_input[-1], num_classes))
        
    X_words_batch = np.array(X_words_batch)
    X_chars = []
    for sample in X_char_batch:
        X_chars.extend(sample)
    X_char_batch = np.array(X_chars)
    y_batch = np.array(y_batch)
    return X_words_batch, X_char_batch, y_batch
    
    