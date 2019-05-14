# -*- coding: UTF-8 -*-
import re
import os
import codecs
import numpy as np
from Config import Config

from utils import create_dico, create_mapping, zero_digits
np.random.seed(1337)

def get_lample_embedding(emb_file, id_to_word, word_dim):
    pretrained = {}
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_file, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pretrained[line[0]] = np.array(
                [float(x) for x in line[1:]]
            ).astype(np.float32)
        else:
            emb_invalid += 1
            
    print('Loaded %i pretrained embeddings.' % len(pretrained))
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)
    
    new_weights = np.zeros((len(id_to_word), word_dim))
    c_found = 0
    c_lower = 0
    c_zeros = 0
    for i in range(len(id_to_word)):
        word = id_to_word[i]
        if word in pretrained:
            new_weights[i] = pretrained[word]
            c_found += 1
        elif word.lower() in pretrained:
            new_weights[i] = pretrained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pretrained:
            new_weights[i] = pretrained[
                re.sub('\d', '0', word.lower())
            ]
            c_zeros += 1

    print(('%i / %i (%.4f%%) words have been initialized with '
            'pretrained embeddings.') % (
                c_found + c_lower + c_zeros, len(id_to_word),
                100. * (c_found + c_lower + c_zeros) / len(id_to_word)
        ))
    print(('%i found directly, %i after lowercasing, '
            '%i after lowercasing + zero.') % (
                c_found, c_lower, c_zeros
        ))
    # os._exit(0)
    return new_weights


def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split('\t')
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    dico['<PAD>'] = 10000001
    word_to_id, id_to_word = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word_to_id, id_to_word


def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    dico['<UNK>'] = 10000000
    dico['<PAD>'] = 10000001
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique characters" % len(dico))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    # dico['<UNK>'] = 10000000
    dico['<PAD>'] = 10000000
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3


def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, lower=False):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    def f(x): return x.lower() if lower else x
    data = []
    for s in sentences:
        str_words = [w[0] for w in s]
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        # Skip characters that are not in the training set
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]
        caps = [cap_feature(w) for w in str_words]
        tags = [tag_to_id[w[-1]] for w in s]
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'caps': caps,
            'tags': tags,
        })
    return data


def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])
    print('old dictionary size:', len(dictionary))
    print('len of pretrained emb', len(pretrained))
    print('len of words/dev+test words', len(words))

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0
    
    print('new dictionary size:', len(dictionary))

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word