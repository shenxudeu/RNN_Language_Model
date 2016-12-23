'''
Preprocess the reddit comments dataset
1. read the dataset and parse into setences.
2. convert word to tokens.
3. build training and testing dense data.

Derived from Denny Britz's NLP code https://github.com/dennybritz/rnn-tutorial-rnnlm.
'''

import csv
import itertools
import numpy as np
import nltk
import sys
import os
from IPython import embed
import cPickle

#DATA_FN = os.path.join('data','reddit-comments-2015-08.csv')
DATA_FN = os.path.join('data','debug_comments.csv')
sentence_stat_token = 'SENTENCE_START'
sentence_end_token = 'SENTENCE_END'
unknown_token = 'UNKNOWN_TOKEN'

def get_data(vocabulary_size = 8000):
    with open(DATA_FN,'rb') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        # Split comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        # Append SENTENCE_START and SENTENCE_END into each sentence
        sentences = ['%s %s %s'%(sentence_stat_token, sent, sentence_end_token) for sent in sentences]
    
    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences] # list of list

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print 'Found %d uinque words tokens.'%len(word_freq.items())
       
    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i, w in enumerate(index_to_word)])

    print 'Using vocabulary size %d.'%vocabulary_size
    print 'The least frequent word in our vocabulary is \'%s\' and appeared %d times.'%(vocab[-1][0], vocab[-1][1])
    
    # Replace all words not in our vocabulary with the unknown token
    for (i, sent) in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    # Create training data
    train_x = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    train_y = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    return train_x, train_y, vocab, index_to_word, word_to_index

if __name__ == '__main__':
    vocabulary_size=8000
    (train_x, train_y, vocab, index_to_word, word_to_index) = get_data(vocabulary_size)
    with open(os.path.join('data','processed_reddit_comments_%s_vocab_size.plk'), 'w') as f:
        cPickle.dump([train_x, train_y, vocab, index_to_word, word_to_index, vocabulary_size], f)
    embed()
    HERE
