#!/usr/bin/env python

from __future__ import print_function, division
#import example_helper
import json
import numpy as np
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_emojis
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
#from sys import argv 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#SENTENCE = argv 
SENTENCE = ['']
#SENTENCE = input("Enter the sentence")

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

maxlen = 30
batch_size = 32


def get_senti(SENTENCE):
#print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)
    st = SentenceTokenizer(vocabulary, maxlen)
    tokenized, _, _ = st.tokenize_sentences(SENTENCE)
    
    #print('Loading model from {}.'.format(PRETRAINED_PATH))
    model = deepmoji_emojis(maxlen, PRETRAINED_PATH)
    emotion = ['Tears of Joy', 'Unamused', 'Crying', 'Loudly Crying', 'Smiling with Heart Eyes',
               'Pensive','Ok','Smiling Face','Heart','Smirking', 'Grimacing','Music','Flushed',
               '100','Sleeping','Grinning','Smiling','Raising Hands','Hearts','Expressionless',
               'Grinning with Sweat','Folded Hands','Smirking','Blowing a Kiss','Red Heart',
               'Neutral','Tipping Hand','Pensive','Closing Eyes','Persevering','Peace','Cool',
               'Very angry','Thumbs up','Crying','Sleepy', 'Showing Tounge','Steam from Nose',
               'Raised Hand','Medical Mask','Clap','Eyes','Gun','Weary','Evil Smile','Downcast',
               'Heart Broken','Empty Heart','Music with Headphones','Closing Mouth','Winking',
               'Skull','Confounded','Laughing','Winking with Tounge','Angry','No Gesture',
               'Muscles','Fist','Violet Heart','Heart with Shines','Blue Heart','Grimacing','Sparks']
    #model.summary()
    
    #print('Running predictions.')
    prob = model.predict(tokenized)
    
    #print('Writing results to {}'.format(OUTPUT_PATH))
    scores = []
    for i, t in enumerate(SENTENCE):
        t_score = [t]
        t_prob = prob[i]
        ind_top = top_elements(t_prob, 5)
        t_score.append(sum(t_prob[ind_top]))
        t_score.extend(ind_top)
        t_score.extend([t_prob[ind] for ind in ind_top])
        scores.append(t_score)
        for j in list(range(2,7)):
            Senti.append(emotion[scores[0][j]])
    return Senti

Sentiment = get_senti(SENTENCE=['Have a nice day'])

