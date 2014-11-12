import string
import re

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

def tokenize(text):
    char_from = '[' + string.punctuation + string.whitespace + ']'
    char_to = ' '
    words = re.sub(char_from, char_to, text)
    words = words.lower().split()
    tokens = []
    excluded = set([w.encode('utf-8') for w in stopwords.words('english')])
    stemmer = PorterStemmer()
    excluded_re = '\\\\|[0-9]|\.[a-z]+'
    for word in words:
      if word not in excluded and re.match(excluded_re, word) is None:
        tokens.append(stemmer.stem(word))
    bigrams = []
    last_token = 'START'
    separator = '_'
    for token in tokens:
      bigram = separator.join([last_token, token])
      bigrams.append(bigram)
      last_token = token
    tokens.extend(bigrams)
    return tokens
