import string
import re
import json

from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

def tokenize(text):
  return word_tokenize(text.lower())
    #char_from = '[' + string.punctuation + string.whitespace + ']'
    #char_to = ' '
    #words = re.sub(char_from, char_to, text)
    #words = words.lower().split()
    #tokens = []
    #excluded = set([w.encode('utf-8') for w in stopwords.words('english')])
    #stemmer = PorterStemmer()
    #excluded_re = '\\\\|[0-9]|\.[a-z]+'
    #for word in words:
    #  if word not in excluded and re.match(excluded_re, word) is None:
    #    tokens.append(stemmer.stem(word))
    #bigrams = []
    #last_token = 'START'
    #separator = '_'
    #for token in tokens:
    #  bigram = separator.join([last_token, token])
    #  bigrams.append(bigram)
    #  last_token = token
    #tokens.extend(bigrams)
    #return tokens

def write_json_file(obj, file_path):
  with open(file_path, 'w') as json_file:
    json_file.write(json.dumps(obj))

def load_json_file(file_path):
  with open(file_path, 'r') as json_file:
    json_str = json_file.readline()
    return json.loads(json_str)
