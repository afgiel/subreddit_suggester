import json

from nltk.tokenize import word_tokenize
from nltk.util import ngrams
#from nltk.tokenize.punkt import PunktWordTokenizer
#from unidecode import unidecode
#from tokenize import untokenize


def tokenize(text, n):

  #tokenized = PunktWordTokenizer().tokenize(text.lower())
  text = text.decode('utf8')
  tokenized = word_tokenize(text.lower())

  if n == 1:
  	return tokenized

  return list(ngrams(tokenized, n))  

  #return word_tokenize(text.lower())


def write_json_file(obj, file_path):
  data = json.dumps(obj)
  with open(file_path, 'w') as json_file:
    json_file.write(data)

def load_json_file(file_path):
  with open(file_path, 'r') as json_file:
    json_str = json_file.readline()
  return json.loads(json_str)
