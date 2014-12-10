import json

from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
#from nltk.tokenize.punkt import PunktWordTokenizer
#from unidecode import unidecode
#from tokenize import untokenize

stemmer = PorterStemmer()
stop_words = stopwords.words("english")


def tokenize(text, n, no_stop_words, stem):

  #tokenized = PunktWordTokenizer().tokenize(text.lower())
  text = text.decode('utf8')
  tokenized = word_tokenize(text.lower())

  # Perform stemming or remove stopwords
  if no_stop_words or stem:
    new_tokenized = []
    for word in tokenized:
      if no_stop_words and stem:
        if word not in stop_words:
          new_tokenized.append(stemmer.stem_word(word))
      elif no_stop_words:
        if word not in stop_words:
          new_tokenized.append(word)
      else:
        new_tokenized.append(stemmer.stem_word(word))
    tokenized = new_tokenized

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

def write_vectors_to_file(file_path, vectors):
  with open(file_path, 'w') as vector_file:
    vector_file.write(vectors)
    
