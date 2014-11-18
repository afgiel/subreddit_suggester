import json

from nltk.tokenize import word_tokenize

def tokenize(text):
    return word_tokenize(text)
 
def write_json_file(obj, file_path):
  with open(file_path, 'w') as json_file:
    json_file.write(json.dumps(obj))

def load_json_file(file_path):
  with open(file_path, 'r') as json_file:
    json_str = json_file.readline()
    return json.loads(json_str)
