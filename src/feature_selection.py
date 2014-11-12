from collections import Counter

import utils

def count(posts_and_labels):
  word_count = dict()
  doc_count = dict()
  for post, label in posts_and_labels:
    title = post.title
    tokens = utils.tokenize(title)
    for token in tokens:
      if label not in word_count:
        word_count[label] = Counter()
      word_count[label][token] += 1.
    for token in set(tokens):
      if label not in doc_count:
        doc_count[label] = Counter()
       doc_count[label] += 1.
  return word_count, doc_count


def select_top_n_mi_features()
      
def select_all_features(posts_and_labels):
  token_to_index = dict()
  index = 0
  for post, label in posts_and_labels:
    title = post.title
    text = post.text
    title_tokens = utils.tokenize(title)
    text_tokens = utils.tokenize(text)
    tokens = set(title_tokens).union(set(text_tokens)) 
    for token in tokens:
      if not token in token_to_index:
        index += 1
        token_to_index[token] = index
  return token_to_index


