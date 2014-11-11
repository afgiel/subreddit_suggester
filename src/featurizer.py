import numpy as np

import utils

def binary_featurize(posts, feature_map):
  n = len(feature_map) + 1
  m = len(posts)
  x = np.zeros((m, n))
  for i in range(m):
    post = posts[i]
    title = post.title
    self_text = post.self_text
    title_tokens = utils.tokenize(title)
    text_tokens = utils.tokenize(self_text)
    tokens = set(title_tokens).union(set(text_tokens)) 
    for token in tokens:
      if token in feature_map:
        j = feature_map[token]
        x[i][j] = 1.
  return x  

def make_label_vector(labels):
  m = len(labels)
  y = np.zeros(m)
  for i in range(m):
    y[i] = labels[i]
  return y
