from collections import Counter

import utils

def select_all_features(posts_and_labels):
  token_to_index = dict()
  index = 0
  for post, label in posts_and_labels:
    title = post.title
    self_text = post.self_text
    title_tokens = utils.tokenize(title)
    text_tokens = utils.tokenize(self_text)
    tokens = set(title_tokens).union(set(text_tokens)) 
    for token in tokens:
      if not token in token_to_index:
        index += 1
        token_to_index[token] = index
  return token_to_index
