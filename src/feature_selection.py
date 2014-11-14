import math
from collections import Counter

import utils
import constants

# returns word_count and doc_count 
# word_count is a map from token -> counter label
# doc_count is a map from label -> counter token
# all_words is the set of all tokens seen 
def count(text_and_labels):
  word_count = dict()
  doc_count = dict()
  tokenized_docs = []
  all_words = set()
  for text, label in text_and_labels:
    tokens = utils.tokenize(text)
    tokenized_docs.append(tokens)
    for token in tokens:
      if token not in word_count:
        word_count[token] = Counter()
      word_count[token][label] += 1.
    for token in set(tokens):
      if label not in doc_count:
        doc_count[label] = Counter()
      doc_count[label][token] += 1.
      all_words.add(token)
  return word_count, doc_count, tokenized_docs, all_words


def select_top_n_mi_features(word_counts, doc_counts, all_words, m, n):
  mi = Counter()
  num_tokens = (sum([sum(word_counts[x].values()) for x in word_counts]))
  count = 0
  total = str(len(all_words))
  for word in all_words:
    count += 1
    if count % 100 == 0: print '\t' + str(count) + ' of ' + total
    mi[word] = compute_mi(word, word_counts, doc_counts, num_tokens, m)  
  return mi.most_common(n) 
  

def compute_mi(word, word_counts, doc_counts, num_tokens, m):
  mi = 0.0
  word_prob = float(sum(word_counts[word].values()))/num_tokens 
  for c in range(len(constants.subreddits)):
    pos_joint_prob = float(doc_counts[c][word])/m 
    class_prob = sum(doc_counts[c].values())
    pos_denom = class_prob*word_prob
    neg_joint_prob = 1.0 - pos_joint_prob
    neg_denom = class_prob*(1.0 - word_prob) 
    if not pos_joint_prob <= 0.0:
      mi += pos_joint_prob*math.log(pos_joint_prob/pos_denom)
    if not neg_joint_prob <= 0.0:
      mi += neg_joint_prob*math.log(neg_joint_prob/neg_denom)
  return mi


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


