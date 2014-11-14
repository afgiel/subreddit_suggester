from collections import Counter
import math

import numpy as np

import utils

def binary_featurize(docs, feature_map):
  n = len(feature_map) + 1
  m = len(docs) 
  x = np.zeros((m, n))
  for i in range(len(docs)):
    if i%100 == 0: print i, ' of ', m
    doc = docs[i]
    for token in doc:
      if token in feature_map:
        j = feature_map[token]
        x[i][j] = 1.
  return x  

def tfidf_featurize(docs, feature_map, doc_counts): 
  n = len(feature_map) + 1
  m = len(docs)
  x = np.zeros((m, n))
  for i in range(len(docs)):
    doc = docs[i]
    counter = Counter(doc)
    max_occur = counter.most_common(1)[1]
    for token in doc:
      if token in feature_map:
        j = feature_map[token]
        tf = 0.5 + float(0.5*counter[token])/max_occur 
        token_in_num_docs = sum([doc_counts[x][token] for x in doc_counts])
        idf = math.log(float(m)/token_in_num_docs)
        x[i][j] = tf*idf
  return x


def lda_featurize(lda, dictionary, tfidf, texts):
    m = len(texts)
    n = lda.num_topics
    x = np.zeros((m, n))
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpus = [tfidf[doc] for doc in corpus]
    for i in range(len(corpus)):
      doc = corpus[i]
      topic_vec = lda[doc]
      for index, val in topic_vec:
        x[i][index] = val
    return x


def make_label_vector(labels):
  m = len(labels)
  y = np.zeros(m)
  for i in range(m):
    y[i] = labels[i]
  return y
