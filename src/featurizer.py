from collections import Counter
import os.path as path
import math

import numpy as np

import feature_selection
import utils
import constants
from unidecode import unidecode
from textblob import TextBlob

class Featurizer():
  
  def __init__(self, title_split, ngram, select_func, feat_func):
    self.title_split = title_split
    self.ngram = ngram
    self.select_func = select_func
    self.feat_func = feat_func

  def get_file_path(self, feat_type, num_features, fold_num):
    file_name = '_'.join([feat_type, str(fold_num), str(self.select_func.__name__), str(num_features)])
    return path.join(constants.STORAGE_PATH_FROM_SRC, 'features', file_name) 


  def choose_features(self, train_set, fold_num):
    self.train_set = train_set
    print 'COUNTING TRAIN SET'  
    if self.title_split:  
      title_train_set = [(x.title, y) for x, y in train_set]
      text_train_set = [(x.text, y) for x, y in train_set]
      title_word_counts, title_doc_counts, train_tokenized_titles, title_words = feature_selection.count(title_train_set, self.ngram)
      text_word_counts, text_doc_counts, train_tokenized_text, text_words = feature_selection.count(text_train_set, self.ngram) 
      print 'SELECTING FEATURES'
      title_feature_file_path = self.get_file_path('title', constants.NUM_TITLE_FEATURES, fold_num)
      text_feature_file_path = self.get_file_path('text', constants.NUM_TEXT_FEATURES, fold_num)
      if not path.isfile(title_feature_file_path):
        title_feature_map = self.select_func(title_words, title_word_counts, title_doc_counts, len(train_set), constants.NUM_TITLE_FEATURES)
        utils.write_json_file(title_feature_map, title_feature_file_path)
      else: 
        title_feature_map = utils.load_json_file(title_feature_file_path)
      if not path.isfile(text_feature_file_path):
        text_feature_map = self.select_func(text_words, text_word_counts, text_doc_counts, len(train_set), constants.NUM_TEXT_FEATURES)
        utils.write_json_file(text_feature_map, text_feature_file_path) 
      else: 
        text_feature_map = utils.load_json_file(text_feature_file_path)
      self.title_word_counts = title_word_counts
      self.title_doc_counts = title_doc_counts 
      self.train_tokenized_titles = train_tokenized_titles
      self.title_words = title_words
      self.title_feature_map = title_feature_map
      self.text_word_counts = text_word_counts
      self.text_doc_counts = text_doc_counts 
      self.train_tokenized_text = train_tokenized_text
      self.text_words = text_words
      self.text_feature_map = text_feature_map
    else: 
      both_train_set = [(x.title + ' ' + x.text, y) for x, y in train_set]
      both_word_counts, both_doc_counts, train_tokenized_both, both_words = feature_selection.count(both_train_set, self.ngram)
      print 'SELECTING FEATURES'
      both_file_path = self.get_file_path('both', constants.NUM_BOTH_FEATURES, fold_num) 
      if not path.isfile(both_file_path):
        both_feature_map = self.select_func(both_words, both_word_counts, both_doc_counts,  len(both_train_set), constants.NUM_BOTH_FEATURES)
        utils.write_json_file(both_feature_map, both_file_path)
      else: 
        both_feature_map = utils.load_json_file(both_file_path)
      self.both_word_counts = both_word_counts
      self.both_doc_counts = both_doc_counts 
      self.train_tokenized_both = train_tokenized_both
      self.both_words = both_words
      self.both_feature_map = both_feature_map

  def featurize_train(self): 
    train_labels = [x[1] for x in self.train_set]
    train_y = make_label_vector(train_labels)
    if self.title_split:
      train_title_x = self.feat_func(self.train_tokenized_titles, self.title_feature_map, doc_counts=self.title_doc_counts)
      train_text_x = self.feat_func(self.train_tokenized_text, self.text_feature_map, doc_counts=self.text_doc_counts)
      train_x = np.concatenate((train_title_x, train_text_x), axis=1) 
      train_y = make_label_vector(train_labels)  
      return train_x, train_y
    else:
      train_x = self.feat_func(self.train_tokenized_both, self.both_feature_map, doc_counts=self.both_doc_counts)
      return train_x, train_y


  def featurize_test(self, test_posts):
    if self.title_split:
      test_titles = [x.title for x in test_posts]
      test_text = [x.text for x in test_posts]
      test_tokenized_titles = [utils.tokenize(x, self.ngram) for x in test_titles] 
      test_tokenized_text = [utils.tokenize(x, self.ngram) for x in test_text]
      test_title_x = self.feat_func(test_tokenized_titles, self.title_feature_map, doc_counts=self.title_doc_counts)
      test_text_x = self.feat_func(test_tokenized_text, self.text_feature_map, doc_counts=self.text_doc_counts)
      test_x = np.concatenate((test_title_x, test_text_x), axis=1)
      return test_x
    else:
      test_both = [x.title + ' ' + x.text for x in test_posts]
      test_tokenized_both = [utils.tokenize(x, self.ngram) for x in test_both]
      test_x = self.feat_func(test_tokenized_both, self.both_feature_map, doc_counts=self.both_doc_counts)
      return test_x

# INTERFACE FOR FEATURIZER
#   params
#     docs 
#     feature_map
#     doc_counts 


def binary_featurize(docs, feature_map, doc_counts=None):
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

def tfidf_featurize(docs, feature_map, doc_counts=None): 
  num_train_docs = doc_counts['NUM_TRAIN_DOCS']
  n = len(feature_map) + 1
  m = len(docs)
  x = np.zeros((m, n)) 
  base_score = 0.5*math.log(float(num_train_docs)) 
  x.fill(base_score)
  for i in range(m):
    doc = docs[i]
    counter = Counter(doc)
    if len(counter) == 0: continue
    max_occur = counter.most_common(1)[0][1]
    for token in doc:
      if token in feature_map:
        j = feature_map[token]
        tf = 0.5 + float(0.5*counter[token])/max_occur 
        token_in_num_docs = sum([doc_counts[a][token] for a in doc_counts if a != 'NUM_TRAIN_DOCS']) + 1
        idf = math.log(float(num_train_docs)/token_in_num_docs)
        x[i][j] = tf*idf
  return x

def count_tfidf_featurize(docs, feature_map, doc_counts=None): 
  n = 1
  m = len(docs)
  x = np.zeros((m, n))
  for i in range(len(docs)):
    doc = docs[i]
    x[i][0] = len(doc)
  tfidf = tfidf_featurize(docs, feature_map, doc_counts)
  return np.concatenate((x, tfidf), axis=1)
  
def count_binary_featurize(docs, feature_map, doc_counts=None): 
  n = 1
  m = len(docs)
  x = np.zeros((m, n))
  for i in range(len(docs)):
    doc = docs[i]
    x[i][0] = len(doc)
  binary = binary_featurize(docs, feature_map, doc_counts)
  return np.concatenate((x, binary), axis=1)

def lda_featurize(lda, dictionary, tfidf, texts):
  m = len(texts)
  n = lda.num_topics
  x = np.zeros((m, n))
  corpus = [dictionary.doc2bow(text) for text in texts]
  corpus = [tfidf[doc] for doc in corpus]
  for i in range(len(corpus)):
    doc = corpus[i]
    gamma, _ = lda.inference([doc])
    topic_dist = gamma[0] / sum(gamma[0])
    x[i] = topic_dist
  return x

def senitment_tfidf_featurize(docs, feature_map, doc_counts=None):
  n = 1
  m = len(docs)
  x = np.zeros((m, n))
  for i in range(len(docs)):
    doc = docs[i]
    sentence = TextBlob(make_string(doc))
    sentiment_score = 0
    subjectivity_score = 0
    if sentence != "c-4": #for some reason it breaks trying to find synonyms for c-4
      sentiment_score = sentence.sentiment.polarity
      subjectivity_score = sentence.sentiment.subjectivity
    
    x[i][0] = sentiment_score #consider having both sentiment score and subjectivity score

  tfidf = tfidf_featurize(docs, feature_map, doc_counts)
  return np.concatenate((x, tfidf), axis=1)

def make_string(word_list):
  punctuation = [".", "!", "?", ",", "'", "(", ")"]
  sentence = ""
  for i in range(len(word_list)):
    word = unidecode(word_list[i])
    if word in punctuation:
      sentence = sentence + word
    elif i == 0:
      sentence = sentence + word
    else:
      sentence = sentence + " " + word

  return sentence

def make_label_vector(labels):
  m = len(labels)
  y = np.zeros(m)
  for i in range(m):
    y[i] = labels[i]
  return y

