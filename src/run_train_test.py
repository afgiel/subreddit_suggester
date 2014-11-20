import random
import os.path as path

import numpy as np
from sklearn.metrics import classification_report

import constants
import load_subreddit_data
import reddit_post
import feature_selection
import featurizer


def run(title_split, ngram, select_func, feat_func, model, train_set, test_set):
  f = featurizer.Featurizer(title_split, ngram, select_func, feat_func) 
  f.choose_features(train_set)
  print 'FEATURIZING TRAIN SET' 
  train_x, train_y = f.featurize_train()
  print 'TRAINING'
  m = model()
  m.fit(train_x, train_y)
  print 'EVALUATING TRAIN'
  pred_y = m.predict(train_x)
  print classification_report(train_y, pred_y)
  print 'FEATURIZING TEST SET'  
  test_posts = [x[0] for x in test_set]
  test_labels = [x[1] for x in test_set]
  des_y = featurizer.make_label_vector(test_labels)
  test_x = f.featurize_test(test_posts)
  print 'TESTING'
  pred_y = m.predict(test_x)
  print 'EVALUATING TEST' 
  print classification_report(des_y, pred_y)
