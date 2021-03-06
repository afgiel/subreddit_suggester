import random
import os.path as path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

import constants
import load_subreddit_data
import reddit_post
import feature_selection
import featurizer


def run(fold_num, title_split, ngram, select_func, feat_func, model, train_set, test_set, no_stop_words, stem, pca, num_both_features, num_title_features, num_text_features):
  f = featurizer.Featurizer(title_split, ngram, select_func, feat_func, no_stop_words, stem, pca, num_both_features, num_title_features, num_text_features) 
  f.choose_features(train_set, fold_num)
  print 'FEATURIZING TRAIN SET: FOLD ' + str(fold_num)  
  train_x, train_y = f.featurize_train()
  print 'TRAINING: FOLD ' + str(fold_num)
  model.fit(train_x, train_y)
  print 'EVALUATING TRAIN: FOLD ' + str(fold_num)  
  pred_y = model.predict(train_x)
  print classification_report(train_y, pred_y)
  print 'FEATURIZING TEST SET: FOLD ' + str(fold_num)   
  test_posts = [x[0] for x in test_set]
  test_labels = [x[1] for x in test_set]
  des_y = f.make_label_vector(test_labels)
  test_x = f.featurize_test(test_posts)
  print 'TESTING: FOLD ' + str(fold_num)  
  pred_y = model.predict(test_x)
  print 'EVALUATING TEST: FOLD ' + str(fold_num)   
  print classification_report(des_y, pred_y)
  cm = confusion_matrix(des_y, pred_y)
  plt.matshow(cm) 
  plt.colorbar()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()
  return des_y, pred_y
