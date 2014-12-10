

import random
import os.path as path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

import constants
import load_subreddit_data
import reddit_post
import feature_selection
import featurizer
import reddit_post


title_split = True
ngram = 1
select_func = feature_selection.select_top_n_mi_features
feat_func = featurizer.Featurizer.count_binary_featurize 
model = LogisticRegression
C = .25
fold_num = 'all'

train_set, test_set =  load_subreddit_data.get_train_and_test_sets()
train_set = train_set + test_set
f = featurizer.Featurizer(title_split, ngram, select_func, feat_func) 
f.choose_features(train_set, fold_num)
print 'FEATURIZING TRAIN SET: FOLD ' + str(fold_num)  
train_x, train_y = f.featurize_train()
print 'TRAINING: FOLD ' + str(fold_num)
m = model(C=C)
m.fit(train_x, train_y)
print 'EVALUATING TRAIN: FOLD ' + str(fold_num)  
pred_y = m.predict(train_x)
print classification_report(train_y, pred_y)
while (True):
  title = raw_input('Enter the title of your post:')
  text = raw_input('Enter the text of your post:')
  post = reddit_post.RedditPost(title, text)
  test_x = f.featurize_test([post])
  pred_y = m.predict(test_x)
  print pred_y
