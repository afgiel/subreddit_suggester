import random

import numpy as np
import sklearn

import constants
import load_subreddit_data
import reddit_post
import feature_selection

def run():
  train_set = [] 
  test_set = []
  print 'LOADING DATA FROM CSVS'
  for subreddit in constants.subreddits:
    sub_all = load_subreddit_data.get_all_posts_and_labels(subreddit) 
    num_posts = len(sub_all)  
    sub_train = random.sample(sub_all, int(num_posts*.9)) 
    sub_test = [x for x in sub_all if x not in sub_train]
    train_set.extend(sub_train)
    test_set.extend(sub_test)
  print 'SELECTING FEATURES'
  feature_map = feature_selection.select_all_features(train_set)
  #train_x, train_y = featurizer.featurize(train_set, features)
