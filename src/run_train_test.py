import random

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import constants
import load_subreddit_data
import reddit_post
import feature_selection
import featurizer
import evaluate

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
  print 'FEATURIZING TRAIN SET'
  train_posts = [x[0] for x in train_set]
  train_labels = [x[1] for x in train_set]
  train_x = featurizer.binary_featurize(train_posts, feature_map)
  train_y = featurizer.make_label_vector(train_labels)
  print 'TRAINING'
  naive_bayes = MultinomialNB()
  logistic_regression = LogisticRegression()
  #svm = SVC() 
  print '\tNAIVE BAYES'
  naive_bayes.fit(train_x, train_y)
  print '\tLOGISTIC REGRESSION'
  logistic_regression.fit(train_x, train_y)
  #print '\tSVM'
  #svm.fit(train_x, train_y)
  print 'FEATURIZING TEST SET'  
  test_posts = [x[0] for x in test_set]
  test_labels = [x[1] for x in test_set]
  test_x = featurizer.binary_featurize(test_posts, feature_map)
  desired_y = featurizer.make_label_vector(test_labels)
  print 'TESTING'
  nb_predicted_y = naive_bayes.predict(test_x)
  logres_predicted_y = logistic_regression.predict(test_x) 
  print 'EVALUATING'
  print '\tNAIVE BAYES'
  evaluate.evaluate(desired_y, nb_predicted_y)
  print '\tLOGISTIC REGRESSION'
  evaluate.evaluate(desired_y, logres_predicted_y)
 
  
