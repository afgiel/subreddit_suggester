import random
import os.path as path

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
<<<<<<< HEAD
#from sklearn.svm import SVC
=======
from sklearn.svm import SVC
>>>>>>> changed evaluation and added svm
from sklearn.metrics import classification_report

import constants
import load_subreddit_data
import reddit_post
import feature_selection
import featurizer
<<<<<<< HEAD
import evaluate
import utils
=======
>>>>>>> changed evaluation and added svm

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
  train_posts = [x[0] for x in train_set]
  train_titles = [x.title for x in train_posts]
  train_text = [x.text for x in train_posts] 
  title_feature_map, train_tokenized_titles = feature_selection.select_all_features(train_titles) 
  text_feature_map, train_tokenized_text  = feature_selection.select_all_features(train_text) 
  print 'FEATURIZING TRAIN SET' 
  train_labels = [x[1] for x in train_set]
  train_title_x = featurizer.binary_featurize(train_tokenized_titles, title_feature_map)
  train_text_x = featurizer.binary_featurize(train_tokenized_text, text_feature_map)
  train_x = np.concatenate((train_title_x, train_text_x), axis=1) 
  train_y = featurizer.make_label_vector(train_labels)
  print 'TRAINING'
  naive_bayes = MultinomialNB()
  logistic_regression = LogisticRegression()
  svm = SVC() 
  print '\tNAIVE BAYES'
  naive_bayes.fit(train_x, train_y)
  print '\tLOGISTIC REGRESSION'
  logistic_regression.fit(train_x, train_y)
<<<<<<< HEAD
  #print '\tSVM'
  #svm.fit(train_x, train_y)
  print 'EVALUTATING TRAIN'
  print '\tNAIVE BAYES'
  nb_pred_y = naive_bayes.predict(train_x)
  print classification_report(train_y, nb_pred_y) 
  print '\tLOGISTIC REGRESSION'
  lg_pred_y = logistic_regression.predict(train_x)
  print classification_report(train_y, lg_pred_y)
=======
  print '\tSVM'
  svm.fit(train_x, train_y)
>>>>>>> changed evaluation and added svm
  print 'FEATURIZING TEST SET'  
  test_posts = [x[0] for x in test_set]
  test_titles = [x.title for x in test_posts]
  test_text = [x.text for x in test_posts]
  test_tokenized_titles = [utils.tokenize(x) for x in test_titles] 
  test_tokenized_text = [utils.tokenize(x) for x in test_text]
  test_labels = [x[1] for x in test_set]
  test_title_x = featurizer.binary_featurize(test_tokenized_titles, title_feature_map)
  test_text_x = featurizer.binary_featurize(test_tokenized_text, text_feature_map)
  test_x = np.concatenate((test_title_x, test_text_x), axis=1)
  desired_y = featurizer.make_label_vector(test_labels)
  print 'TESTING'
  nb_predicted_y = naive_bayes.predict(test_x)
  logres_predicted_y = logistic_regression.predict(test_x) 
<<<<<<< HEAD
  #svm_predicted_y = svm.predict(test_x)
=======
  svm_predicted_y = svm.predict(test_x)
>>>>>>> changed evaluation and added svm
  print 'EVALUATING'
  print '\tNAIVE BAYES'
  print classification_report(desired_y, nb_predicted_y)
  print '\tLOGISTIC REGRESSION'
  print classification_report(desired_y, logres_predicted_y)
<<<<<<< HEAD
  #print '\tSVM' 
  #print classification_report(desired_y, svm_predicted_y)
  
=======
  print '\tSVM'
  print classification_report(desired_y, svm_predicted_y)
>>>>>>> changed evaluation and added svm
