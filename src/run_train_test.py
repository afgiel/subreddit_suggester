import random
import os.path as path

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
from sklearn.metrics import classification_report

import constants
import load_subreddit_data
import reddit_post
import feature_selection
import featurizer


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
  print 'COUNTING TRAIN SET' 
  title_train_set = [(x.title, y) for x, y in train_set]
  text_train_set = [(x.text, y) for x, y in train_set]
  title_word_counts, title_doc_counts, train_tokenized_titles, title_words = feature_selection.count(title_train_set)
  text_word_counts, text_doc_counts, train_tokenized_text, text_words = feature_selection.count(text_train_set) 
  print 'SELECTING FEATURES'
  title_feature_file_path = path.join(constants.STORAGE_PATH_FROM_SRC, 'features/title_mi_' + str(constants.NUM_TITLE_FEATURES)) 
  text_feature_file_path = path.join(constants.STORAGE_PATH_FROM_SRC, 'features/text_mi_' + str(constants.NUM_TEXT_FEATURES)) 
  if not path.isfile(title_feature_file_path):
    title_feature_map = feature_selection.select_top_n_mi_features(title_word_counts, title_doc_counts, title_words, len(train_set), constants.NUM_TITLE_FEATURES)
    utils.write_json_file(title_feature_map, title_feature_file_path)
  else: 
    title_feature_map = utils.load_json_file(title_feature_file_path)
  if not path.isfile(text_feature_file_path):
    text_feature_map = feature_selection.select_top_n_mi_features(text_word_counts, text_doc_counts, text_words, len(train_set), constants.NUM_TEXT_FEATURES)
    utils.write_json_file(text_feature_map, text_feature_file_path) 
  else: 
    text_feature_map = utils.load_json_file(text_feature_file_path)
  print 'FEATURIZING TRAIN SET' 
  train_labels = [x[1] for x in train_set]
  train_title_x = featurizer.tfidf_featurize(train_tokenized_titles, title_feature_map, title_doc_counts)
  train_text_x = featurizer.tfidf_featurize(train_tokenized_text, text_feature_map, text_doc_counts)
  train_x = np.concatenate((train_title_x, train_text_x), axis=1) 
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
  print 'EVALUATING TRAIN'
  nb_predicted_y = naive_bayes.predict(train_x)
  logres_predicted_y = logistic_regression.predict(train_x) 
  print '\tNAIVE BAYES'
  print classification_report(train_y, nb_predicted_y)
  print '\tLOGISTIC REGRESSION'
  print classification_report(train_y, logres_predicted_y)
  print 'FEATURIZING TEST SET'  
  test_posts = [x[0] for x in test_set]
  test_titles = [x.title for x in test_posts]
  test_text = [x.text for x in test_posts]
  test_tokenized_titles = [utils.tokenize(x) for x in test_titles] 
  test_tokenized_text = [utils.tokenize(x) for x in test_text]
  test_labels = [x[1] for x in test_set]
  test_title_x = featurizer.tfidf_featurize(test_tokenized_titles, title_feature_map, title_doc_counts)
  test_text_x = featurizer.tfidf_featurize(test_tokenized_text, text_feature_map, text_doc_counts)
  test_x = np.concatenate((test_title_x, test_text_x), axis=1)
  desired_y = featurizer.make_label_vector(test_labels)
  print 'TESTING'
  nb_predicted_y = naive_bayes.predict(test_x)
  logres_predicted_y = logistic_regression.predict(test_x) 
  #svm_predicted_y = svm.predict(test_x)
  print 'EVALUATING TEST'
  print '\tNAIVE BAYES'
  print classification_report(desired_y, nb_predicted_y)
  print '\tLOGISTIC REGRESSION'
  print classification_report(desired_y, logres_predicted_y)
  #print '\tSVM' 
  #print classification_report(desired_y, svm_predicted_y)
  
