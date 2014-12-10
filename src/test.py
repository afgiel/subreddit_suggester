import run_train_test
from featurizer import Featurizer
import feature_selection
import kfold
import load_subreddit_data
import constants

import argparse

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report

FEAT_FUNCS = {
  'tfidf': Featurizer.tfidf_featurize, 
  'binary': Featurizer.binary_featurize,
  'count_tfidf': Featurizer.count_tfidf_featurize,
  'count_binary': Featurizer.count_binary_featurize,
  'sentiment_tfidf': Featurizer.sentiment_tfidf_featurize,
  'pos_binary': Featurizer.count_binary_pos_featurize,
  'sentiment_binary': Featurizer.sentiment_binary_featurize,
  'lda': Featurizer.lda_featurize,
  'lda_binary': Featurizer.lda_binary_featurize,
  'lda_tfidf': Featurizer.lda_tfidf_featurize
}

SELECT_FUNCS = {
  'mi': feature_selection.select_top_n_mi_features, 
  'all': feature_selection.select_all_features
}

MODELS = {
  'logistic_regression': LogisticRegression,
  'naive_bayes': MultinomialNB,
  'svm': SVC
}

# Returns tuple of parsed command-line arguments.  Tuple contains functions
# so the conversion from string to corresponding function is also done in
# this method.
def get_args():
  parser = argparse.ArgumentParser(description="Test different learning models, feature representations" +
                                                "and parameters on the reddit datasets.")
  parser.add_argument('-model', '-m', choices=MODELS.keys(), default='logistic_regression', type=str)
  parser.add_argument('-featureSelector', '-s', choices=SELECT_FUNCS.keys(), default='mi', type=str)
  parser.add_argument('-featureRepresentation', '-f',choices=FEAT_FUNCS.keys(), default='tfidf', type=str)
  parser.add_argument('-kfolds', '-k', default = 0, type=int)
  parser.add_argument('-C', default = 1.0, type=float)
  parser.add_argument('-ngram', '-n', default = 1, type=int)
  parser.add_argument('--titleSplit', '--t', action = 'store_true', default=False)
  parser.add_argument('--noStopWords', action = 'store_true', default=False)
  parser.add_argument('--stem',  action = 'store_true', default=False)
  args = parser.parse_args()

  C = args.C
  if args.model == 'naive_bayes':
    model = MODELS[args.model]()
  else:
    model = MODELS[args.model](C=C)
  feature_sel = SELECT_FUNCS[args.featureSelector]
  feature_rep = FEAT_FUNCS[args.featureRepresentation]
  kfolds = args.kfolds
  ngram = args.ngram
  title_split = args.titleSplit
  no_stop_words = args.noStopWords
  stem = args.stem
  print "THE SETTINGS:"
  print "----------------"
  print "Model:", args.model
  print "Feature Selector:", args.featureSelector
  print "Feature Representation:", args.featureRepresentation
  if args.model == 'naive_bayes':
    if C != 1.0:
      print "WARNING: Disregarding regularization parameter C - naive bayes can not be regularized"
  else:
    print "C (regularization param not the language):", args.C
  print "Ngram:", args.ngram
  print "Title Split:", args.titleSplit
  if args.featureSelector != 'all' and args.titleSplit:
    print "Number of title features:", constants.NUM_TITLE_FEATURES
    print "Number of text features:", constants.NUM_TEXT_FEATURES
  else:
    print "Number of features:", constants.NUM_BOTH_FEATURES
  print "Remove Stop Words:", args.noStopWords
  print "Stem Words:", args.stem
  return (feature_sel, feature_rep, model, kfolds, ngram, title_split, no_stop_words, stem)


# TODO make ngrams actually do something within util.py
feature_sel, feature_rep, model, kfolds, ngram, title_split, no_stop_words, stem = get_args()

# By default kfolds is 0 unless specified at command-line. Thus by default k-fold cross validation
# is not ran.
if kfolds:
  print 'RUNNING KFOLD CROSS-VALIDATION WITH', kfolds, 'FOLDS'
  k_folder = kfold.KFolder(kfolds)
  des_y = []
  pred_y = []
  while k_folder.has_next_fold():
    train_set, test_set = k_folder.get_next_fold()
    print '******** TESTING ON FOLD', k_folder.current_fold, '***********'
    des_i_y, pred_i_y = run_train_test.run(k_folder.current_fold, title_split, ngram, feature_sel, feature_rep, model, train_set, test_set, no_stop_words, stem)
    des_y.extend(des_i_y)
    pred_y.extend(pred_i_y)
  print '******EVALUATING OVER ALL FOLDS******'
  print classification_report(des_y, pred_y)
else:
  train_set, test_set =  load_subreddit_data.get_train_and_test_sets()
  run_train_test.run('all', title_split, ngram, feature_sel, feature_rep, model, train_set, test_set, no_stop_words, stem)

