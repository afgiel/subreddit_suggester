import run_train_test
import featurizer
import feature_selection
import kfold
import load_subreddit_data

import argparse

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

FEAT_FUNCS = {
  'tfidf': featurizer.tfidf_featurize, 
  'binary': featurizer.binary_featurize,
}

SELECT_FUNCS = {
  'mi': feature_selection.select_top_n_mi_features, 
  'all': feature_selection.select_all_features
}

MODELS = {
  'naive_bayes': MultinomialNB
}

# Returns tuple of parsed command-line arguments.  Tuple contains functions
# so the conversion from string to corresponding function is also done in
# this method.
def get_args():
  parser = argparse.ArgumentParser(description="Test different learning models, feature representations" +
                                                "and parameters on the reddit datasets.")
  parser.add_argument('-model', choices=MODELS.keys(), default=MODELS.keys()[0], type=str)
  parser.add_argument('-featureSelector', choices=SELECT_FUNCS.keys(), default=SELECT_FUNCS.keys()[0], type=str)
  parser.add_argument('-featureRepresentation', choices=FEAT_FUNCS.keys(), default=FEAT_FUNCS.keys()[0], type=str)
  parser.add_argument('-kfolds', default = 0, type=int)

  args = parser.parse_args()

  model = MODELS[args.model]
  feature_sel = SELECT_FUNCS[args.featureSelector]
  feature_rep = FEAT_FUNCS[args.featureRepresentation]
  kfolds = args.kfolds
  return (feature_sel, feature_rep, model, kfolds)


# TODO make ngrams actually do something within util.py
feature_sel, feature_rep, model, kfolds = get_args()

# By default kfolds is 0 unless specified at command-line. Thus by default k-fold cross validation
# is not ran.
if kfolds:
  print 'RUNNING KFOLD CROSS-VALIDATION WITH', kfolds, 'FOLDS'
  k_folder = kfold.KFolder(kfolds)
  while k_folder.has_next_fold():
    print '******** TESTING ON FOLD', k_folder.current_fold, '***********'
    train_set, test_set = k_folder.get_next_fold()
    run_train_test.run(False, 1, feature_sel, feature_rep, model, train_set, test_set)
else:
  train_set, test_set =  load_subreddit_data.get_train_and_test_sets()
  run_train_test.run(False, 1, feature_sel, feature_rep, model, train_set, test_set)

