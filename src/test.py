import run_train_test
import featurizer
import feature_selection

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

# TODO grab and pass args
run_train_test.run(True, 1, FEAT_FUNCS['binary'], SELECT_FUNCS['all'], MODELS['naive_bayes'])
