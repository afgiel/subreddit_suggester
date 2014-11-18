import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from gensim import corpora, models

import featurizer
import constants
import load_subreddit_data
import utils

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
  print 'PREPROCESSING TRAIN DATA'
  train_posts = [x[0] for x in train_set]   
  train_titles = [utils.tokenize(x.title) for x in train_posts]
  train_text = [utils.tokenize(x.text) for x in train_posts]
  train_labels = [x[1] for x in train_set]
  print 'MAKING DICTIONARIES'
  title_dict = corpora.Dictionary(train_titles)
  text_dict = corpora.Dictionary(train_text)
  print 'CHANGING TO VECTOR REPRESENTATION'
  train_title_corpus = [title_dict.doc2bow(x) for x in train_titles]
  train_text_corpus = [text_dict.doc2bow(x) for x in train_text]
  print 'TRAINING TFIDF'
  title_tfidf = models.TfidfModel(train_title_corpus)
  text_tfidf = models.TfidfModel(train_text_corpus)
  print 'APPLYING TFIDF'
  train_title_corpus = [title_tfidf[x] for x in train_title_corpus]
  train_text_corpus = [text_tfidf[x] for x in train_text_corpus]
  print 'TRAINING TITLE LDA' 
  title_lda = models.ldamodel.LdaModel(corpus=train_title_corpus, id2word=title_dict, num_topics=constants.NUM_TITLE_TOPICS) 
  print 'TRAINING TEXT LDA'
  text_lda = models.ldamodel.LdaModel(corpus=train_text_corpus, id2word=text_dict, num_topics=constants.NUM_TEXT_TOPICS)  
  print 'PRINTING TITLE TOPICS'
  for i in range(title_lda.num_topics):
    print title_lda.print_topic(i)
  print 'PRINTING TEXT TOPICS'
  for i in range(text_lda.num_topics):
    print text_lda.print_topic(i)
  print 'FEATURIZING TRAIN SET'
  train_title_x = featurizer.lda_featurize(title_lda, title_dict, title_tfidf, train_titles)
  train_text_x = featurizer.lda_featurize(text_lda, text_dict, text_tfidf, train_text)
  train_x = np.concatenate((train_title_x, train_text_x), axis=1)
  train_y = featurizer.make_label_vector(train_labels)
  print 'TRAINING MODEL'
  model = LogisticRegression()
  svm = SVC()
  model.fit(train_x, train_y)
  svm.fit(train_x, train_y)
  print 'EVALUATING TRAIN'
  pred_y = model.predict(train_x)
  print classification_report(train_y, pred_y)
  pred_y = svm.predict(train_x)
  print classification_report(train_y, pred_y)
  print 'PREPROCESSING TEST DATA'
  test_posts = [x[0] for x in test_set]
  test_titles = [utils.tokenize(x.title) for x in test_posts]
  test_text = [utils.tokenize(x.text) for x in test_posts]
  des_y = [x[1] for x in test_set]
  print 'FEATURIZING TEST SET'
  test_title_x = featurizer.lda_featurize(title_lda, title_dict, title_tfidf, test_titles)
  test_text_x = featurizer.lda_featurize(text_lda, text_dict, text_tfidf, test_text) 
  test_x = np.concatenate((test_title_x, test_text_x), axis=1)
  print 'MAKING PREDICTIONS'
  pred_y = model.predict(test_x)
  print 'EVALUATING TEST '
  print classification_report(des_y, pred_y)
  pred_y = svm.predict(test_x)
  print classification_report(des_y, pred_y)
  
