import random
import numpy as np
from sklearn.metrics import classification_report
from gensim import corpora, models
import math

import featurizer
import constants
import load_subreddit_data
import utils

def calc_score(model, dictionary, text):
  text_vec = dictionary.doc2bow(text)
  gamma, _ = model.inference([text_vec])
  topic_dist = gamma[0] / sum(gamma[0])
  score = 0.0
  print model.show_topics()
  for word in text:   
    for topic in range(constants.NUM_TOPICS):
      if word in dictionary.keys():
        beta = model.state.get_lambda()[topic]  
        beta = beta / beta.sum()
        print word 
        print dictionary[word]
        print beta[dictionary[word]]
        score += beta[dictionary.get(word)]*topic_dist[topic]
  return score 

def lda_predict(lda_models, dictionary, texts): 
  labels = []
  for text in texts:
    bestIndex = -1
    bestScore = 0.0
    for i in range(len(lda_models)): 
      score = calc_score(lda_models[i], dictionary, text) 
      if score > bestScore:
        bestScore = score
        bestIndex = i
    labels.append(bestIndex)
  return labels 

def run():
  test_set = []
  train_set = []
  print 'LOADING DATA FROM CSVS'
  data_map = {}
  for subreddit in constants.subreddits:
    sub_all = load_subreddit_data.get_all_posts_and_labels(subreddit) 
    num_posts = len(sub_all)  
    sub_train = random.sample(sub_all, int(num_posts*.9)) 
    sub_test = [x for x in sub_all if x not in sub_train]
    data_map[subreddit] = sub_train
    train_set.extend(sub_train)
    test_set.extend(sub_test)
  dictionary = corpora.Dictionary() 
  print 'MAKING DICTIONARY'
  for subreddit in constants.subreddits:
    train_posts = [x[0] for x in data_map[subreddit]]   
    train_text = [utils.tokenize(x.title + ' ' + x.text, 1, True, False) for x in train_posts]
    data_map[subreddit] = train_text 
    text_dict = dictionary.add_documents(train_text)
  print 'CHANGING TO VECTOR REPRESENTATION'
  for subreddit in constants.subreddits:
    data_map[subreddit] = [dictionary.doc2bow(x) for x in data_map[subreddit]]
  lda_models = {} 
  for i in range(len(constants.subreddits)):
    subreddit = constants.subreddits[i]
    print 'TRAINING ' + subreddit + ' LDA MODEL'
    lda_models[i] = models.ldamodel.LdaModel(corpus=data_map[subreddit], id2word=dictionary, num_topics=constants.NUM_TOPICS)
  train_y = [x[1] for x in train_set]
  train_set = [x[0] for x in train_set]
  train_text = [utils.tokenize(x.title + ' ' + x.text, 1, True, False) for x in train_set]
  print 'EVALUATING TRAIN'
  pred_y = lda_predict(lda_models, dictionary, train_text)
  print classification_report(train_y, pred_y)
  print 'PREPROCESSING TEST DATA'
  test_posts = [x[0] for x in test_set]
  test_text = [utils.tokenize(x.title + ' ' + x.text, 1, True, False) for x in test_posts]
  des_y = [x[1] for x in test_set]
  print 'MAKING PREDICTIONS'
  pred_y = model.lda_predict(lda_models, dictionary, test_text)
  print 'EVALUATING TEST '
  print classification_report(des_y, pred_y)
  
