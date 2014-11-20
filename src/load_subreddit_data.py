import csv
import os
import os.path as path
from collections import Counter
import random

import constants
import reddit_post

def get_all_posts_and_labels(subreddit):
  post_and_labels = [] 
  subreddit_file_name = subreddit + '.csv' 
  subreddit_index = constants.subreddits.index(subreddit)
  file_path = path.join(constants.DATA_PATH_FROM_SRC, subreddit_file_name) 
  with open(file_path) as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
      if 'self.' in row[2]: 
        title = row[4]        
        text = row[10]
        post = reddit_post.RedditPost(title, text) 
        post_and_labels.append((post, subreddit_index))
  return post_and_labels

def get_train_and_test_sets():
  train_set = [] 
  test_set = [] 
  print 'LOADING DATA FROM CSVS'
  for subreddit in constants.subreddits:
    sub_all = get_all_posts_and_labels(subreddit) 
    num_posts = len(sub_all)  
    sub_train = random.sample(sub_all, int(num_posts*.9)) 
    sub_test = [x for x in sub_all if x not in sub_train]
    train_set.extend(sub_train)
    test_set.extend(sub_test)
  return train_set, test_set
