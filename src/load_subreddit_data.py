import csv
import os
import os.path as path
from collections import Counter

import constants
import reddit_post

def get_all_posts_and_labels(subreddit):
  post_and_labels = [] 
  subreddit_file_name = subreddit + '.csv' 
  file_path = path.join(constants.DATA_PATH_FROM_SRC, subreddit_file_name) 
  with open(file_path) as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
      if 'self.' in row[2]: 
        title = row[4]        
        text = row[10]
        post = reddit_post.RedditPost(title, text) 
        subreddit_index = constants.subreddits.index(subreddit)
        post_and_labels.append((post, subreddit_index))
  return post_and_labels
