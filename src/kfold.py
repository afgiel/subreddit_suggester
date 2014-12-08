import random

import constants
import load_subreddit_data

# Utility class for performing k-fold cross-validation.
# Effectively performs k-folding on data from an individual
# subreddit which guarantees that each fold contains the same
# number of posts from a given subreddit.
class KFolder():

  SEED = .69

  def __init__(self, num_folds = 10,  shuffle = True):
    self.num_folds = num_folds
    self.fold_intervals = []
    # Stores a sublist of reddit posts for each subreddit.
    self.dataset = []
    for subreddit in constants.subreddits:
      subreddit_data = load_subreddit_data.get_all_posts_and_labels(subreddit)
      if shuffle:
        random.shuffle(subreddit_data, lambda: SEED)
      self.dataset.append(subreddit_data)
      self.fold_intervals.append(len(subreddit_data)/num_folds)
    self.current_fold = 0

  def get_test_fold_interval(self, data_idx):
    test_start_idx = self.current_fold * self.fold_intervals[data_idx]
    test_end_idx = (self.current_fold + 1) * self.fold_intervals[data_idx]
    return (test_start_idx, test_end_idx)

  # Returns a train list of reddit posts and test list of reddit posts.
  def get_next_fold(self):
    train_fold = []
    test_fold = []
    for data_idx, subdata in enumerate(self.dataset):
      test_start_idx, test_end_idx = self.get_test_fold_interval(data_idx)
      train_fold.extend(subdata[0:test_start_idx])
      test_fold.extend(subdata[test_start_idx:test_end_idx])
      train_fold.extend(subdata[test_end_idx:])
    self.current_fold += 1
    return (train_fold, test_fold)

  def has_next_fold(self):
    return self.current_fold < self.num_folds
