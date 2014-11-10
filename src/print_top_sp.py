import csv
import os
import os.path as path
from collections import Counter

DATA_PATH = '../../reddit-top-2.5-million/data/'
self_counts = Counter()
for dirpath, dirnames, filenames in os.walk(DATA_PATH):
  for filename in filenames:
    with open(path.join(DATA_PATH, filename)) as csvfile:
      csvreader = csv.reader(csvfile)
      for row in csvreader:
        if 'self.' in row[2]:
          self_counts[filename] += 1


for sub, count in self_counts.most_common(100):
  print sub, count 
