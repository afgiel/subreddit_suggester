import constants

def evaluate(des, pred):
  num_right = 0. 
  num_wrong = 0.
  for i in range(len(des)):
    des_i = des[i] 
    pred_i = pred[i]
    if des_i == pred_i: 
      num_right += 1
    else: 
      num_wrong += 1 
  accuracy = num_right/(num_right + num_wrong)
  print '\tACCURACY: ' + str(accuracy)
  print '\t======='

  for s in range(len(constants.subreddits)): 
    tp = 0.
    fp = 0.
    fn = 0.
    for i in range(len(des)):
      des_i = des[i] 
      pred_i = pred[i]
      if des_i == s:
        if des_i == pred_i:
          tp += 1
        else: 
          fn += 1 
      elif pred_i == s:
        fp += 1  
    precision = tp/(tp + fn)  
    recall = tp/(tp + fp)
    print '\tFOR SUBREDDIT ' + constants.subreddits[s] 
    print '\tPRECISION: ' + str(precision)
    print '\tRECALL: ' + str(recall)
    print '\t======='
  print '\n--------\n'
