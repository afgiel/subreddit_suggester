

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
  print 'ACCURACY ===== ' + str(accuracy)
